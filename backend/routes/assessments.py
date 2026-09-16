"""
Assessment routes - prediction history and listing.

The actual ML model in backend.model_service is the primary and only
prediction engine. Assessment results are persisted to MongoDB and
high-risk predictions can create alerts.
"""

from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from backend.models.assessment import (
    AssessmentCreate,
    AssessmentInDB,
    AssessmentResponse,
    AssessmentDetailResponse,
    AssessmentListQuery,
    AssessmentListResponse,
)
from backend.schemas import (
    MachineInput,
    PredictionResponse,
    RecommendationDecisionRequest,
    RecommendationDecisionResponse,
)
from backend.model_service import predict_machine
from backend.db.mongodb import get_collection


router = APIRouter(
    prefix="/api/v1/assessments",
    tags=["Assessments"],
)


@router.post("/predict", response_model=PredictionResponse)
async def predict(machine: MachineInput):
    """
    Run the existing ML model, persist the assessment, and create an
    alert when the predicted health status is High Risk or Critical.
    """

    from backend.services.assessment import create_assessment
    from backend.services.alert import create_alert_from_assessment

    # predict_machine() performs synchronous ML inference.
    # Running it in a worker thread prevents it from blocking FastAPI's
    # async event loop when multiple requests arrive.
    result = await run_in_threadpool(
        predict_machine,
        product_type=machine.product_type,
        air_temperature=machine.air_temperature,
        process_temperature=machine.process_temperature,
        rotational_speed=machine.rotational_speed,
        torque=machine.torque,
        tool_wear=machine.tool_wear,
    )

    # The prediction comes directly from our existing FastAPI ML model.
    result["source"] = "fastapi"

    prediction = PredictionResponse(**result)

    # Store the input values and prediction in MongoDB.
    assessment = AssessmentCreate(
        machine_id=machine.machine_id or "UNKNOWN",
        inputs=machine.model_dump(exclude_none=True),
        prediction=prediction,
        source="fastapi",
    )

    assessment_doc = await create_assessment(assessment)

    # Automatically create an alert for high-risk predictions.
    if prediction.health_status in ("High Risk", "Critical"):
        await create_alert_from_assessment(assessment_doc)

    return prediction


@router.get("", response_model=AssessmentListResponse)
async def list_assessments(
    query: AssessmentListQuery = Depends(),
):
    """List assessments with filters and pagination."""

    assessments_collection = get_collection("assessments")

    # Build MongoDB filter.
    q = {}

    if query.machine_id:
        q["machine_id"] = query.machine_id

    if query.health_status:
        q["prediction.health_status"] = query.health_status

    if query.risk_level:
        q["prediction.risk_level"] = query.risk_level

    if query.date_from or query.date_to:
        date_q = {}

        if query.date_from:
            date_q["$gte"] = query.date_from

        if query.date_to:
            date_q["$lte"] = query.date_to

        q["ts"] = date_q

    total = await assessments_collection.count_documents(q)
    total_pages = (total + query.page_size - 1) // query.page_size

    cursor = (
        assessments_collection
        .find(q)
        .sort("ts", -1)
        .skip((query.page - 1) * query.page_size)
        .limit(query.page_size)
    )

    items = []

    async for doc in cursor:
        # Convert MongoDB ObjectId to string.
        doc["_id"] = str(doc["_id"])
# from_db() expects an AssessmentInDB object, not a raw dict.
        assessment = AssessmentInDB(**doc)

        items.append(
            AssessmentResponse.from_db(assessment)
        )

    return AssessmentListResponse(
        items=items,
        total=total,
        page=query.page,
        page_size=query.page_size,
        total_pages=total_pages,
    )


@router.get("/{assessment_id}", response_model=AssessmentDetailResponse)
async def get_assessment(assessment_id: str):
    """Get a single assessment by MongoDB ID."""

    # Validate the ObjectId before querying MongoDB.
    if not ObjectId.is_valid(assessment_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid assessment ID",
        )

    doc = await get_collection("assessments").find_one(
        {"_id": ObjectId(assessment_id)}
    )

    if not doc:
        raise HTTPException(
            status_code=404,
            detail="Assessment not found",
        )

    # Convert MongoDB ObjectId to the string expected by the Pydantic model.
    doc["_id"] = str(doc["_id"])

    # Convert the database document into the correct internal model.
    assessment = AssessmentInDB(**doc)

    # Build the detailed API response from the database model.
    return AssessmentDetailResponse(
        id=assessment.id,
        machine_id=assessment.machine_id,
        ts=assessment.ts,
        source=assessment.source,
        failure_probability=assessment.prediction.failure_probability,
        health_status=assessment.prediction.health_status,
        risk_level=assessment.prediction.risk_level,
        predicted_class=assessment.prediction.predicted_class,
        failure_mode=assessment.prediction.failure_mode,
        anomaly_percentile=assessment.prediction.anomaly_percentile,
        inputs=assessment.inputs,
        prediction=assessment.prediction,
    )


@router.post(
    "/{assessment_id}/recommendation-decision",
    response_model=RecommendationDecisionResponse,
)
async def recommendation_decision(
    assessment_id: str,
    decision: RecommendationDecisionRequest,
):
    """
    Persist the operator's recommendation choice on an assessment.

    The choice is stored as `selectedRecommendationId` together
    with a `recommendationDecisionTs` ISO timestamp on the
    assessment document in MongoDB.
    """

    # Validate the ObjectId before querying MongoDB.
    if not ObjectId.is_valid(assessment_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid assessment ID",
        )

    decision_ts = datetime.now(timezone.utc)

    try:
        assessments_collection = get_collection("assessments")

        existing = await assessments_collection.find_one(
            {"_id": ObjectId(assessment_id)}
        )

        if not existing:
            raise HTTPException(
                status_code=404,
                detail="Assessment not found",
            )

        await assessments_collection.update_one(
            {"_id": ObjectId(assessment_id)},
            {
                "$set": {
                    "selectedRecommendationId": decision.selectedOptionId,
                    "recommendationDecisionTs": decision_ts.isoformat(),
                }
            },
        )
    except HTTPException:
        # Re-raise client errors (e.g. 404) untouched.
        raise
    except Exception:
        # MongoDB outages must not crash the service.
        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        )

    return RecommendationDecisionResponse(
        assessmentId=assessment_id,
        selectedOptionId=decision.selectedOptionId,
        updatedAt=decision_ts.isoformat(),
    )