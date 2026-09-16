"""
Model Service
-------------
This file is responsible for loading the ML components
and combining their results.

The three ML components are:

1. Main failure prediction model
   -> Predicts overall machine failure probability.

2. Anomaly detection model
   -> Determines how unusual the current machine condition is.

3. Failure-mode model
   -> Estimates probabilities for supported failure modes
      such as HDF, PWF and OSF.

The returned response also exposes:
- condition evidence (reusing src/condition_engine rules)
- rich per-mode records (reusing FailureModeComponent.rank)
- feature contributions (leave-one-out on the frozen model)
- model metadata, decision threshold and decision-support notice
"""

import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml

from backend.decision_service import (
    get_health_status,
    get_maintenance_action
)

from backend.services.llm_client import get_llm_recommendations

from src.condition_engine import condition_evidence


logger = logging.getLogger(__name__)


# =========================================================
# 1. FIND PROJECT ROOT
# =========================================================

# This file is:
#
# Predictive-Maintenance-Agent/
# └── backend/
#     └── model_service.py
#
# Therefore .parent.parent gives us the project root.

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =========================================================
# 2. MAKE src/ AVAILABLE
# =========================================================

# The saved Joblib models depend on Python modules inside
# our project's src/ directory.
#
# For example:
# src.feature_engineering
# src.anomaly_detection
# src.failure_mode_model
#
# Adding the project root allows Python to find those modules.

sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# 3. MODEL PATHS
# =========================================================

MAIN_MODEL_PATH = (
    PROJECT_ROOT / "models" / "model_pipeline.joblib"
)

ANOMALY_MODEL_PATH = (
    PROJECT_ROOT / "models" / "anomaly_model.joblib"
)

FAILURE_MODE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "failure_mode_pipeline.joblib"
)


# =========================================================
# 4. LOAD MODELS
# =========================================================

logger.info("Loading predictive maintenance models...")

# Overall failure prediction model
main_model = joblib.load(MAIN_MODEL_PATH)

# Anomaly detection component
anomaly_model = joblib.load(ANOMALY_MODEL_PATH)

# Failure-mode component
failure_mode_model = joblib.load(FAILURE_MODE_MODEL_PATH)

logger.info("All predictive maintenance models loaded successfully!")


# =========================================================
# 5. LOAD MODEL METADATA AND DECISION CONFIGURATION
# =========================================================

# ---------------------------------------------------------
# Load the decision threshold selected during model training.
#
# IMPORTANT:
# We should NOT automatically use model.predict() here,
# because the trained model has a custom operating threshold
# selected on validation data.
# ---------------------------------------------------------

THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold.json"

with open(THRESHOLD_PATH, "r") as file:
    threshold_config = json.load(file)

FAILURE_THRESHOLD = threshold_config["threshold"]

logger.info(
    "Failure decision threshold loaded: %.4f",
    FAILURE_THRESHOLD
)

# ---------------------------------------------------------
# Load the trained model version from the metadata file.
# ---------------------------------------------------------

METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

with open(METADATA_PATH, "r") as file:
    model_metadata = json.load(file)

MODEL_VERSION = model_metadata["model_version"]

# ---------------------------------------------------------
# Load the decision-support notice from the maintenance
# rules configuration (single source of truth).
# ---------------------------------------------------------

MAINTENANCE_RULES_PATH = (
    PROJECT_ROOT / "configs" / "maintenance_rules.yaml"
)

with open(MAINTENANCE_RULES_PATH, "r", encoding="utf-8") as file:
    maintenance_rules = yaml.safe_load(file)

DECISION_SUPPORT_NOTICE = maintenance_rules["notice"]

# ---------------------------------------------------------
# Load the baseline (training median) feature values.
#
# These are used ONLY for the leave-one-out attribution
# shown in contributing_features. They are descriptive
# reference values, not model parameters.
# ---------------------------------------------------------

BASELINE_PATH = PROJECT_ROOT / "models" / "explanation_baseline.json"

with open(BASELINE_PATH, "r") as file:
    BASELINE = json.load(file)


# =========================================================
# 6. POLICY MAPPINGS
# =========================================================
# These are fixed display vocabulary rules applied by the
# application. They do NOT come from the ML models.

# decision_service returns one of Normal / Warning / High / Critical.
# The frontend contract uses "High Risk" instead of "High".
HEALTH_STATUS_DISPLAY = {
    "Normal": "Normal",
    "Warning": "Warning",
    "High": "High Risk",
    "Critical": "Critical",
}

# Risk level uses the exact same probability bands as health_status.
RISK_LEVEL = {
    "Normal": "Low",
    "Warning": "Medium",
    "High": "High",
    "Critical": "Critical",
}

# Urgency is a simple action-priority label for the UI.
URGENCY = {
    "Normal": "Routine",
    "Warning": "Scheduled",
    "High": "Urgent",
    "Critical": "Immediate",
}

# Raw sensor features used for leave-one-out attribution.
# These are the numeric inputs of the trained pipeline.
CONTRIBUTING_FEATURES = [
    ("air_temperature", "Air temperature"),
    ("process_temperature", "Process temperature"),
    ("rotational_speed", "Rotational speed"),
    ("torque", "Torque"),
    ("tool_wear", "Tool wear"),
]


# =========================================================
# 7. HELPERS
# =========================================================

def _build_condition_evidence(evidence_rows: list) -> list:
    """
    Convert the raw condition-engine records into plain,
    human-readable strings.

    `evidence_rows` comes from src.condition_engine.condition_evidence()
    and is reused as-is (no duplicate rule definitions here).
    """

    entries = []
    for ev in evidence_rows:

        # measured can be empty (e.g. RNF has no sensor cause)
        measured = ev.get("measured") or {}
        if measured:
            measured_text = "; ".join(
                f"{key}={value:g}"
                for key, value in measured.items()
            )
        else:
            measured_text = "n/a"

        entries.append(
            f"{ev['mode']} ({ev['kind']}): {ev['rule']} "
            f"- triggered={ev['triggered']} - measured: {measured_text}"
        )

    return entries


def _map_likely_failure_modes(rank_items: list) -> list:
    """
    Map FailureModeComponent.rank() output into the response
    records. rank() is already computed by the existing model
    component; we only reshape it here.
    """

    modes = []
    for item in rank_items:

        probability = item.get("probability")

        modes.append({
            "mode": item["mode"],
            "name": item["name"],
            "probability": (
                None if probability is None else float(probability)
            ),
            "score": float(item["score"]),
            "score_kind": item["score_kind"],
            "supporting_condition": bool(item["supporting_condition"]),
            "note": item.get("limitation"),
        })

    return modes


def _compute_contributing_features(input_data: pd.DataFrame) -> list:
    """
    Compute per-feature contribution to the failure probability.

    This is NOT a SHAP value. Each raw sensor value is replaced
    by its training baseline (median) and the frozen model's
    failure probability is recomputed. The absolute change is
    reported as `magnitude` and the sign of the change as
    `direction`. All values come from the real trained model.
    """

    # Baseline failure probability for the actual input row.
    actual_probability = float(
        main_model.predict_proba(input_data)[0, 1]
    )

    features = []
    for feature, label in CONTRIBUTING_FEATURES:

        # One-at-a-time replacement with the training baseline.
        perturbed_row = input_data.iloc[[0]].copy()
        perturbed_row[feature] = float(BASELINE[feature])

        perturbed_probability = float(
            main_model.predict_proba(perturbed_row)[0, 1]
        )

        # Positive delta  -> the current value raises risk.
        # Negative delta  -> the current value lowers risk.
        delta = actual_probability - perturbed_probability

        features.append({
            "feature": feature,
            "label": label,
            "value": float(input_data.iloc[0][feature]),
            "magnitude": round(abs(delta), 6),
            "sign": 1 if delta >= 0 else -1,
            "direction": "increases" if delta >= 0 else "decreases",
            "method": "leave_one_out_replacement_with_training_baseline",
        })

    # Order by strongest contribution first.
    features.sort(key=lambda item: item["magnitude"], reverse=True)

    return features


# =========================================================
# 8. PREDICTION FUNCTION
# =========================================================

def predict_machine(
    product_type: str,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float
):
    """
    Run all ML components and the decision layer
    for one machine.

    Flow:

        Machine data
             ↓
        ML predictions
             ↓
        Decision layer
             ↓
        Complete application result
    """

    # Start the latency measurement (covers the ML work below).
    start_seconds = time.perf_counter()

    # =====================================================
    # CREATE INPUT DATAFRAME
    # =====================================================

    input_data = pd.DataFrame([
        {
            "product_type": product_type,
            "air_temperature": air_temperature,
            "process_temperature": process_temperature,
            "rotational_speed": rotational_speed,
            "torque": torque,
            "tool_wear": tool_wear
        }
    ])


    # =====================================================
    # 1. OVERALL FAILURE PREDICTION
    # =====================================================

    probabilities = main_model.predict_proba(input_data)

    # Probability that the machine will fail
    failure_probability = float(probabilities[0][1])

    # ---------------------------------------------------------
    # The trained model uses a validation-selected threshold
    # instead of the default 0.50 threshold.
    # ---------------------------------------------------------
    predicted_failure = (
        failure_probability >= FAILURE_THRESHOLD
    )

    # =====================================================
    # 2. ANOMALY DETECTION
    # =====================================================

    raw_score, percentile, is_anomaly = anomaly_model.score(
        input_data
    )

    anomaly_score = float(raw_score[0])

    # Reported as a 0.0 - 100.0 percentile rank to match the
    # frontend contract (0-100 scale).
    anomaly_percentile = float(percentile[0]) * 100.0

    is_anomalous = bool(is_anomaly[0])


    # =====================================================
    # 3. CONDITION EVIDENCE (reuses the existing rule engine)
    # =====================================================

    evidence_rows = condition_evidence(input_data.iloc[0])

    condition_evidence_list = _build_condition_evidence(
        evidence_rows
    )


    # =====================================================
    # 4. FAILURE-MODE PREDICTION
    # =====================================================

    # probabilities() -> only modes with standalone models.
    # rank()          -> richer records for every mode.
    mode_probabilities = failure_mode_model.probabilities(
        input_data
    )

    failure_modes = {}

    for mode, values in mode_probabilities.items():
        failure_modes[mode] = float(values[0])

    rank_items = failure_mode_model.rank(
        input_data,
        evidence_rows
    )

    likely_failure_modes = _map_likely_failure_modes(rank_items)


    # =====================================================
    # 5. DETERMINE MACHINE HEALTH
    # =====================================================

    backend_health_status = get_health_status(
        failure_probability
    )

    # Frontend contract uses "High Risk" instead of "High".
    health_status = HEALTH_STATUS_DISPLAY[backend_health_status]

    risk_level = RISK_LEVEL[backend_health_status]

    urgency = URGENCY[backend_health_status]


    # =====================================================
    # 6. GENERATE EXPLANATION AND RECOMMENDATION
    # =====================================================

    maintenance_info = get_maintenance_action(
        failure_modes=failure_modes,
        predicted_failure=predicted_failure,
        tool_wear=tool_wear
    )

    if predicted_failure and maintenance_info["failure_mode"]:
        predicted_class = maintenance_info["failure_mode"]
    elif not predicted_failure:
        predicted_class = "NO FAILURE"
    else:
        predicted_class = "FAILURE"


    # =====================================================
    # 6b. LLM RECOMMENDATION OPTIONS (OpenRouter, optional)
    # =====================================================

    # The LLM client never raises: on missing API key, timeout,
    # HTTP error or parse failure it returns None and the
    # deterministic recommendation above remains the fallback.
    # The LLM_TIMEOUT_SECONDS setting bounds the latency impact.
    llm_recommendations = get_llm_recommendations(
        product_type=product_type,
        readings={
            "air_temperature": air_temperature,
            "process_temperature": process_temperature,
            "rotational_speed": rotational_speed,
            "torque": torque,
            "tool_wear": tool_wear,
        },
        failure_probability=failure_probability,
        failure_mode_code=(
            maintenance_info["failure_mode"]
        ),
        failure_mode_name=(
            maintenance_info["failure_mode_name"]
        ),
    )


    # =====================================================
    # 7. FEATURE CONTRIBUTIONS (leave-one-out on the model)
    # =====================================================

    contributing_features = _compute_contributing_features(
        input_data
    )


    # =====================================================
    # 8. PREDICTION IDENTIFIERS AND TIMING
    # =====================================================

    prediction_id = uuid.uuid4().hex[:12]

    prediction_timestamp = datetime.now(timezone.utc).isoformat()

    latency_ms = round(
        (time.perf_counter() - start_seconds) * 1000,
        2
    )


    # =====================================================
    # 9. COMBINE EVERYTHING
    # =====================================================

    result = {
        # Overall prediction
        "failure_probability": failure_probability,
        "predicted_failure": predicted_failure,

        # Anomaly information
        "anomaly_score": anomaly_score,
        "anomaly_percentile": anomaly_percentile,
        "is_anomaly": is_anomalous,

        # Failure modes (probabilities only)
        "failure_modes": failure_modes,

        # Decision layer
        "health_status": health_status,

        "failure_mode": (
            maintenance_info["failure_mode"]
        ),

        "failure_mode_name": (
            maintenance_info["failure_mode_name"]
        ),

        "explanation": (
            maintenance_info["explanation"]
        ),

        "maintenance_recommendation": (
            maintenance_info[
                "maintenance_recommendation"
            ]
        ),

        # Extended frontend contract fields
        # -------------------------------------------------
        "predicted_class": predicted_class,

        "recommended_maintenance_action": (
            maintenance_info["maintenance_recommendation"]
        ),

        "decision_threshold": float(FAILURE_THRESHOLD),

        "risk_level": risk_level,

        "urgency": urgency,

        "likely_failure_modes": likely_failure_modes,

        "contributing_features": contributing_features,

        "condition_evidence": condition_evidence_list,

        "model_version": MODEL_VERSION,

        "prediction_id": prediction_id,

        "prediction_timestamp": prediction_timestamp,

        "latency_ms": latency_ms,

        "decision_support_notice": DECISION_SUPPORT_NOTICE,
    }

    # Optional LLM-backed recommendation options: attached only
    # when the LLM client returned usable options.
    if llm_recommendations:
        result["recommendationOptions"] = (
            llm_recommendations["recommendationOptions"]
        )

        result["selectedRecommendationId"] = (
            llm_recommendations["selectedRecommendationId"]
        )

    return result


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    result = predict_machine(
        product_type="L",
        air_temperature=298.1,
        process_temperature=308.6,
        rotational_speed=1450,
        torque=45,
        tool_wear=120
    )

    print("\nCombined prediction result:")

    print(result)