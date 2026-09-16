"""
FastAPI Backend
---------------
Main entry point for the Predictive Maintenance backend.

Responsibilities:
1. Start the FastAPI application
2. Provide a health-check endpoint
3. Receive machine sensor data
4. Validate the input
5. Call the ML service
6. Return the combined prediction to the frontend
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import MachineInput, PredictionResponse
from backend.model_service import predict_machine


# =========================================================
# CREATE FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Predictive Maintenance Agent",
    description=(
        "Backend API for the manufacturing "
        "predictive maintenance system"
    ),
    version="1.0.0"
)


# =========================================================
# CORS CONFIGURATION
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():
    """
    Check whether the backend is running.
    """

    return {
        "status": "healthy",
        "message": "Predictive Maintenance backend is running"
    }


# =========================================================
# PREDICT MACHINE CONDITION
# =========================================================

@app.post(
    "/v1/predict",
    response_model=PredictionResponse
)
def predict(machine: MachineInput):
    """
    Receive machine sensor data and run all ML components.

    Flow:

    Frontend
        ↓
    Input validation
        ↓
    Main failure model
        ↓
    Anomaly model
        ↓
    Failure-mode models
        ↓
    Combined JSON response
        ↓
    Frontend
    """

    # -----------------------------------------------------
    # Send the validated machine data to the ML service.
    # -----------------------------------------------------

    result = predict_machine(
        product_type=machine.product_type,
        air_temperature=machine.air_temperature,
        process_temperature=machine.process_temperature,
        rotational_speed=machine.rotational_speed,
        torque=machine.torque,
        tool_wear=machine.tool_wear
    )


    # -----------------------------------------------------
    # Return the combined ML result.
    #
    # FastAPI automatically converts this dictionary
    # into JSON.
    # -----------------------------------------------------

    return result