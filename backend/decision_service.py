"""
Decision Service
----------------
This file converts raw ML predictions into application-level
decisions that can be displayed by the frontend.

The ML model tells us WHAT it predicts.

The decision service tells us HOW the application should
interpret that prediction.
"""


# =========================================================
# RISK THRESHOLDS
# =========================================================

# These values represent the probability boundaries for
# the machine health status.
#
# IMPORTANT:
# The ML model returns probability as a value between 0 and 1.
#
# Example:
# 0.75 = 75%
#
# These thresholds are the pilot operating thresholds
# supplied with our project.

WARNING_THRESHOLD = 0.10
HIGH_THRESHOLD = 0.40
CRITICAL_THRESHOLD = 0.80


# =========================================================
# HEALTH STATUS
# =========================================================

def get_health_status(failure_probability: float) -> str:
    """
    Convert failure probability into a health status.

    Rules:

        probability < 0.10
            -> Normal

        0.10 <= probability < 0.40
            -> Warning

        0.40 <= probability < 0.80
            -> High

        probability >= 0.80
            -> Critical
    """

    # -----------------------------------------------------
    # Critical condition
    # -----------------------------------------------------

    if failure_probability >= CRITICAL_THRESHOLD:
        return "Critical"


    # -----------------------------------------------------
    # High-risk condition
    # -----------------------------------------------------

    if failure_probability >= HIGH_THRESHOLD:
        return "High"


    # -----------------------------------------------------
    # Warning condition
    # -----------------------------------------------------

    if failure_probability >= WARNING_THRESHOLD:
        return "Warning"


    # -----------------------------------------------------
    # Normal condition
    # -----------------------------------------------------

    return "Normal"


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    # Test different probabilities to make sure our
    # decision logic works correctly.

    test_probabilities = [
        0.05,
        0.20,
        0.50,
        0.90
    ]

    for probability in test_probabilities:

        status = get_health_status(probability)

        print(
            f"Failure probability: {probability:.0%}"
            f" -> Health status: {status}"
        )


        # =========================================================
# FAILURE MODE INFORMATION
# =========================================================

# These descriptions explain what each supported failure
# mode generally represents in the AI4I dataset.
#
# IMPORTANT:
# These are likely failure mechanisms, not guaranteed
# physical root causes in a real manufacturing machine.

FAILURE_MODE_INFO = {

    "TWF": {
        "name": "Tool Wear Failure",
        "explanation": (
            "The machine condition indicates potentially "
            "high or critical tool wear."
        ),
        "recommendation": (
            "Inspect the tool condition and replace the tool "
            "if wear is excessive."
        )
    },

    "HDF": {
        "name": "Heat Dissipation Failure",
        "explanation": (
            "The operating condition may indicate insufficient "
            "heat dissipation, especially when the temperature "
            "difference is low and rotational speed is low."
        ),
        "recommendation": (
            "Inspect the cooling system and heat-dissipation "
            "conditions."
        )
    },

    "PWF": {
        "name": "Power Failure",
        "explanation": (
            "The operating condition may indicate abnormal "
            "mechanical power."
        ),
        "recommendation": (
            "Inspect machine load, power delivery, and the "
            "drive system."
        )
    },

    "OSF": {
        "name": "Overstrain Failure",
        "explanation": (
            "The operating condition may indicate excessive "
            "mechanical strain on the machine or tool."
        ),
        "recommendation": (
            "Inspect tool condition and check for excessive "
            "mechanical load."
        )
    },

    "RNF": {
        "name": "Random Failure",
        "explanation": (
            "This failure mode does not have a deterministic "
            "sensor-based cause in the dataset."
        ),
        "recommendation": (
            "Perform a broader machine inspection rather than "
            "assuming a specific sensor-based root cause."
        )
    }
}


# =========================================================
# FIND MOST LIKELY FAILURE MODE
# =========================================================

def get_failure_mode(failure_modes: dict) -> str | None:
    """
    Find the strongest supported failure-mode signal.

    If all available probabilities are zero, we return None
    instead of incorrectly selecting the first mode.

    Example:

        {
            "HDF": 0.20,
            "PWF": 0.75,
            "OSF": 0.10
        }

    returns:

        "PWF"
    """

    # -----------------------------------------------------
    # No failure-mode information available
    # -----------------------------------------------------

    if not failure_modes:
        return None

    # -----------------------------------------------------
    # Find the mode with the highest probability.
    # -----------------------------------------------------

    mode = max(
        failure_modes,
        key=failure_modes.get
    )

    # -----------------------------------------------------
    # IMPORTANT:
    # If every available mode has probability 0, there is
    # no evidence for a specific failure mode.
    #
    # Without this check, Python would select HDF simply
    # because it appears first.
    # -----------------------------------------------------

    if failure_modes[mode] <= 0:
        return None

    return mode


# =========================================================
# CREATE EXPLANATION AND RECOMMENDATION
# =========================================================

def get_maintenance_action(
    failure_modes: dict,
    predicted_failure: bool,
    tool_wear: float | None = None
) -> dict:
    """
    Convert failure-mode predictions into a human-readable
    explanation and maintenance recommendation.
    """

    # -----------------------------------------------------
    # If the main model does NOT predict failure, we don't
    # want to alarm the maintenance team unnecessarily.
    # -----------------------------------------------------

    if not predicted_failure:

        return {
            "failure_mode": None,
            "failure_mode_name": None,
            "explanation": (
                "The predictive model does not currently "
                "indicate a machine failure."
            ),
            "maintenance_recommendation": (
                "No immediate maintenance action is indicated. "
                "Continue monitoring the machine."
            )
        }


    # -----------------------------------------------------
    # Prefer a failure mode produced by the trained
    # failure-mode models (HDF / PWF / OSF) whenever one
    # carries evidence (probability > 0). This must run
    # BEFORE the wear-based TWF heuristic so a strong model
    # signal is never masked by tool wear alone.
    # -----------------------------------------------------

    mode = get_failure_mode(failure_modes)

    if (
        mode is not None
        and mode in FAILURE_MODE_INFO
        and failure_modes.get(mode, 0) > 0
    ):

        info = FAILURE_MODE_INFO[mode]

        return {
            "failure_mode": mode,
            "failure_mode_name": info["name"],
            "explanation": info["explanation"],
            "maintenance_recommendation": info["recommendation"]
        }


    # -----------------------------------------------------
    # TWF CONDITION CHECK
    # -----------------------------------------------------
    # The trained failure-mode component does not contain
    # a standalone TWF model because TWF has very few
    # training examples.
    #
    # Therefore, we do NOT create a fake TWF probability.
    # Instead, when a failure is predicted and tool wear
    # is very high, we treat it as a condition-based TWF
    # indication (fallback, only if no modeled mode has
    # strong evidence).
    #
    # The AI4I dataset associates TWF with high tool wear.
    # -----------------------------------------------------

    if predicted_failure and tool_wear is not None:
        if tool_wear >= 200:
            info = FAILURE_MODE_INFO["TWF"]
            return {
                "failure_mode": "TWF",
                "failure_mode_name": info["name"],
                "explanation": info["explanation"],
                "maintenance_recommendation": info["recommendation"]
            }


    # -----------------------------------------------------
    # If there is no supported failure-mode prediction,
    # provide a general recommendation.
    # -----------------------------------------------------

    if mode is None or mode not in FAILURE_MODE_INFO:

        return {
            "failure_mode": None,
            "failure_mode_name": None,
            "explanation": (
                "A machine failure is predicted, but the "
                "available failure-mode models do not provide "
                "enough evidence for a specific failure mode."
            ),
            "maintenance_recommendation": (
                "Perform a general machine inspection and "
                "investigate the operating conditions."
            )
        }


    # -----------------------------------------------------
    # Get information for the selected failure mode.
    # -----------------------------------------------------

    info = FAILURE_MODE_INFO[mode]


    return {
        "failure_mode": mode,
        "failure_mode_name": info["name"],
        "explanation": info["explanation"],
        "maintenance_recommendation": info["recommendation"]
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Test 1: Normal machine
    # -----------------------------------------------------

    result = get_maintenance_action(
        failure_modes={
            "HDF": 0.00,
            "PWF": 0.00,
            "OSF": 0.00
        },
        predicted_failure=False
    )

    print("\nNormal machine:")
    print(result)


    # -----------------------------------------------------
    # Test 2: Machine predicted to fail
    #
    # PWF has the highest probability.
    # -----------------------------------------------------

    result = get_maintenance_action(
        failure_modes={
            "HDF": 0.20,
            "PWF": 0.75,
            "OSF": 0.10
        },
        predicted_failure=True
    )

    print("\nFailure predicted:")
    print(result)