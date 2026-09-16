"""
LLM client - OpenRouter-backed maintenance recommendation options.

Calls the OpenRouter chat-completions API (OpenAI-compatible) over
httpx and returns structured recommendation options for the frontend.

Contract:

- Returns None when OPENROUTER_API_KEY is empty or when ANY
  exception occurs (timeout, HTTP error, invalid payload,
  parse failure).
- NEVER raises. The caller falls back to the deterministic
  recommendation text produced by the decision layer.
- Stateless by design: no cache, no shared session state.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from backend.config import settings


logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

# Exactly 4 options are expected, matching the frontend contract.
MAX_OPTIONS = 4

ALLOWED_EFFORTS = {"low", "medium", "high"}


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """You are an expert predictive maintenance engineer working with the AI4I 2020 Predictive Maintenance dataset for a milling machine.

Failure modes:
- TWF: tool wear failure
- HDF: heat dissipation failure
- PWF: power failure
- OSF: overstrain failure
- RNF: random failure (no deterministic sensor cause)

Return STRICT JSON only, with no markdown, no commentary and no code fences, using exactly this shape:
{"recommendationOptions": [{"id": "<short-unique-id>", "label": "<short action label>", "description": "<one sentence explanation>", "effort": "low|medium|high"}], "recommendedOptionId": "<id of the best option>"}

Rules:
- Return exactly 4 recommendation options: actionable maintenance actions ordered best-first.
- "effort" must be one of: low, medium, high.
- "recommendedOptionId" must match the id of the first (best) option.
- Base the actions on the provided sensor readings, failure probability and dominant failure mode.
"""


# =========================================================
# HELPERS
# =========================================================

def _slugify(value: str) -> str:
    """
    Convert an option id or label into a URL-safe slug.
    """
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly parse the model output.

    Strips markdown code fences and takes the payload between the
    first '{' and the last '}' before attempting json.loads.
    """
    if not text:
        return None

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(cleaned[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(parsed, dict):
        return None

    return parsed


def _normalize_options(raw_options: Any) -> List[Dict[str, str]]:
    """
    Validate and normalize raw LLM options into the frontend shape.

    - id       -> slug
    - effort   -> one of low / medium / high (default medium)
    - source   -> "llm" for every option produced here
    - capped at MAX_OPTIONS entries
    """
    if not isinstance(raw_options, list):
        return []

    options = []
    seen_ids = set()

    for raw in raw_options:
        if not isinstance(raw, dict):
            continue

        label = str(raw.get("label") or "").strip()
        if not label:
            continue

        description = str(raw.get("description") or "").strip() or label

        option_id = _slugify(str(raw.get("id") or "")) or _slugify(label)
        if not option_id:
            continue

        # De-duplicate ids the model may have reused.
        base_id = option_id
        suffix = 2
        while option_id in seen_ids:
            option_id = f"{base_id}-{suffix}"
            suffix += 1
        seen_ids.add(option_id)

        effort = str(raw.get("effort") or "").strip().lower()
        if effort not in ALLOWED_EFFORTS:
            effort = "medium"

        options.append(
            {
                "id": option_id,
                "label": label,
                "description": description,
                "effort": effort,
                "source": "llm",
            }
        )

        if len(options) >= MAX_OPTIONS:
            break

    return options


# =========================================================
# PUBLIC FUNCTION
# =========================================================

def get_llm_recommendations(
    product_type: str,
    readings: Dict[str, float],
    failure_probability: float,
    failure_mode_code: Optional[str] = None,
    failure_mode_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Ask OpenRouter for maintenance recommendation options.

    Returns either:

        {
            "recommendationOptions": [
                {"id", "label", "description", "effort", "source"}
            ],
            "selectedRecommendationId": "<id of the best option>"
        }

    or None when the API key is missing or anything fails.
    """

    if not settings.OPENROUTER_API_KEY:
        return None

    try:
        readings_text = "; ".join(
            f"{key}={float(value):g}"
            for key, value in (readings or {}).items()
        )

        dominant_mode = failure_mode_code or "UNKNOWN"
        dominant_mode_name = failure_mode_name or "Not identified"

        user_prompt = (
            "Machine assessment:\n"
            f"- Product type: {product_type}\n"
            f"- Sensor readings: {readings_text}\n"
            f"- Predicted failure probability: {failure_probability:.4f}\n"
            f"- Dominant failure mode: {dominant_mode} "
            f"({dominant_mode_name})\n"
        )

        response = httpx.post(
            f"{settings.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            },
            json={
                "model": settings.OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 600,
                "response_format": {"type": "json_object"},
            },
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]

        parsed = _parse_llm_json(content)

        options = _normalize_options(
            parsed.get("recommendationOptions") if parsed else None
        )

    except Exception:
        logger.warning(
            "LLM recommendation generation failed; "
            "falling back to deterministic recommendations"
        )
        return None

    if not options:
        logger.warning(
            "LLM returned no usable recommendation options; "
            "falling back to deterministic recommendations"
        )
        return None

    recommended_id = str(parsed.get("recommendedOptionId") or "").strip()

    valid_ids = {option["id"] for option in options}
    if recommended_id not in valid_ids:
        recommended_id = options[0]["id"]

    return {
        "recommendationOptions": options,
        "selectedRecommendationId": recommended_id,
    }
