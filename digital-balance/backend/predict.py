"""
predict.py — Prediction logic for Digital Balance backend.
Model and feature importance are loaded once at module import (startup), not per-request.
"""

import os
import json
import logging
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd

from .schemas import WellnessInput, WellnessOutput
from .tips_engine import generate_tips

logger = logging.getLogger(__name__)

# ── Resolve artifact paths ─────────────────────────────────────────────────────
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR   = os.path.join(_BACKEND_DIR, "..", "model")

MODEL_PATH      = os.path.join(_MODEL_DIR, "wellness_model.pkl")
METRICS_PATH    = os.path.join(_MODEL_DIR, "metrics.json")
FEAT_IMP_PATH   = os.path.join(_MODEL_DIR, "feature_importance.json")

# ── Load at startup ────────────────────────────────────────────────────────────
def _load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run `python model/train_model.py` first."
        )
    pipeline         = joblib.load(MODEL_PATH)
    with open(METRICS_PATH, "r") as f:
        metrics      = json.load(f)
    with open(FEAT_IMP_PATH, "r") as f:
        feat_imp     = json.load(f)
    logger.info(f"Model loaded — {metrics.get('best_model')}  test_R²={metrics.get('tuned_test_r2')}")
    return pipeline, metrics, feat_imp


try:
    _PIPELINE, _METRICS, _FEAT_IMP = _load_artifacts()
except FileNotFoundError as exc:
    logger.warning(str(exc))
    _PIPELINE = _METRICS = _FEAT_IMP = None


# ── Category classification ────────────────────────────────────────────────────
def _classify(score: float, thresholds: dict) -> str:
    q25 = thresholds["poor_below"]
    q50 = thresholds["fair_below"]
    q75 = thresholds["good_below"]
    if score < q25:
        return "Poor"
    elif score < q50:
        return "Fair"
    elif score < q75:
        return "Good"
    else:
        return "Excellent"


# ── Top factors ────────────────────────────────────────────────────────────────
def _top_factors(feat_imp: Dict[str, float], n: int = 3) -> List[str]:
    """
    Return human-readable labels for the top-n features by importance.
    Collapses OHE-encoded dummies back to their parent column name.
    """
    LABELS = {
        "stress_level_0_10":          "Stress level",
        "sleep_quality_1_5":          "Sleep quality",
        "sleep_hours":                "Hours of sleep",
        "leisure_screen_hours":       "Leisure screen time",
        "screen_time_hours":          "Total screen time",
        "productivity_0_100":         "Self-rated productivity",
        "exercise_minutes_per_week":  "Weekly exercise",
        "social_hours_per_week":      "Social interaction time",
        "work_screen_hours":          "Work screen time",
        "age":                        "Age",
        "gender":                     "Gender",
        "occupation":                 "Occupation",
        "work_mode":                  "Work mode",
    }
    seen, results = set(), []
    for feat_name in feat_imp.keys():
        # Map OHE suffixes back to base name
        base = feat_name
        for key in LABELS:
            if feat_name.startswith(key):
                base = key
                break
        if base not in seen:
            seen.add(base)
            results.append(LABELS.get(base, base.replace("_", " ").title()))
        if len(results) >= n:
            break
    return results


# ── Main prediction function ───────────────────────────────────────────────────
def predict_wellness(inp: WellnessInput) -> WellnessOutput:
    if _PIPELINE is None:
        raise RuntimeError("Model is not loaded. Train the model first.")

    # Build DataFrame matching training feature order
    row = {
        "age":                       inp.age,
        "gender":                    inp.gender,
        "occupation":                inp.occupation,
        "work_mode":                 inp.work_mode,
        "screen_time_hours":         inp.screen_time_hours,
        "work_screen_hours":         inp.work_screen_hours,
        "leisure_screen_hours":      inp.leisure_screen_hours,
        "sleep_hours":               inp.sleep_hours,
        "sleep_quality_1_5":         inp.sleep_quality_1_5,
        "stress_level_0_10":         inp.stress_level_0_10,
        "productivity_0_100":        inp.productivity_0_100,
        "exercise_minutes_per_week": inp.exercise_minutes_per_week,
        "social_hours_per_week":     inp.social_hours_per_week,
    }
    df_input = pd.DataFrame([row])

    # Predict & clip to valid range
    raw_score = float(_PIPELINE.predict(df_input)[0])
    score     = float(np.clip(raw_score, 0.0, 100.0))

    # Category from data-derived thresholds
    thresholds = _METRICS["category_thresholds"]
    category   = _classify(score, thresholds)

    # Top driving factors
    top_factors = _top_factors(_FEAT_IMP, n=3)

    # Personalised tips
    tips = generate_tips(
        input_dict=row,
        feature_importance=_FEAT_IMP,
        predicted_score=score,
        category=category,
    )

    return WellnessOutput(
        predicted_score=round(score, 1),
        category=category,
        top_factors=top_factors,
        tips=tips,
    )


def get_metrics() -> dict:
    return _METRICS or {}

def get_feature_importance() -> dict:
    return _FEAT_IMP or {}
