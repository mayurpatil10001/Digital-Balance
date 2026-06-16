"""
tips_engine.py — Rule-based personalized tips generator for Digital Balance.
"""

import json
import os
from typing import Dict, List, Any

# Dataset-derived reference values (used for thresholds; updated from feature_importance.json at runtime)
# These are fallback defaults matching the dataset stats
DATASET_REFS = {
    "leisure_screen_hours_p75":       8.5,   # 75th percentile
    "screen_time_hours_p75":          11.5,
    "sleep_quality_threshold":        3,     # below = flagged
    "sleep_hours_low":                6.5,
    "stress_threshold":               7.0,
    "exercise_median":                100,   # minutes/week
    "social_hours_low":               5.0,
    "productivity_low":               45.0,
}

# ── Tip catalogue ──────────────────────────────────────────────────────────────
# Each entry: (feature_key, condition_fn, tip_text)
TIP_RULES = [
    (
        "leisure_screen_hours",
        lambda v: v > DATASET_REFS["leisure_screen_hours_p75"],
        "Your leisure screen time is well above average in this dataset. "
        "Consider setting a phone-free wind-down period of 45–60 minutes before bed — "
        "even small reductions in evening screen use are linked to better sleep onset."
    ),
    (
        "screen_time_hours",
        lambda v: v > DATASET_REFS["screen_time_hours_p75"],
        "Your total daily screen exposure is in the top quarter of this dataset. "
        "Try the 20-20-20 rule: every 20 minutes, look at something 20 feet away for 20 seconds "
        "to reduce eye strain and mental fatigue."
    ),
    (
        "sleep_quality_1_5",
        lambda v: v < DATASET_REFS["sleep_quality_threshold"],
        "Your sleep quality score is below average. Consistent sleep and wake times — "
        "even on weekends — and reducing screen use 1 hour before bed are the two habits "
        "most strongly linked to improved sleep quality in survey data like this."
    ),
    (
        "sleep_hours",
        lambda v: v < DATASET_REFS["sleep_hours_low"],
        "You're averaging fewer sleep hours than most in this dataset. "
        "Even a 30-minute increase in nightly sleep is associated with meaningfully "
        "better wellness scores — protecting sleep time from late-night screen use can help."
    ),
    (
        "stress_level_0_10",
        lambda v: v > DATASET_REFS["stress_threshold"],
        "Your stress score is elevated relative to others in this dataset. "
        "Short, deliberate breaks during work — 5 minutes every hour — are linked to "
        "lower reported stress in remote and hybrid workers in this data."
    ),
    (
        "exercise_minutes_per_week",
        lambda v: v < DATASET_REFS["exercise_median"],
        "Your weekly exercise is below the dataset median. Even adding 30–60 minutes of "
        "moderate activity per week is associated with better stress resilience and higher "
        "wellness scores in this dataset — walking counts."
    ),
    (
        "social_hours_per_week",
        lambda v: v < DATASET_REFS["social_hours_low"],
        "Your reported social interaction time is low compared to others in this dataset. "
        "In-person or scheduled social time — even brief — correlates with higher wellness scores. "
        "Consider scheduling one regular social touchpoint per week."
    ),
    (
        "productivity_0_100",
        lambda v: v < DATASET_REFS["productivity_low"],
        "Your self-rated productivity is on the lower end. Breaking work into focused "
        "25-minute blocks (Pomodoro technique) and reducing leisure screen use during "
        "work hours is associated with higher productivity in this dataset."
    ),
]

# Remote work + high stress compound tip
def _remote_stress_tip(inp: dict) -> str | None:
    if inp.get("work_mode") == "Remote" and inp.get("stress_level_0_10", 0) > DATASET_REFS["stress_threshold"]:
        return (
            "Remote workers with elevated stress in this dataset reported higher wellness "
            "when they enforced a clear end-of-workday routine. Consider a physical 'shutdown ritual' "
            "— closing your laptop, a short walk, or changing clothes — to signal the end of work time."
        )
    return None


# Maintenance tips for Good/Excellent scorers
MAINTENANCE_TIPS = [
    "Great balance! Maintaining your current sleep and stress management habits is key — "
    "the dataset shows that consistency matters more than perfection.",
    "Your wellness indicators look solid. Consider tracking your screen-free time intentionally "
    "to protect the habits that are working well for you.",
    "You're in a positive range. One evidence-backed habit to reinforce: keeping a regular "
    "sleep schedule even when screen time increases during busy periods.",
]


def generate_tips(
    input_dict: Dict[str, Any],
    feature_importance: Dict[str, float],
    predicted_score: float,
    category: str,
) -> List[str]:
    """
    Generate 2–3 personalized, rule-based, non-clinical tips.

    Args:
        input_dict: raw user input dict matching WellnessInput fields
        feature_importance: {feature_name: importance} sorted descending
        predicted_score: predicted wellness score 0–100
        category: 'Poor' | 'Fair' | 'Good' | 'Excellent'

    Returns:
        List of 2–3 tip strings
    """
    # For Good/Excellent, return maintenance tips
    if category in ("Good", "Excellent"):
        return MAINTENANCE_TIPS[:2]

    # Build ordered list of tip candidates based on feature importance ranking
    # Get top feature base names (strip OHE suffixes)
    top_features_ordered = []
    for feat_name in feature_importance.keys():
        # Map OHE names back to base column names
        base = feat_name.split("_")[0] if feat_name.startswith(("gender", "occupation", "work")) else feat_name
        if base not in top_features_ordered:
            top_features_ordered.append(base)
        if len(top_features_ordered) >= 8:
            break

    triggered_tips = []

    # Check compound tip first
    compound = _remote_stress_tip(input_dict)

    # Evaluate rules in importance order
    for feature_key, condition_fn, tip_text in TIP_RULES:
        val = input_dict.get(feature_key)
        if val is not None:
            try:
                if condition_fn(float(val)):
                    triggered_tips.append(tip_text)
            except (TypeError, ValueError):
                continue
        if len(triggered_tips) >= 3:
            break

    # Insert compound tip if relevant and we have room
    if compound and len(triggered_tips) < 3 and compound not in triggered_tips:
        triggered_tips.insert(0, compound)

    # If no rules triggered, return generic maintenance
    if not triggered_tips:
        return MAINTENANCE_TIPS[:2]

    return triggered_tips[:3]
