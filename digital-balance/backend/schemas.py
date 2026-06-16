"""
schemas.py — Pydantic models for Digital Balance API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class WellnessInput(BaseModel):
    age: int = Field(..., ge=16, le=100, description="Age in years (16–100)")
    gender: str = Field(..., description="Female | Male | Non-binary/Other")
    occupation: str = Field(..., description="Employed | Student | Self-employed | Retired | Unemployed")
    work_mode: str = Field(..., description="Remote | In-person | Hybrid")
    screen_time_hours: float = Field(..., ge=0, le=24, description="Total daily screen time (hours)")
    work_screen_hours: float = Field(..., ge=0, le=24, description="Daily work-related screen time (hours)")
    leisure_screen_hours: float = Field(..., ge=0, le=24, description="Daily leisure screen time (hours)")
    sleep_hours: float = Field(..., ge=0, le=24, description="Daily sleep (hours)")
    sleep_quality_1_5: int = Field(..., ge=1, le=5, description="Sleep quality 1 (poor) to 5 (excellent)")
    stress_level_0_10: float = Field(..., ge=0, le=10, description="Stress level 0 (none) to 10 (severe)")
    productivity_0_100: float = Field(..., ge=0, le=100, description="Self-rated productivity 0–100")
    exercise_minutes_per_week: int = Field(..., ge=0, le=10080, description="Weekly exercise in minutes")
    social_hours_per_week: float = Field(..., ge=0, le=168, description="Weekly social interaction hours")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 28,
                "gender": "Female",
                "occupation": "Employed",
                "work_mode": "Remote",
                "screen_time_hours": 9.5,
                "work_screen_hours": 6.0,
                "leisure_screen_hours": 3.5,
                "sleep_hours": 7.0,
                "sleep_quality_1_5": 3,
                "stress_level_0_10": 6.5,
                "productivity_0_100": 65.0,
                "exercise_minutes_per_week": 120,
                "social_hours_per_week": 8.0,
            }
        }
    }


class WellnessOutput(BaseModel):
    predicted_score: float = Field(..., description="Predicted wellness score 0–100")
    category: str = Field(..., description="Poor | Fair | Good | Excellent")
    top_factors: List[str] = Field(..., description="Top 2–3 features driving the score")
    tips: List[str] = Field(..., description="2–3 personalised improvement tips")
    disclaimer: str = Field(
        default=(
            "This is an estimated wellness indicator based on a statistical model trained on "
            "survey data. It is not a diagnosis and not a substitute for professional mental "
            "health advice."
        )
    )


class HealthResponse(BaseModel):
    status: str


class MetadataResponse(BaseModel):
    gender_options: List[str]
    occupation_options: List[str]
    work_mode_options: List[str]
    field_ranges: dict
    category_thresholds: dict
    model_info: dict
