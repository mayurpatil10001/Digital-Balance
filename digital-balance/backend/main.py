"""
main.py — FastAPI application for Digital Balance.
Run: uvicorn backend.main:app --reload  (from digital-balance/ root)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import WellnessInput, WellnessOutput, HealthResponse, MetadataResponse
from .predict import predict_wellness, get_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
logger = logging.getLogger(__name__)


# ── Lifespan (replaces deprecated on_event) ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    metrics = get_metrics()
    if metrics:
        logger.info("=" * 55)
        logger.info("  Digital Balance API — Model loaded successfully")
        logger.info(f"  Model   : {metrics.get('best_model', 'unknown')}")
        logger.info(f"  Test R2 : {metrics.get('tuned_test_r2', 'N/A')}")
        logger.info(f"  Test MAE: {metrics.get('tuned_test_mae', 'N/A')}")
        logger.info(f"  Test RMSE:{metrics.get('tuned_test_rmse', 'N/A')}")
        logger.info("=" * 55)
    else:
        logger.warning("Model artifacts not found — run train_model.py first!")
    yield  # app runs here


app = FastAPI(
    title="Digital Balance API",
    description="Mental Wellness Score predictor based on screen time, sleep, stress and lifestyle data.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.get("/metadata", response_model=MetadataResponse, tags=["System"])
def metadata():
    """
    Returns dropdown options, field ranges and category thresholds
    so the frontend can build its form dynamically.
    """
    metrics = get_metrics()
    thresholds = metrics.get("category_thresholds", {}) if metrics else {}
    model_info = {
        "model_name":  metrics.get("best_model", "unknown") if metrics else "unknown",
        "test_r2":     metrics.get("tuned_test_r2", None) if metrics else None,
        "test_mae":    metrics.get("tuned_test_mae", None) if metrics else None,
        "test_rmse":   metrics.get("tuned_test_rmse", None) if metrics else None,
        "n_train":     metrics.get("dataset_stats", {}).get("n_train") if metrics else None,
    }

    return MetadataResponse(
        gender_options=["Female", "Male", "Non-binary/Other"],
        occupation_options=["Employed", "Student", "Self-employed", "Retired", "Unemployed"],
        work_mode_options=["Remote", "In-person", "Hybrid"],
        field_ranges={
            "age":                       {"min": 16,   "max": 100,   "step": 1,    "default": 28},
            "screen_time_hours":         {"min": 0,    "max": 24,    "step": 0.5,  "default": 8.0},
            "work_screen_hours":         {"min": 0,    "max": 24,    "step": 0.5,  "default": 4.0},
            "leisure_screen_hours":      {"min": 0,    "max": 24,    "step": 0.5,  "default": 4.0},
            "sleep_hours":               {"min": 0,    "max": 24,    "step": 0.5,  "default": 7.0},
            "sleep_quality_1_5":         {"min": 1,    "max": 5,     "step": 1,    "default": 3},
            "stress_level_0_10":         {"min": 0,    "max": 10,    "step": 0.5,  "default": 5.0},
            "productivity_0_100":        {"min": 0,    "max": 100,   "step": 1,    "default": 60},
            "exercise_minutes_per_week": {"min": 0,    "max": 600,   "step": 10,   "default": 120},
            "social_hours_per_week":     {"min": 0,    "max": 50,    "step": 0.5,  "default": 8.0},
        },
        category_thresholds=thresholds,
        model_info=model_info,
    )


@app.post("/predict", response_model=WellnessOutput, tags=["Prediction"])
def predict(inp: WellnessInput):
    """
    Predict mental wellness score from lifestyle inputs.
    Returns score (0–100), category, top factors, and personalised tips.
    """
    try:
        result = predict_wellness(inp)
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")
