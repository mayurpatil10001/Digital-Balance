"""
test_api.py — pytest tests for Digital Balance API.
Run from digital-balance/: pytest tests/ -v
"""

import pytest
from httpx import AsyncClient, ASGITransport

# The app must be imported after model artifacts exist
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.main import app

BASE = "http://test"

VALID_PAYLOAD = {
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

# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE) as ac:
        yield ac


# ── Test 1: Health check ───────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── Test 2: Valid predict returns 200 and score in [0, 100] ───────────────────
@pytest.mark.anyio
async def test_predict_valid(client):
    resp = await client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "predicted_score" in body
    assert 0.0 <= body["predicted_score"] <= 100.0
    assert body["category"] in ("Poor", "Fair", "Good", "Excellent")
    assert isinstance(body["top_factors"], list)
    assert len(body["top_factors"]) > 0
    assert isinstance(body["tips"], list)
    assert len(body["tips"]) > 0
    assert "disclaimer" in body


# ── Test 3: Missing required field returns 422 ─────────────────────────────────
@pytest.mark.anyio
async def test_predict_missing_field(client):
    payload = VALID_PAYLOAD.copy()
    del payload["stress_level_0_10"]
    resp = await client.post("/predict", json=payload)
    assert resp.status_code == 422


# ── Test 4: Out-of-range value returns 422 ────────────────────────────────────
@pytest.mark.anyio
async def test_predict_out_of_range(client):
    payload = VALID_PAYLOAD.copy()
    payload["stress_level_0_10"] = 15.0  # max is 10
    resp = await client.post("/predict", json=payload)
    assert resp.status_code == 422


# ── Test 5: Directional sanity — low-stress/high-sleep > high-stress/low-sleep ─
@pytest.mark.anyio
async def test_predict_directionality(client):
    """A wellness-promoting profile should score higher than a risk-factor profile."""
    good_profile = {
        "age": 30,
        "gender": "Female",
        "occupation": "Employed",
        "work_mode": "In-person",
        "screen_time_hours": 4.0,
        "work_screen_hours": 2.0,
        "leisure_screen_hours": 2.0,
        "sleep_hours": 8.5,
        "sleep_quality_1_5": 5,
        "stress_level_0_10": 1.0,
        "productivity_0_100": 90.0,
        "exercise_minutes_per_week": 300,
        "social_hours_per_week": 15.0,
    }
    poor_profile = {
        "age": 30,
        "gender": "Female",
        "occupation": "Employed",
        "work_mode": "Remote",
        "screen_time_hours": 16.0,
        "work_screen_hours": 8.0,
        "leisure_screen_hours": 8.0,
        "sleep_hours": 4.5,
        "sleep_quality_1_5": 1,
        "stress_level_0_10": 10.0,
        "productivity_0_100": 20.0,
        "exercise_minutes_per_week": 0,
        "social_hours_per_week": 0.0,
    }
    r_good = await client.post("/predict", json=good_profile)
    r_poor = await client.post("/predict", json=poor_profile)
    assert r_good.status_code == 200 and r_poor.status_code == 200

    good_score = r_good.json()["predicted_score"]
    poor_score = r_poor.json()["predicted_score"]

    # Good profile must outscore poor profile by a meaningful margin
    assert good_score > poor_score, (
        f"Expected good_score ({good_score}) > poor_score ({poor_score})"
    )
