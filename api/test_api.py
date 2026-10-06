"""Test suite for OTT Audience Segmentation & Personalization REST API (Phase 2).

Tests:
1. GET /health -> HTTP 200, status="ok", model_loaded=True
2. POST /recommend (valid profile) -> HTTP 200, segment and recommendations returned
3. POST /recommend (missing field) -> HTTP 422
4. POST /recommend (negative watch time) -> HTTP 422
5. POST /recommend (invalid percentage > 1.0) -> HTTP 422
6. POST /recommend (wrong type string instead of float) -> HTTP 422
7. POST /recommend (empty user_id) -> HTTP 422
8. Repeat valid request -> Identical deterministic output (no retraining)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Silence external library deprecation notices during test runs
warnings.filterwarnings("ignore")

from fastapi.testclient import TestClient

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from api.main import app


def run_tests() -> None:
    print("=" * 60)
    print("RUNNING OTT AUDIENCE SEGMENTATION API TEST SUITE")
    print("=" * 60)

    # Use TestClient with lifespan context
    with TestClient(app) as client:
        # TEST 1: GET /health
        print("\n[TEST 1] GET /health")
        resp = client.get("/health")
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {resp.json()}")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        print("-> PASSED")

        # TEST 2: POST /recommend (Valid Profile)
        print("\n[TEST 2] POST /recommend (Valid Profile)")
        valid_payload = {
            "user_id": "USR-8192",
            "watch_time_hours": 32.5,
            "sessions_per_week": 5.0,
            "avg_session_mins": 85.0,
            "completion_rate": 0.82,
            "action_pct": 0.45,
            "comedy_pct": 0.10,
            "drama_pct": 0.15,
            "thriller_pct": 0.20,
            "sci_fi_pct": 0.10,
            "weekend_viewing_pct": 0.65,
            "days_since_last_watch": 2.0,
        }
        resp = client.post("/recommend", json=valid_payload)
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {resp.json()}")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        res_data = resp.json()
        assert res_data["user_id"] == "USR-8192"
        assert "segment_id" in res_data
        assert "segment_name" in res_data
        assert "recommendations" in res_data
        assert isinstance(res_data["recommendations"], list)
        assert len(res_data["recommendations"]) > 0
        assert "distance_to_centroid" in res_data
        assert isinstance(res_data["distance_to_centroid"], float)
        print(f"-> Predicted Segment: {res_data['segment_name']} (ID: {res_data['segment_id']})")
        print(f"-> Recommendations: {res_data['recommendations']}")
        print(f"-> Distance to Centroid: {res_data['distance_to_centroid']}")
        print("-> PASSED")

        # TEST 3: POST /recommend (Missing Required Field: watch_time_hours missing)
        print("\n[TEST 3] POST /recommend (Missing Required Field)")
        payload_missing = dict(valid_payload)
        del payload_missing["watch_time_hours"]
        resp = client.post("/recommend", json=payload_missing)
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {resp.json()}")
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"
        print("-> PASSED")

        # TEST 4: POST /recommend (Negative watch_time_hours)
        print("\n[TEST 4] POST /recommend (Negative Watch Time)")
        payload_negative = dict(valid_payload, watch_time_hours=-5.0)
        resp = client.post("/recommend", json=payload_negative)
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {resp.json()}")
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"
        print("-> PASSED")

        # TEST 5: POST /recommend (Invalid Percentage: completion_rate = 1.5)
        print("\n[TEST 5] POST /recommend (Invalid Completion Rate > 1.0)")
        payload_invalid_pct = dict(valid_payload, completion_rate=1.5)
        resp = client.post("/recommend", json=payload_invalid_pct)
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {resp.json()}")
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"
        print("-> PASSED")

        # TEST 6: POST /recommend (Wrong Numeric Type)
        print("\n[TEST 6] POST /recommend (Wrong Numeric Type: string instead of float)")
        payload_wrong_type = dict(valid_payload, watch_time_hours="hello")
        resp = client.post("/recommend", json=payload_wrong_type)
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {resp.json()}")
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"
        print("-> PASSED")

        # TEST 7: POST /recommend (Empty user_id)
        print("\n[TEST 7] POST /recommend (Empty user_id string)")
        payload_empty_user = dict(valid_payload, user_id="   ")
        resp = client.post("/recommend", json=payload_empty_user)
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {resp.json()}")
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"
        print("-> PASSED")

        # TEST 8: Repeat Exact Request (Verify Determinism & No Retraining)
        print("\n[TEST 8] Repeat Request (Determinism Verification)")
        resp2 = client.post("/recommend", json=valid_payload)
        assert resp2.status_code == 200
        res_data2 = resp2.json()
        assert res_data == res_data2, "Repeat request produced different response!"
        print(f"Repeated Segment ID: {res_data2['segment_id']} ({res_data2['segment_name']})")
        print(f"Repeated Distance: {res_data2['distance_to_centroid']}")
        print("-> Output is 100% consistent across repeated calls. No retraining occurs.")
        print("-> PASSED")

    print("\n" + "=" * 60)
    print("ALL API TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
