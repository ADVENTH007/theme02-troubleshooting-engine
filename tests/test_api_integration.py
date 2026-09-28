"""
tests/test_api_integration.py
--------------------------------
End-to-end test: spins up the actual FastAPI app in-process (no server
needed) and POSTs every one of the 20 real sample rows at it, checking
each response is schema-valid. This is the closest thing to "does this
actually work" without a live deployment.

Run with: pytest tests/test_api_integration.py -v

Needs `fastapi`, `pydantic`, and `httpx` installed (all in requirements.txt).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app.main import app

SAMPLES_PATH = Path(__file__).resolve().parents[1] / "samples" / "siis_responses.json"


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_rows():
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["responses"]


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # These report which optional backend is active (diskcache vs in-memory,
    # semantic vs keyword matching) — present regardless of which extras are
    # installed, so we check they exist and are sensible rather than
    # asserting a specific value that would differ by environment.
    assert body["cache_backend"] in ("diskcache", "memory")
    assert body["deeplink_matching_tier"] in ("embeddings", "keyword-fallback")
    assert isinstance(body["semantic_relevance_available"], bool)


def test_every_sample_row_returns_valid_shape(client, sample_rows):
    for row in sample_rows:
        payload = {
            "id": row["id"],
            "original_query": row["original_query"],
            "siis_response": row["siis_response"],
        }
        resp = client.post("/v1/troubleshoot", json=payload)
        assert resp.status_code == 200, f"{row['id']} failed: {resp.text}"

        body = resp.json()
        assert body["id"] == row["id"]
        assert "latency_ms" in body
        assert isinstance(body["contexts"], list)

        for goal in body["contexts"]:
            assert 0.0 <= goal["score"] <= 1.0
            for action in goal["actions"]:
                assert action["category"] in ("auto", "manual", "critical")
                assert len(action["stepGroups"]) >= 1
                for step_group in action["stepGroups"]:
                    assert len(step_group["steps"]) >= 1


def test_missing_required_field_returns_422(client):
    resp = client.post("/v1/troubleshoot", json={"original_query": "no siis_response given"})
    assert resp.status_code == 422  # FastAPI/pydantic request validation error


def test_repeated_request_hits_cache_and_is_fast(client, sample_rows):
    row = sample_rows[0]
    payload = {
        "id": row["id"],
        "original_query": row["original_query"],
        "siis_response": row["siis_response"],
    }
    first = client.post("/v1/troubleshoot", json=payload).json()
    second = client.post("/v1/troubleshoot", json=payload).json()
    assert first["contexts"] == second["contexts"]
    # Assert on the explicit cache_hit flag rather than comparing raw
    # latency numbers: for a cheap query, both the cold and warm call can
    # round to the same millisecond figure (e.g. "0.02" vs "0.02"), which
    # made a plain "second < first" comparison flaky even though caching
    # was working correctly. The flag is the ground truth; latency is a
    # secondary, non-strict sanity check (warm should never be slower).
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["latency_ms"] <= first["latency_ms"]
