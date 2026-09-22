from __future__ import annotations

import pytest

from app import create_app


@pytest.fixture()
def client():
    app = create_app({"TESTING": True})
    return app.test_client()


def test_prediction_endpoint_returns_normalized_1x2(client):
    response = client.get("/api/predict?home=Arsenal&away=Chelsea")
    assert response.status_code == 200
    payload = response.get_json()

    probabilities = payload["probabilities"]
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(0 < probability < 1 for probability in probabilities.values())
    assert payload["expected_goals"]["home"] > 0
    assert payload["expected_goals"]["away"] > 0
    assert payload["most_likely_score"]["home"] >= 0
    assert payload["most_likely_score"]["away"] >= 0


def test_prediction_rejects_same_team(client):
    response = client.get("/api/predict?home=Arsenal&away=Arsenal")
    assert response.status_code == 400
    assert "different" in response.get_json()["error"]


def test_main_pages_and_health_are_available(client):
    assert client.get("/").status_code == 200
    assert client.get("/fixtures").status_code == 200
    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["matches"] > 0

