from __future__ import annotations

import pytest


def test_v2_prediction_endpoint_returns_normalized_1x2(client):
    response = client.get("/api/predict?home=Arsenal&away=Chelsea")
    assert response.status_code == 200
    payload = response.get_json()
    assert sum(payload["probabilities"].values()) == pytest.approx(1.0, abs=1e-9)
    assert payload["expected_goals"]["home"] >= 0
    assert payload["expected_goals"]["away"] >= 0
    assert len(payload["top_scorelines"]) == 5
    assert payload["model"] == "Enhanced Dixon-Coles V2"
    assert payload["version"] == "2.0"
    assert payload["app_version"] == "2.1"
    assert payload["model_version"] == "2.0"
    assert payload["data_through"]
    assert "freshness" in payload["metadata"]


def test_same_team_and_unknown_team_are_rejected(client):
    same = client.get("/api/predict?home=Arsenal&away=Arsenal")
    unknown = client.get("/api/predict?home=Made%20Up%20FC&away=Arsenal")
    assert same.status_code == 400
    assert "different" in same.get_json()["error"]
    assert unknown.status_code == 400
    assert "loaded EPL data" in unknown.get_json()["error"]


def test_v1_baseline_remains_available(client):
    response = client.get("/api/predict?home=Arsenal&away=Chelsea&model=v1")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["version"] == "1.0"
    assert payload["model"] == "Elo + Poisson V1"
    assert sum(payload["probabilities"].values()) == pytest.approx(1.0, abs=1e-9)


def test_health_and_mobile_pages_load(client):
    assert client.get("/").status_code == 200
    model_page = client.get("/model")
    assert model_page.status_code == 200
    assert "HISTORICAL BACKTEST" in model_page.get_data(as_text=True)
    assert "LIVE PERFORMANCE" in model_page.get_data(as_text=True)
    assert "Small sample — live performance is not yet statistically reliable." in model_page.get_data(as_text=True)
    health = client.get("/health")
    assert health.status_code == 200
    payload = health.get_json()
    assert payload["status"] == "ok"
    assert payload["app_version"] == "2.1"
    assert payload["model_version"] == "2.0"
    assert payload["matches"] > 0
    assert "data_stale" in payload
    assert "refresh_in_progress" in payload
    assert "last_refresh_error" in payload

    model_api = client.get("/api/model").get_json()
    assert model_api["app_version"] == "2.1"
    assert model_api["model_version"] == "2.0"
    assert model_api["live_performance"]["historical_backtest_included"] is False
    assert "last_refresh_success" in model_api


def test_fixtures_and_results_are_labelled_correctly(client):
    response = client.get("/fixtures")
    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "即將舉行" in body
    assert "最近完成" in body
    assert "未能取得即將舉行的賽程" in body


def test_no_public_refresh_endpoint_remains(client):
    response = client.post("/api/refresh")
    assert response.status_code == 404
