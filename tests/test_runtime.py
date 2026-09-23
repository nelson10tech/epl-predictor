from __future__ import annotations

import copy
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event

import pytest

from app import create_app
from app.runtime import RuntimeState, build_predictors


SOURCE_DATA = Path(__file__).resolve().parent.parent / "data" / "raw" / "2627_E0.csv"


@pytest.fixture()
def refresh_app(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    shutil.copy2(SOURCE_DATA, data_dir / "2627_E0.csv")
    return create_app({
        "TESTING": True,
        "AUTO_REFRESH": False,
        "STARTUP_REFRESH": False,
        "DATA_DIR": str(data_dir),
        "REPORTS_DIR": str(tmp_path / "reports"),
        "HISTORY_SEASONS": 1,
        "REFRESH_CHECK_INTERVAL_SECONDS": 300,
    })


def prepare_stale_runtime(app, now):
    runtime = app.extensions["prediction_runtime"]
    runtime.auto_refresh = True
    runtime.snapshot().repository.last_refresh_success = now - timedelta(hours=13)
    runtime.reset_check_timer_for_tests()
    return runtime


def test_stale_request_triggers_background_refresh(refresh_app, monkeypatch):
    now = datetime.now(timezone.utc)
    runtime = prepare_stale_runtime(refresh_app, now)
    called = Event()

    def refreshed_state(started_at):
        called.set()
        return runtime.snapshot()

    monkeypatch.setattr(runtime, "_create_refreshed_state", refreshed_state)
    assert refresh_app.test_client().get("/health").status_code == 200
    assert called.wait(1)
    assert runtime.wait_for_refresh()


def test_fresh_request_does_not_start_refresh(refresh_app, monkeypatch):
    runtime = refresh_app.extensions["prediction_runtime"]
    runtime.auto_refresh = True
    runtime.snapshot().repository.last_refresh_success = datetime.now(timezone.utc)
    runtime.reset_check_timer_for_tests()

    def unexpected_refresh(started_at):
        raise AssertionError("fresh request must not start a download")

    monkeypatch.setattr(runtime, "_create_refreshed_state", unexpected_refresh)
    assert refresh_app.test_client().get("/health").status_code == 200
    metadata = runtime.freshness_metadata()
    assert metadata["refresh_in_progress"] is False
    assert metadata["last_refresh_error"] is None


def test_normal_request_is_not_blocked_by_refresh(refresh_app, monkeypatch):
    runtime = prepare_stale_runtime(refresh_app, datetime.now(timezone.utc))
    started = Event()
    release = Event()

    def slow_refresh(started_at):
        started.set()
        release.wait(2)
        return runtime.snapshot()

    monkeypatch.setattr(runtime, "_create_refreshed_state", slow_refresh)
    began = time.monotonic()
    response = refresh_app.test_client().get("/")
    elapsed = time.monotonic() - began
    assert response.status_code == 200
    assert started.wait(1)
    assert elapsed < 0.5
    release.set()
    assert runtime.wait_for_refresh()


def test_multiple_requests_do_not_duplicate_refresh(refresh_app, monkeypatch):
    runtime = prepare_stale_runtime(refresh_app, datetime.now(timezone.utc))
    started = Event()
    release = Event()
    calls = []

    def slow_refresh(started_at):
        calls.append(started_at)
        started.set()
        release.wait(2)
        return runtime.snapshot()

    monkeypatch.setattr(runtime, "_create_refreshed_state", slow_refresh)
    client = refresh_app.test_client()
    assert client.get("/").status_code == 200
    assert started.wait(1)
    assert client.get("/api/model").status_code == 200
    assert len(calls) == 1
    release.set()
    assert runtime.wait_for_refresh()


def test_successful_refresh_atomically_replaces_model(refresh_app, monkeypatch):
    runtime = prepare_stale_runtime(refresh_app, datetime.now(timezone.utc))
    original = runtime.snapshot()
    candidate_repository = copy.deepcopy(original.repository)
    candidate_repository.data_stale = False
    candidate_repository.last_refresh_success = datetime.now(timezone.utc)
    replacement = RuntimeState(
        repository=candidate_repository,
        predictors=build_predictors(candidate_repository, runtime.calibration_temperature),
    )
    started = Event()
    release = Event()

    def completed_refresh(started_at):
        started.set()
        release.wait(2)
        return replacement

    monkeypatch.setattr(runtime, "_create_refreshed_state", completed_refresh)
    assert refresh_app.test_client().get("/health").status_code == 200
    assert started.wait(1)
    assert runtime.snapshot() is original
    release.set()
    assert runtime.wait_for_refresh()
    assert runtime.snapshot() is replacement
    assert runtime.snapshot().predictors["v2"] is not original.predictors["v2"]


def test_failed_refresh_keeps_previous_model_available(refresh_app, monkeypatch):
    runtime = prepare_stale_runtime(refresh_app, datetime.now(timezone.utc))
    original = runtime.snapshot()

    def failed_refresh(started_at):
        raise RuntimeError("upstream temporarily unavailable")

    monkeypatch.setattr(runtime, "_create_refreshed_state", failed_refresh)
    client = refresh_app.test_client()
    assert client.get("/").status_code == 200
    assert runtime.wait_for_refresh()
    assert runtime.snapshot() is original
    prediction = client.get("/api/predict?home=Arsenal&away=Chelsea")
    assert prediction.status_code == 200
    metadata = runtime.freshness_metadata()
    assert metadata["data_stale"] is True
    assert "temporarily unavailable" in metadata["last_refresh_error"]
