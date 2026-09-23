from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app import create_app
from app.data import MatchRepository, reset_startup_refresh_state_for_tests


SOURCE_DATA = Path(__file__).resolve().parent.parent / "data" / "raw" / "2627_E0.csv"


def repository_with_data(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    shutil.copy2(SOURCE_DATA, data_dir / "2627_E0.csv")
    repository = MatchRepository(data_dir=data_dir, history_seasons=1)
    repository.load()
    return repository


def test_stale_data_triggers_refresh_attempt(tmp_path, monkeypatch):
    repository = repository_with_data(tmp_path)
    now = datetime(2026, 9, 23, tzinfo=timezone.utc)
    repository.last_refresh_success = now - timedelta(hours=13)
    called = []

    def fake_refresh(now=None):
        called.append(now)
        repository.last_refresh_success = now
        return {"matches": len(repository.matches)}

    monkeypatch.setattr(repository, "refresh", fake_refresh)
    assert repository.ensure_fresh(max_age_hours=12, now=now) is True
    assert called == [now]


def test_fresh_data_does_not_download(tmp_path, monkeypatch):
    repository = repository_with_data(tmp_path)
    now = datetime(2026, 9, 23, tzinfo=timezone.utc)
    repository.last_refresh_success = now - timedelta(hours=2)

    def unexpected_refresh(now=None):
        raise AssertionError("fresh data must not trigger refresh")

    monkeypatch.setattr(repository, "refresh", unexpected_refresh)
    assert repository.ensure_fresh(max_age_hours=12, now=now) is False
    assert repository.last_refresh_attempt is None


def test_refresh_failure_keeps_existing_data(tmp_path, monkeypatch):
    repository = repository_with_data(tmp_path)
    original_count = len(repository.matches)
    now = datetime(2026, 9, 23, tzinfo=timezone.utc)
    repository.last_refresh_success = now - timedelta(days=2)

    def failed_refresh(now=None):
        raise RuntimeError("temporary upstream failure")

    monkeypatch.setattr(repository, "refresh", failed_refresh)
    assert repository.ensure_fresh(max_age_hours=12, now=now) is False
    assert len(repository.matches) == original_count
    assert repository.data_stale is True
    assert "temporary upstream failure" in repository.last_refresh_error


def test_app_starts_after_startup_refresh_failure(tmp_path, monkeypatch):
    reset_startup_refresh_state_for_tests()
    repository = repository_with_data(tmp_path)
    old = (datetime.now(timezone.utc) - timedelta(days=3)).timestamp()
    os.utime(repository.loaded_files[0], (old, old))

    def failed_refresh(self, now=None):
        raise RuntimeError("Football-Data unavailable")

    monkeypatch.setattr(MatchRepository, "refresh", failed_refresh)
    app = create_app({
        "TESTING": True,
        "AUTO_REFRESH": True,
        "DATA_DIR": str(repository.data_dir),
        "HISTORY_SEASONS": 1,
    })
    client = app.test_client()
    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["data_stale"] is True
    assert "Football-Data unavailable" in health.get_json()["last_refresh_error"]
    assert health.get_json()["refresh_in_progress"] is False
    assert client.get("/api/predict?home=Arsenal&away=Chelsea").status_code == 200
