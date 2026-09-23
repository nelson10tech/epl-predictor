from __future__ import annotations

import json
from datetime import date, datetime, timezone

from app.data import Fixture, Match
from app.live import load_prediction_records, score_predictions, snapshot_predictions


class FixtureRepository:
    def __init__(self, fixtures=None, matches=None):
        self.fixtures = fixtures or []
        self.matches = matches or []


class FixedPredictor:
    def predict(self, home_team, away_team, as_of=None):
        return {
            "model": "Enhanced Dixon-Coles V2",
            "version": "2.0",
            "probabilities": {"home_win": 0.5, "draw": 0.3, "away_win": 0.2},
            "expected_goals": {"home": 1.6, "away": 0.9},
            "most_likely_score": {"home": 1, "away": 0},
            "data_through": "2026-09-20",
            "xg_enabled": True,
        }


def future_fixture():
    return Fixture(
        played_on=date(2026, 9, 24),
        season="2627",
        home_team="Arsenal",
        away_team="Chelsea",
        kickoff="20:00",
    )


def test_live_prediction_snapshot_is_created_before_result(tmp_path):
    reports = tmp_path / "reports"
    now = datetime(2026, 9, 23, 10, tzinfo=timezone.utc)
    summary = snapshot_predictions(
        FixtureRepository(fixtures=[future_fixture()]),
        FixedPredictor(),
        reports,
        now=now,
    )
    assert summary["created"] == 1
    records = load_prediction_records(reports)
    assert len(records) == 1
    assert datetime.fromisoformat(records[0]["prediction_created_at"]) < datetime.fromisoformat(
        records[0]["kickoff_utc"]
    )
    unsettled = score_predictions(FixtureRepository(), reports, now=now)
    assert unsettled["report"]["settled_predictions"] == 0


def test_duplicate_live_snapshots_are_prevented(tmp_path):
    reports = tmp_path / "reports"
    repository = FixtureRepository(fixtures=[future_fixture()])
    first = snapshot_predictions(
        repository,
        FixedPredictor(),
        reports,
        now=datetime(2026, 9, 23, 10, tzinfo=timezone.utc),
    )
    second = snapshot_predictions(
        repository,
        FixedPredictor(),
        reports,
        now=datetime(2026, 9, 23, 11, tzinfo=timezone.utc),
    )
    assert first["created"] == 1
    assert second["created"] == 0
    assert second["skipped_duplicate"] == 1
    assert len(load_prediction_records(reports)) == 1


def test_scoring_attaches_result_without_overwriting_probabilities(tmp_path):
    reports = tmp_path / "reports"
    now = datetime(2026, 9, 23, 10, tzinfo=timezone.utc)
    snapshot_predictions(
        FixtureRepository(fixtures=[future_fixture()]),
        FixedPredictor(),
        reports,
        now=now,
    )
    prediction_path = reports / "live_predictions" / "2026-09-23.json"
    original_prediction_bytes = prediction_path.read_bytes()
    completed = Match(
        played_on=date(2026, 9, 24),
        season="2627",
        home_team="Arsenal",
        away_team="Chelsea",
        home_goals=2,
        away_goals=1,
    )
    summary = score_predictions(
        FixtureRepository(matches=[completed]),
        reports,
        now=datetime(2026, 9, 25, 10, tzinfo=timezone.utc),
    )
    assert summary["newly_settled"] == 1
    assert prediction_path.read_bytes() == original_prediction_bytes
    results = json.loads((reports / "live_results.json").read_text())["results"]
    assert results[0]["actual_result"] == "H"
    assert summary["report"]["overall"]["sample_size"] == 1
    assert summary["report"]["historical_backtest_included"] is False
