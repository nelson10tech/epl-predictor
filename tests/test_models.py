from __future__ import annotations

from datetime import date

import pytest

from app.backtest import _walk_season, run_smoke_backtest
from app.backtest_v3 import run_v3_smoke_backtest, select_ensemble_weight
from app.data import Match
from app.features import PreMatchFeatureBuilder
from app.feature_store import ChronologicalFeatureStore
from app.model import dixon_coles_score_matrix, outcome_probabilities
from app.xg import FootballDataXGProvider


def match(day, season, home, away, home_goals, away_goals):
    return Match(
        played_on=day,
        season=season,
        home_team=home,
        away_team=away,
        home_goals=home_goals,
        away_goals=away_goals,
    )


def test_dixon_coles_score_probabilities_normalize():
    matrix = dixon_coles_score_matrix(1.72, 1.04)
    outcomes = outcome_probabilities(matrix)
    assert sum(probability for _, _, probability in matrix) == pytest.approx(1.0, abs=1e-12)
    assert sum(outcomes.values()) == pytest.approx(1.0, abs=1e-12)
    assert all(probability >= 0 for _, _, probability in matrix)


def test_feature_builder_uses_only_pre_match_information():
    first = match(date(2025, 8, 10), "A", "Arsenal", "Chelsea", 2, 0)
    target = match(date(2025, 8, 17), "A", "Arsenal", "Chelsea", 0, 4)
    builder = PreMatchFeatureBuilder([first])
    before = builder.snapshot("Arsenal", "Chelsea", target.played_on)
    assert before.training_through == first.played_on
    assert before.training_through < target.played_on

    builder.update_day([target])
    after = builder.snapshot("Arsenal", "Chelsea", date(2025, 8, 18))
    assert after.values["home_recent5_ga"] > before.values["home_recent5_ga"]


def test_target_match_xg_is_not_a_feature():
    first = Match(
        played_on=date(2025, 8, 10), season="A", home_team="Arsenal",
        away_team="Chelsea", home_goals=2, away_goals=0, home_xg=1.2, away_xg=0.4,
    )
    target = Match(
        played_on=date(2025, 8, 17), season="A", home_team="Arsenal",
        away_team="Chelsea", home_goals=0, away_goals=4, home_xg=9.0, away_xg=8.0,
    )
    store = ChronologicalFeatureStore(
        [first, target], xg_provider=FootballDataXGProvider()
    )
    target_row = store.rows[1]
    assert target_row.snapshot.training_through == first.played_on
    assert target_row.snapshot.values["home_xg_samples"] == 1
    assert target_row.snapshot.values["home_recent5_xg"] == pytest.approx(1.2)


def test_walk_forward_backtest_never_trains_on_future_matches():
    matches = [
        match(date(2024, 8, 1), "A", "Arsenal", "Chelsea", 2, 1),
        match(date(2024, 8, 2), "A", "Liverpool", "Arsenal", 1, 1),
        match(date(2025, 8, 1), "B", "Chelsea", "Liverpool", 0, 1),
        match(date(2025, 8, 1), "B", "Arsenal", "Promoted", 2, 0),
        match(date(2025, 8, 9), "B", "Promoted", "Chelsea", 1, 1),
    ]
    records = _walk_season(matches, "B", temperature=1.0, include_v1=True)
    assert records
    assert all(record["training_through"] < record["match_date"] for record in records)
    same_day = [record for record in records if record["match_date"] == date(2025, 8, 1)]
    assert len(same_day) == 2
    assert len({record["training_through"] for record in same_day}) == 1


def test_ci_smoke_backtest_passes(app):
    matches = app.extensions["prediction_runtime"].snapshot().repository.matches
    result = run_smoke_backtest(matches)
    assert result["sample_size"] > 0
    assert result["probabilities_normalized"] is True
    assert result["leakage_check_passed"] is True
    v3 = run_v3_smoke_backtest(matches, max_matches=10)
    assert v3["sample_size"] > 0
    assert v3["probabilities_normalized"] is True
    assert v3["leakage_check_passed"] is True


def test_ensemble_weight_uses_only_passed_validation_rows():
    validation = [
        {
            "result": "H",
            "v2_probabilities": {"home_win": 0.7, "draw": 0.2, "away_win": 0.1},
            "ml_probabilities": {"home_win": 0.2, "draw": 0.3, "away_win": 0.5},
        },
        {
            "result": "A",
            "v2_probabilities": {"home_win": 0.2, "draw": 0.2, "away_win": 0.6},
            "ml_probabilities": {"home_win": 0.6, "draw": 0.2, "away_win": 0.2},
        },
    ] * 60
    weight, candidates = select_ensemble_weight(validation)
    assert weight == 1.0
    assert len(candidates) == 11
