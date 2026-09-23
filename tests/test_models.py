from __future__ import annotations

from datetime import date

import pytest

from app.backtest import _walk_season, run_smoke_backtest
from app.data import Match
from app.features import PreMatchFeatureBuilder
from app.model import dixon_coles_score_matrix, outcome_probabilities


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
