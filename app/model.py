from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from .data import Match
from .features import FeatureSnapshot, PreMatchFeatureBuilder
from .xg import FootballDataXGProvider, XGProvider


OUTCOMES = ("home_win", "draw", "away_win")


@dataclass
class WeightedStats:
    weight: float = 0.0
    goals_for: float = 0.0
    goals_against: float = 0.0


class TemperatureCalibrator:
    """One-parameter multiclass calibration that preserves outcome ordering."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = max(0.4, min(3.0, temperature))

    def apply(self, probabilities: dict[str, float]) -> dict[str, float]:
        powered = {
            key: max(probabilities[key], 1e-12) ** (1.0 / self.temperature)
            for key in OUTCOMES
        }
        total = sum(powered.values())
        return {key: powered[key] / total for key in OUTCOMES}

    @classmethod
    def fit(cls, samples: list[tuple[dict[str, float], str]]) -> "TemperatureCalibrator":
        if len(samples) < 100:
            return cls(1.0)
        best_temperature = 1.0
        best_loss = float("inf")
        for step in range(40, 201, 2):
            temperature = step / 100.0
            calibrator = cls(temperature)
            loss = 0.0
            for probabilities, result in samples:
                calibrated = calibrator.apply(probabilities)
                outcome = {"H": "home_win", "D": "draw", "A": "away_win"}[result]
                loss -= math.log(max(calibrated[outcome], 1e-12))
            loss /= len(samples)
            if loss < best_loss:
                best_loss = loss
                best_temperature = temperature
        return cls(best_temperature)


class EloPoissonPredictor:
    """The retained V1 baseline for comparison and regression testing."""

    HOME_ELO_ADVANTAGE = 65.0
    ELO_K = 24.0
    HALF_LIFE_DAYS = 365.0
    PRIOR_MATCH_WEIGHT = 6.0
    MAX_GOALS = 10

    def __init__(self, matches: list[Match]) -> None:
        if not matches:
            raise ValueError("At least one completed match is required")
        self.matches = sorted(matches, key=lambda match: match.played_on)
        self.ratings: defaultdict[str, float] = defaultdict(lambda: 1500.0)
        self.home_stats: defaultdict[str, WeightedStats] = defaultdict(WeightedStats)
        self.away_stats: defaultdict[str, WeightedStats] = defaultdict(WeightedStats)
        self.global_home_xg = 1.5
        self.global_away_xg = 1.2
        self._fit()

    def _fit(self) -> None:
        for match in self.matches:
            home_rating = self.ratings[match.home_team]
            away_rating = self.ratings[match.away_team]
            expected = 1.0 / (
                1.0 + 10.0 ** (-((home_rating + self.HOME_ELO_ADVANTAGE - away_rating) / 400.0))
            )
            actual = 1.0 if match.home_goals > match.away_goals else 0.0
            if match.home_goals == match.away_goals:
                actual = 0.5
            change = self.ELO_K * (actual - expected)
            self.ratings[match.home_team] += change
            self.ratings[match.away_team] -= change

        reference_day = self.matches[-1].played_on
        weighted_home_goals = 0.0
        weighted_away_goals = 0.0
        total_weight = 0.0
        for match in self.matches:
            age_days = max((reference_day - match.played_on).days, 0)
            weight = 0.5 ** (age_days / self.HALF_LIFE_DAYS)
            total_weight += weight
            weighted_home_goals += weight * match.home_goals
            weighted_away_goals += weight * match.away_goals
            home = self.home_stats[match.home_team]
            home.weight += weight
            home.goals_for += weight * match.home_goals
            home.goals_against += weight * match.away_goals
            away = self.away_stats[match.away_team]
            away.weight += weight
            away.goals_for += weight * match.away_goals
            away.goals_against += weight * match.home_goals

        if total_weight:
            self.global_home_xg = weighted_home_goals / total_weight
            self.global_away_xg = weighted_away_goals / total_weight

    def predict(
        self, home_team: str, away_team: str, allow_unseen: bool = False
    ) -> dict:
        known_teams = set(self.ratings)
        if not allow_unseen and (home_team not in known_teams or away_team not in known_teams):
            raise ValueError("Both teams must exist in the loaded EPL data")
        if home_team == away_team:
            raise ValueError("Home and away teams must be different")

        home = self.home_stats[home_team]
        away = self.away_stats[away_team]
        home_attack = self._shrunk_rate(home.goals_for, home.weight, self.global_home_xg) / self.global_home_xg
        home_defence = self._shrunk_rate(home.goals_against, home.weight, self.global_away_xg) / self.global_away_xg
        away_attack = self._shrunk_rate(away.goals_for, away.weight, self.global_away_xg) / self.global_away_xg
        away_defence = self._shrunk_rate(away.goals_against, away.weight, self.global_home_xg) / self.global_home_xg

        home_goals = self.global_home_xg * home_attack * away_defence
        away_goals = self.global_away_xg * away_attack * home_defence
        elo_gap = self.ratings[home_team] + self.HOME_ELO_ADVANTAGE - self.ratings[away_team]
        elo_multiplier = math.exp(max(-500.0, min(500.0, elo_gap)) / 1100.0)
        expected_home = _clamp(home_goals * elo_multiplier, 0.2, 4.0)
        expected_away = _clamp(away_goals / elo_multiplier, 0.2, 4.0)
        matrix = poisson_score_matrix(expected_home, expected_away, self.MAX_GOALS)
        return prediction_payload(
            home_team,
            away_team,
            expected_home,
            expected_away,
            matrix,
            model="Elo + Poisson V1",
            version="1.0",
            data_through=self.matches[-1].played_on,
            xg_enabled=False,
            factors=[],
        )

    def _shrunk_rate(self, goals: float, weight: float, prior: float) -> float:
        return (goals + self.PRIOR_MATCH_WEIGHT * prior) / (weight + self.PRIOR_MATCH_WEIGHT)


class DixonColesPredictor:
    """V2: pre-match features, recency weighting, context and low-score correction."""

    MAX_GOALS = 10
    RHO = -0.08

    def __init__(
        self,
        matches: list[Match],
        calibration_temperature: float = 1.0,
        xg_provider: Optional[XGProvider] = None,
    ) -> None:
        if not matches:
            raise ValueError("At least one completed match is required")
        self.matches = sorted(matches, key=lambda match: match.played_on)
        self.calibrator = TemperatureCalibrator(calibration_temperature)
        self.xg_provider = xg_provider or FootballDataXGProvider()
        self.xg_provider.load(self.matches)
        self.features = PreMatchFeatureBuilder(
            self.matches, xg_provider=self.xg_provider
        )

    def predict(
        self,
        home_team: str,
        away_team: str,
        as_of: Optional[date] = None,
    ) -> dict:
        known_teams = {
            team for match in self.matches for team in (match.home_team, match.away_team)
        }
        if home_team not in known_teams or away_team not in known_teams:
            raise ValueError("Both teams must exist in the loaded EPL data")
        as_of = as_of or (self.matches[-1].played_on + timedelta(days=1))
        snapshot = self.features.snapshot(home_team, away_team, as_of)
        expected_home, expected_away = self.expected_goals(snapshot)
        matrix = dixon_coles_score_matrix(
            expected_home, expected_away, rho=self.RHO, max_goals=self.MAX_GOALS
        )
        matrix = calibrate_score_matrix(matrix, self.calibrator)
        return prediction_payload(
            home_team,
            away_team,
            expected_home,
            expected_away,
            matrix,
            model="Enhanced Dixon-Coles V2",
            version="2.0",
            data_through=self.matches[-1].played_on,
            xg_enabled=self.xg_provider.available(),
            factors=self._explain(snapshot),
            extra_metadata={
                "calibration_temperature": self.calibrator.temperature,
                "dixon_coles_rho": self.RHO,
                "congestion_scope": "EPL matches only",
                "training_through": snapshot.training_through.isoformat()
                if snapshot.training_through else None,
            },
        )

    def expected_goals(self, snapshot: FeatureSnapshot) -> tuple[float, float]:
        value = snapshot.values
        league_home = value["league_home_goals"]
        league_away = value["league_away_goals"]
        home = league_home * value["home_attack_strength"] * value["away_defence_strength"]
        away = league_away * value["away_attack_strength"] * value["home_defence_strength"]

        home_form = 0.6 * value["home_recent5_gf"] + 0.4 * value["home_recent10_gf"]
        away_form = 0.6 * value["away_recent5_gf"] + 0.4 * value["away_recent10_gf"]
        home_opp_def = 0.6 * value["away_recent5_ga"] + 0.4 * value["away_recent10_ga"]
        away_opp_def = 0.6 * value["home_recent5_ga"] + 0.4 * value["home_recent10_ga"]
        home *= _safe_ratio(home_form, league_home) ** 0.16
        home *= _safe_ratio(home_opp_def, league_home) ** 0.12
        away *= _safe_ratio(away_form, league_away) ** 0.16
        away *= _safe_ratio(away_opp_def, league_away) ** 0.12

        elo_multiplier = math.exp(_clamp(value["elo_difference"], -500.0, 500.0) / 1500.0)
        home *= elo_multiplier
        away /= elo_multiplier

        rest_difference = _clamp(value["home_rest_days"] - value["away_rest_days"], -7, 7)
        congestion_difference = value["away_matches_7d"] - value["home_matches_7d"]
        context_multiplier = math.exp(0.010 * rest_difference + 0.025 * congestion_difference)
        home *= context_multiplier
        away /= context_multiplier

        if value["home_xg_samples"] >= 3 and value["away_xg_samples"] >= 3:
            xg_home = (value["home_recent5_xg"] + value["away_recent5_xga"]) / 2.0
            xg_away = (value["away_recent5_xg"] + value["home_recent5_xga"]) / 2.0
            home = 0.85 * home + 0.15 * xg_home
            away = 0.85 * away + 0.15 * xg_away
        else:
            home_shots = value["home_recent5_sot"]
            away_shots = value["away_recent5_sot"]
            if home_shots is not None and away_shots is not None:
                shot_difference = _clamp(home_shots - away_shots, -5.0, 5.0)
                home *= math.exp(shot_difference * 0.018)
                away /= math.exp(shot_difference * 0.018)

        return _clamp(home, 0.2, 4.2), _clamp(away, 0.2, 4.2)

    def _explain(self, snapshot: FeatureSnapshot) -> list[dict]:
        value = snapshot.values
        elo_gap = value["home_elo"] - value["away_elo"]
        if abs(elo_gap) < 15:
            elo_text = "Ratings are close"
        elif elo_gap > 0:
            elo_text = f"{snapshot.home_team} +{round(elo_gap)}"
        else:
            elo_text = f"{snapshot.away_team} +{round(abs(elo_gap))}"

        home_form = value["home_recent5_gf"] - value["home_recent5_ga"]
        away_form = value["away_recent5_gf"] - value["away_recent5_ga"]
        if abs(home_form - away_form) < 0.12:
            form_text = "Recent five-match form is similar"
        else:
            stronger = snapshot.home_team if home_form > away_form else snapshot.away_team
            form_text = f"{stronger} stronger over recent EPL matches"

        return [
            {"label": "Elo", "value": elo_text},
            {"label": "Recent form", "value": form_text},
            {
                "label": "Home attack",
                "value": "Above league average" if value["home_attack_strength"] >= 1.0 else "Below league average",
            },
            {
                "label": "Away defence",
                "value": "Concedes above league average" if value["away_defence_strength"] >= 1.0 else "Concedes below league average",
            },
            {
                "label": "Rest",
                "value": f"{snapshot.home_team} {value['home_rest_days']} days · {snapshot.away_team} {value['away_rest_days']} days",
            },
            {
                "label": "EPL congestion",
                "value": f"Previous 7 days: {value['home_matches_7d']} vs {value['away_matches_7d']} matches",
            },
        ]


def poisson_probability(goals: int, expected_goals: float) -> float:
    return math.exp(-expected_goals) * expected_goals**goals / math.factorial(goals)


def poisson_score_matrix(
    expected_home: float, expected_away: float, max_goals: int = 10
) -> list[tuple[int, int, float]]:
    matrix = [
        (home, away, poisson_probability(home, expected_home) * poisson_probability(away, expected_away))
        for home in range(max_goals + 1)
        for away in range(max_goals + 1)
    ]
    return normalize_matrix(matrix)


def dixon_coles_score_matrix(
    expected_home: float,
    expected_away: float,
    rho: float = -0.08,
    max_goals: int = 10,
) -> list[tuple[int, int, float]]:
    matrix: list[tuple[int, int, float]] = []
    for home in range(max_goals + 1):
        for away in range(max_goals + 1):
            probability = poisson_probability(home, expected_home) * poisson_probability(away, expected_away)
            if home == 0 and away == 0:
                probability *= 1.0 - expected_home * expected_away * rho
            elif home == 0 and away == 1:
                probability *= 1.0 + expected_home * rho
            elif home == 1 and away == 0:
                probability *= 1.0 + expected_away * rho
            elif home == 1 and away == 1:
                probability *= 1.0 - rho
            matrix.append((home, away, max(probability, 0.0)))
    return normalize_matrix(matrix)


def normalize_matrix(matrix: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    total = sum(probability for _, _, probability in matrix)
    if total <= 0:
        raise ValueError("Score probability matrix has no mass")
    return [(home, away, probability / total) for home, away, probability in matrix]


def outcome_probabilities(matrix: list[tuple[int, int, float]]) -> dict[str, float]:
    return {
        "home_win": sum(p for home, away, p in matrix if home > away),
        "draw": sum(p for home, away, p in matrix if home == away),
        "away_win": sum(p for home, away, p in matrix if home < away),
    }


def calibrate_score_matrix(
    matrix: list[tuple[int, int, float]], calibrator: TemperatureCalibrator
) -> list[tuple[int, int, float]]:
    raw = outcome_probabilities(matrix)
    calibrated = calibrator.apply(raw)
    adjusted = []
    for home, away, probability in matrix:
        outcome = "home_win" if home > away else "away_win" if home < away else "draw"
        adjusted.append((home, away, probability * calibrated[outcome] / max(raw[outcome], 1e-12)))
    return normalize_matrix(adjusted)


def prediction_payload(
    home_team: str,
    away_team: str,
    expected_home: float,
    expected_away: float,
    matrix: list[tuple[int, int, float]],
    model: str,
    version: str,
    data_through: date,
    xg_enabled: bool,
    factors: list[dict],
    extra_metadata: Optional[dict] = None,
) -> dict:
    probabilities = outcome_probabilities(matrix)
    ranked = sorted(matrix, key=lambda item: item[2], reverse=True)
    top_scorelines = [
        {"home": home, "away": away, "probability": probability}
        for home, away, probability in ranked[:5]
    ]
    return {
        "home_team": home_team,
        "away_team": away_team,
        "probabilities": probabilities,
        "expected_goals": {"home": expected_home, "away": expected_away},
        "most_likely_score": {"home": ranked[0][0], "away": ranked[0][1]},
        "top_scorelines": top_scorelines,
        "factors": factors,
        "model": model,
        "version": version,
        "data_through": data_through.isoformat(),
        "xg_enabled": xg_enabled,
        "metadata": extra_metadata or {},
    }


def bookmaker_probabilities(match: Match) -> Optional[dict[str, float]]:
    odds = (match.home_odds, match.draw_odds, match.away_odds)
    if any(value is None or value <= 1.0 for value in odds):
        return None
    raw = [1.0 / value for value in odds]  # type: ignore[operator]
    total = sum(raw)
    return dict(zip(OUTCOMES, (value / total for value in raw)))


def _safe_ratio(value: float, baseline: float) -> float:
    return _clamp(value / max(baseline, 0.1), 0.55, 1.80)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
