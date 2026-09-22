from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from .data import Match


@dataclass
class WeightedStats:
    weight: float = 0.0
    goals_for: float = 0.0
    goals_against: float = 0.0


class EloPoissonPredictor:
    """A deliberately compact baseline that combines Elo strength with Poisson goals."""

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
                1.0 + 10.0 ** (-((home_rating + self.HOME_ELO_ADVANTAGE) - away_rating) / 400.0)
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

    def predict(self, home_team: str, away_team: str) -> dict:
        known_teams = set(self.ratings)
        if home_team not in known_teams or away_team not in known_teams:
            raise ValueError("Both teams must exist in the loaded EPL data")
        if home_team == away_team:
            raise ValueError("Home and away teams must be different")

        home = self.home_stats[home_team]
        away = self.away_stats[away_team]

        home_attack = self._shrunk_rate(
            home.goals_for, home.weight, self.global_home_xg
        ) / self.global_home_xg
        home_defence = self._shrunk_rate(
            home.goals_against, home.weight, self.global_away_xg
        ) / self.global_away_xg
        away_attack = self._shrunk_rate(
            away.goals_for, away.weight, self.global_away_xg
        ) / self.global_away_xg
        away_defence = self._shrunk_rate(
            away.goals_against, away.weight, self.global_home_xg
        ) / self.global_home_xg

        poisson_home = self.global_home_xg * home_attack * away_defence
        poisson_away = self.global_away_xg * away_attack * home_defence

        elo_gap = (
            self.ratings[home_team]
            + self.HOME_ELO_ADVANTAGE
            - self.ratings[away_team]
        )
        elo_multiplier = math.exp(max(-500.0, min(500.0, elo_gap)) / 1100.0)
        expected_home_goals = self._clamp(poisson_home * elo_multiplier, 0.2, 4.0)
        expected_away_goals = self._clamp(poisson_away / elo_multiplier, 0.2, 4.0)

        matrix: list[tuple[int, int, float]] = []
        for home_goals in range(self.MAX_GOALS + 1):
            home_probability = self._poisson(home_goals, expected_home_goals)
            for away_goals in range(self.MAX_GOALS + 1):
                probability = home_probability * self._poisson(
                    away_goals, expected_away_goals
                )
                matrix.append((home_goals, away_goals, probability))

        covered_probability = sum(item[2] for item in matrix)
        home_win = sum(p for hg, ag, p in matrix if hg > ag) / covered_probability
        draw = sum(p for hg, ag, p in matrix if hg == ag) / covered_probability
        away_win = sum(p for hg, ag, p in matrix if hg < ag) / covered_probability
        likely_home, likely_away, _ = max(matrix, key=lambda item: item[2])

        return {
            "home_team": home_team,
            "away_team": away_team,
            "probabilities": {
                "home_win": home_win,
                "draw": draw,
                "away_win": away_win,
            },
            "expected_goals": {
                "home": expected_home_goals,
                "away": expected_away_goals,
            },
            "most_likely_score": {
                "home": likely_home,
                "away": likely_away,
            },
            "model": "Elo + Poisson V1",
        }

    def _shrunk_rate(self, goals: float, weight: float, prior: float) -> float:
        return (goals + self.PRIOR_MATCH_WEIGHT * prior) / (
            weight + self.PRIOR_MATCH_WEIGHT
        )

    @staticmethod
    def _poisson(goals: int, expected_goals: float) -> float:
        return math.exp(-expected_goals) * expected_goals**goals / math.factorial(goals)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

