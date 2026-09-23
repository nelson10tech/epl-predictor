from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional, TYPE_CHECKING

from .data import Match

if TYPE_CHECKING:
    from .xg import XGProvider


@dataclass(frozen=True)
class Performance:
    played_on: date
    venue: str
    goals_for: float
    goals_against: float
    shots: Optional[float]
    shots_on_target: Optional[float]
    xg: Optional[float]
    xga: Optional[float]


@dataclass
class TeamHistory:
    elo: float = 1500.0
    matches: list[Performance] = field(default_factory=list)


@dataclass(frozen=True)
class FeatureSnapshot:
    as_of: date
    training_through: Optional[date]
    home_team: str
    away_team: str
    values: dict

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "training_through": self.training_through.isoformat()
            if self.training_through else None,
            "home_team": self.home_team,
            "away_team": self.away_team,
            **self.values,
        }


class PreMatchFeatureBuilder:
    """Stateful feature builder that never observes the target match or future rows."""

    HOME_ELO_ADVANTAGE = 65.0
    ELO_K = 24.0
    HALF_LIFE_DAYS = 240.0
    PRIOR_MATCH_WEIGHT = 8.0

    def __init__(
        self,
        matches: Optional[list[Match]] = None,
        xg_provider: Optional["XGProvider"] = None,
    ) -> None:
        self.teams: defaultdict[str, TeamHistory] = defaultdict(TeamHistory)
        self.xg_provider = xg_provider
        self.latest_processed_date: Optional[date] = None
        self.match_count = 0
        self.home_goals = 0.0
        self.away_goals = 0.0
        self.home_shots = 0.0
        self.away_shots = 0.0
        self.shot_match_count = 0
        if matches:
            self.fit(matches)

    @property
    def league_home_goals(self) -> float:
        return self.home_goals / self.match_count if self.match_count else 1.50

    @property
    def league_away_goals(self) -> float:
        return self.away_goals / self.match_count if self.match_count else 1.20

    @property
    def league_shots(self) -> float:
        if not self.shot_match_count:
            return 12.0
        return (self.home_shots + self.away_shots) / (2.0 * self.shot_match_count)

    def fit(self, matches: list[Match]) -> "PreMatchFeatureBuilder":
        grouped: defaultdict[date, list[Match]] = defaultdict(list)
        for match in sorted(matches, key=lambda item: item.played_on):
            grouped[match.played_on].append(match)
        for played_on in sorted(grouped):
            self.update_day(grouped[played_on])
        return self

    def snapshot(self, home_team: str, away_team: str, as_of: date) -> FeatureSnapshot:
        if home_team == away_team:
            raise ValueError("Home and away teams must be different")
        if self.latest_processed_date is not None and self.latest_processed_date >= as_of:
            raise ValueError("Feature state must end before the predicted match date")

        home = self.teams[home_team]
        away = self.teams[away_team]
        league_home = self.league_home_goals
        league_away = self.league_away_goals

        home_venue = self._summary(home.matches, as_of, venue="home")
        away_venue = self._summary(away.matches, as_of, venue="away")
        home_5 = self._summary(home.matches, as_of, limit=5)
        away_5 = self._summary(away.matches, as_of, limit=5)
        home_10 = self._summary(home.matches, as_of, limit=10)
        away_10 = self._summary(away.matches, as_of, limit=10)

        values = {
            "home_elo": home.elo,
            "away_elo": away.elo,
            "elo_difference": home.elo + self.HOME_ELO_ADVANTAGE - away.elo,
            "league_home_goals": league_home,
            "league_away_goals": league_away,
            "home_attack_strength": self._strength(home_venue, "goals_for", league_home),
            "home_defence_strength": self._strength(home_venue, "goals_against", league_away),
            "away_attack_strength": self._strength(away_venue, "goals_for", league_away),
            "away_defence_strength": self._strength(away_venue, "goals_against", league_home),
            "home_recent5_gf": self._rate(home_5, "goals_for", league_home),
            "home_recent5_ga": self._rate(home_5, "goals_against", league_away),
            "away_recent5_gf": self._rate(away_5, "goals_for", league_away),
            "away_recent5_ga": self._rate(away_5, "goals_against", league_home),
            "home_recent10_gf": self._rate(home_10, "goals_for", league_home),
            "home_recent10_ga": self._rate(home_10, "goals_against", league_away),
            "away_recent10_gf": self._rate(away_10, "goals_for", league_away),
            "away_recent10_ga": self._rate(away_10, "goals_against", league_home),
            "home_recent5_shots": self._optional_rate(home_5, "shots"),
            "away_recent5_shots": self._optional_rate(away_5, "shots"),
            "home_recent5_sot": self._optional_rate(home_5, "shots_on_target"),
            "away_recent5_sot": self._optional_rate(away_5, "shots_on_target"),
            "home_recent5_xg": self._optional_rate(home_5, "xg"),
            "home_recent5_xga": self._optional_rate(home_5, "xga"),
            "away_recent5_xg": self._optional_rate(away_5, "xg"),
            "away_recent5_xga": self._optional_rate(away_5, "xga"),
            "home_recent10_xg": self._optional_rate(home_10, "xg"),
            "home_recent10_xga": self._optional_rate(home_10, "xga"),
            "away_recent10_xg": self._optional_rate(away_10, "xg"),
            "away_recent10_xga": self._optional_rate(away_10, "xga"),
            "home_xg_samples": self._available_count(home_10, "xg"),
            "away_xg_samples": self._available_count(away_10, "xg"),
            "home_rest_days": self._rest_days(home.matches, as_of),
            "away_rest_days": self._rest_days(away.matches, as_of),
            "home_matches_7d": self._congestion(home.matches, as_of, 7),
            "away_matches_7d": self._congestion(away.matches, as_of, 7),
            "home_matches_14d": self._congestion(home.matches, as_of, 14),
            "away_matches_14d": self._congestion(away.matches, as_of, 14),
            "home_sample_count": len(home.matches),
            "away_sample_count": len(away.matches),
        }
        return FeatureSnapshot(
            as_of=as_of,
            training_through=self.latest_processed_date,
            home_team=home_team,
            away_team=away_team,
            values=values,
        )

    def update_day(self, matches: list[Match]) -> None:
        if not matches:
            return
        played_on = matches[0].played_on
        if any(match.played_on != played_on for match in matches):
            raise ValueError("update_day requires matches from one calendar date")
        if self.latest_processed_date is not None and played_on <= self.latest_processed_date:
            raise ValueError("Matches must be processed in strictly chronological day order")

        elo_changes: defaultdict[str, float] = defaultdict(float)
        for match in matches:
            home_rating = self.teams[match.home_team].elo
            away_rating = self.teams[match.away_team].elo
            expected = 1.0 / (
                1.0 + 10.0 ** (-((home_rating + self.HOME_ELO_ADVANTAGE - away_rating) / 400.0))
            )
            actual = 1.0 if match.home_goals > match.away_goals else 0.0
            if match.home_goals == match.away_goals:
                actual = 0.5
            change = self.ELO_K * (actual - expected)
            elo_changes[match.home_team] += change
            elo_changes[match.away_team] -= change

        for team, change in elo_changes.items():
            self.teams[team].elo += change

        for match in matches:
            provider_xg = (
                self.xg_provider.get_match_xg(match) if self.xg_provider else None
            )
            home_xg, away_xg = provider_xg or (match.home_xg, match.away_xg)
            self.teams[match.home_team].matches.append(Performance(
                played_on=match.played_on,
                venue="home",
                goals_for=match.home_goals,
                goals_against=match.away_goals,
                shots=match.home_shots,
                shots_on_target=match.home_shots_on_target,
                xg=home_xg,
                xga=away_xg,
            ))
            self.teams[match.away_team].matches.append(Performance(
                played_on=match.played_on,
                venue="away",
                goals_for=match.away_goals,
                goals_against=match.home_goals,
                shots=match.away_shots,
                shots_on_target=match.away_shots_on_target,
                xg=away_xg,
                xga=home_xg,
            ))
            self.match_count += 1
            self.home_goals += match.home_goals
            self.away_goals += match.away_goals
            if match.home_shots is not None and match.away_shots is not None:
                self.home_shots += match.home_shots
                self.away_shots += match.away_shots
                self.shot_match_count += 1

        self.latest_processed_date = played_on

    def _summary(
        self,
        history: list[Performance],
        as_of: date,
        limit: Optional[int] = None,
        venue: Optional[str] = None,
    ) -> list[tuple[Performance, float]]:
        eligible = [
            performance for performance in history
            if performance.played_on < as_of and (venue is None or performance.venue == venue)
        ]
        if limit is not None:
            eligible = eligible[-limit:]
        return [
            (
                performance,
                0.5 ** (max((as_of - performance.played_on).days, 0) / self.HALF_LIFE_DAYS),
            )
            for performance in eligible
        ]

    def _rate(self, summary: list[tuple[Performance, float]], field_name: str, prior: float) -> float:
        weighted = sum(getattr(item, field_name) * weight for item, weight in summary)
        weight = sum(weight for _, weight in summary)
        return (weighted + self.PRIOR_MATCH_WEIGHT * prior) / (
            weight + self.PRIOR_MATCH_WEIGHT
        )

    def _strength(self, summary: list[tuple[Performance, float]], field_name: str, prior: float) -> float:
        return self._rate(summary, field_name, prior) / max(prior, 0.1)

    @staticmethod
    def _optional_rate(
        summary: list[tuple[Performance, float]], field_name: str
    ) -> Optional[float]:
        values = [
            (getattr(item, field_name), weight) for item, weight in summary
            if getattr(item, field_name) is not None
        ]
        if not values:
            return None
        return sum(value * weight for value, weight in values) / sum(weight for _, weight in values)

    @staticmethod
    def _available_count(summary: list[tuple[Performance, float]], field_name: str) -> int:
        return sum(1 for item, _ in summary if getattr(item, field_name) is not None)

    @staticmethod
    def _rest_days(history: list[Performance], as_of: date) -> int:
        prior = [item.played_on for item in history if item.played_on < as_of]
        return min((as_of - max(prior)).days, 30) if prior else 7

    @staticmethod
    def _congestion(history: list[Performance], as_of: date, days: int) -> int:
        lower = as_of - timedelta(days=days)
        return sum(1 for item in history if lower <= item.played_on < as_of)
