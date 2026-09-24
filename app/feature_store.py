from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

from .data import Match
from .features import FeatureSnapshot, PreMatchFeatureBuilder
from .providers import (
    AllCompetitionScheduleProvider,
    NullAllCompetitionScheduleProvider,
    NullPlayerAvailabilityProvider,
    PlayerAvailabilityProvider,
)
from .xg import CompositeXGProvider, XGProvider


ML_FEATURE_NAMES = (
    "home_elo",
    "away_elo",
    "elo_difference",
    "league_home_goals",
    "league_away_goals",
    "home_attack_strength",
    "home_defence_strength",
    "away_attack_strength",
    "away_defence_strength",
    "home_recent5_gf",
    "home_recent5_ga",
    "away_recent5_gf",
    "away_recent5_ga",
    "home_recent10_gf",
    "home_recent10_ga",
    "away_recent10_gf",
    "away_recent10_ga",
    "home_recent5_shots",
    "away_recent5_shots",
    "home_recent5_sot",
    "away_recent5_sot",
    "home_xg_for_5",
    "home_xg_against_5",
    "away_xg_for_5",
    "away_xg_against_5",
    "home_xg_for_10",
    "home_xg_against_10",
    "away_xg_for_10",
    "away_xg_against_10",
    "home_xg_difference",
    "away_xg_difference",
    "home_xg_trend",
    "away_xg_trend",
    "home_finishing_difference",
    "away_finishing_difference",
    "home_rest_days",
    "away_rest_days",
    "home_matches_7d",
    "away_matches_7d",
    "home_matches_14d",
    "away_matches_14d",
    "home_sample_count",
    "away_sample_count",
    "home_season_matches",
    "away_season_matches",
    "early_season",
)


@dataclass(frozen=True)
class FeatureRow:
    match: Match
    snapshot: FeatureSnapshot

    @property
    def result(self) -> str:
        return self.match.result

    def vector(self) -> list[float]:
        return feature_vector(self.snapshot.values)


def feature_vector(values: dict) -> list[float]:
    return [
        float(values[name]) if values.get(name) is not None else float("nan")
        for name in ML_FEATURE_NAMES
    ]


class ChronologicalFeatureStore:
    """Materialize feature rows before revealing each same-day result batch."""

    def __init__(
        self,
        matches: Iterable[Match],
        xg_provider: Optional[XGProvider] = None,
        schedule_provider: Optional[AllCompetitionScheduleProvider] = None,
        player_provider: Optional[PlayerAvailabilityProvider] = None,
    ) -> None:
        self.matches = sorted(matches, key=lambda item: (item.played_on, item.home_team))
        self.xg_provider = xg_provider or CompositeXGProvider()
        self.schedule_provider = schedule_provider or NullAllCompetitionScheduleProvider()
        self.player_provider = player_provider or NullPlayerAvailabilityProvider()
        self.xg_provider.load(self.matches)
        self.builder = PreMatchFeatureBuilder(xg_provider=self.xg_provider)
        self.rows = self._build()

    def _build(self) -> list[FeatureRow]:
        grouped: defaultdict[date, list[Match]] = defaultdict(list)
        for match in self.matches:
            grouped[match.played_on].append(match)
        rows: list[FeatureRow] = []
        for played_on in sorted(grouped):
            day = grouped[played_on]
            for match in day:
                snapshot = self.builder.snapshot(
                    match.home_team,
                    match.away_team,
                    match.played_on,
                    season=match.season,
                )
                rows.append(FeatureRow(match, snapshot))
            # Critical leakage barrier: the entire day's features exist before any of
            # that day's results are applied to Elo or rolling state.
            self.builder.update_day(day)
        return rows

    def live_snapshot(self, home_team: str, away_team: str, as_of: date) -> FeatureSnapshot:
        return self.builder.snapshot(
            home_team,
            away_team,
            as_of,
        )

    def metadata(self) -> dict:
        return {
            "rows": len(self.rows),
            "features": list(ML_FEATURE_NAMES),
            "chronological": True,
            "same_day_results_withheld": True,
            "xg": self.xg_provider.metadata(),
            "all_competition_schedule": self.schedule_provider.metadata(),
            "player_availability": self.player_provider.metadata(),
        }
