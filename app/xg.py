from __future__ import annotations

from abc import ABC, abstractmethod
import csv
from datetime import date
from pathlib import Path
from typing import Optional

from .data import Match
from .normalization import normalize_team_name


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIVETHIRTYEIGHT_CACHE = (
    PROJECT_ROOT / "data" / "advanced" / "fivethirtyeight_epl_xg_2021_2023.csv"
)


class XGProvider(ABC):
    """Optional expected-goals data source used only when real xG is present."""

    @abstractmethod
    def load(self, matches: list[Match]) -> None:
        raise NotImplementedError

    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_match_xg(self, match: Match) -> Optional[tuple[float, float]]:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict:
        raise NotImplementedError


class FootballDataXGProvider(XGProvider):
    """Reads HxG/AxG only when Football-Data.co.uk actually supplies them."""

    def __init__(self) -> None:
        self._values: dict[tuple, tuple[float, float]] = {}
        self._match_count = 0

    def load(self, matches: list[Match]) -> None:
        self._values = {
            match.identity: (match.home_xg, match.away_xg)
            for match in matches
            if match.home_xg is not None and match.away_xg is not None
        }
        self._match_count = len(self._values)

    def available(self) -> bool:
        return self._match_count > 0

    def get_match_xg(self, match: Match) -> Optional[tuple[float, float]]:
        return self._values.get(match.identity)

    def metadata(self) -> dict:
        return {
            "enabled": self.available(),
            "provider": "Football-Data.co.uk HxG/AxG" if self.available() else None,
            "matches_with_xg": self._match_count,
            "limitation": (
                "Real xG is used only for rows where Football-Data.co.uk exposes HxG/AxG; "
                "shots and shots on target remain the fallback indicators."
            ),
        }


class NullXGProvider(XGProvider):
    def load(self, matches: list[Match]) -> None:
        return None

    def available(self) -> bool:
        return False

    def get_match_xg(self, match: Match) -> Optional[tuple[float, float]]:
        return None

    def metadata(self) -> dict:
        return {
            "enabled": False,
            "provider": None,
            "matches_with_xg": 0,
            "limitation": "No reliable real-xG fields are available; no xG is fabricated.",
        }


class FiveThirtyEightXGProvider(XGProvider):
    """Read the committed CC BY 4.0 FiveThirtyEight EPL xG archive."""

    def __init__(self, cache_path: Optional[Path] = None) -> None:
        self.cache_path = cache_path or DEFAULT_FIVETHIRTYEIGHT_CACHE
        self._values: dict[tuple[date, str, str], tuple[float, float]] = {}

    def load(self, matches: list[Match]) -> None:
        self._values = {}
        try:
            with self.cache_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    key = (
                        date.fromisoformat(row["date"]),
                        normalize_team_name(row["home_team"]),
                        normalize_team_name(row["away_team"]),
                    )
                    self._values[key] = (float(row["home_xg"]), float(row["away_xg"]))
        except (OSError, TypeError, ValueError, KeyError):
            self._values = {}

    def available(self) -> bool:
        return bool(self._values)

    def get_match_xg(self, match: Match) -> Optional[tuple[float, float]]:
        return self._values.get((match.played_on, match.home_team, match.away_team))

    def metadata(self) -> dict:
        return {
            "enabled": self.available(),
            "provider": "FiveThirtyEight Soccer SPI archive" if self.available() else None,
            "source": (
                "FiveThirtyEight Soccer SPI, archived 2023-06-14 via the Internet Archive"
            ),
            "source_url": (
                "https://web.archive.org/web/20230625093532if_/"
                "https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv"
            ),
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "matches_with_xg": len(self._values),
            "coverage": "EPL 2021/22 and 2022/23" if self.available() else None,
            "cache": self.cache_path.name,
            "limitation": (
                "The FiveThirtyEight feed ended in 2023. Later seasons use only genuine "
                "Football-Data HxG/AxG rows when present; no xG is fabricated."
            ),
        }


class CompositeXGProvider(XGProvider):
    """Prefer Football-Data xG, then fall back to the archived 538 source."""

    def __init__(self, providers: Optional[list[XGProvider]] = None) -> None:
        self.providers = providers or [FootballDataXGProvider(), FiveThirtyEightXGProvider()]

    def load(self, matches: list[Match]) -> None:
        for provider in self.providers:
            provider.load(matches)

    def available(self) -> bool:
        return any(provider.available() for provider in self.providers)

    def get_match_xg(self, match: Match) -> Optional[tuple[float, float]]:
        for provider in self.providers:
            value = provider.get_match_xg(match)
            if value is not None:
                return value
        return None

    def metadata(self) -> dict:
        items = [provider.metadata() for provider in self.providers]
        return {
            "enabled": self.available(),
            "provider": "Composite genuine-xG provider" if self.available() else None,
            "matches_with_xg": sum(item.get("matches_with_xg", 0) for item in items),
            "sources": items,
            "limitation": (
                "Coverage is source-dependent and may be sparse after 2022/23; missing "
                "values stay missing and are never replaced by model estimates."
            ),
        }
