from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .data import Match


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

