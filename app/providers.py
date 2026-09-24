from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from .data import Fixture, current_season_code
from .normalization import normalize_team_name


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE_CACHE = PROJECT_ROOT / "data" / "cache" / "upcoming_fixtures.json"
OPENFOOTBALL_URL = (
    "https://raw.githubusercontent.com/openfootball/football.json/master/"
    "{season}/en.1.json"
)


class FixtureProvider(ABC):
    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def fetch_upcoming(
        self, now: Optional[datetime] = None, known_teams: Optional[set[str]] = None
    ) -> list[Fixture]:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict:
        raise NotImplementedError


class OpenFootballFixtureProvider(FixtureProvider):
    """Public-domain fixture feed with a committed/local cache fallback."""

    def __init__(self, cache_path: Optional[Path] = None, timeout: int = 12) -> None:
        self.cache_path = cache_path or DEFAULT_FIXTURE_CACHE
        self.timeout = timeout
        self._fixtures: list[Fixture] = []
        self._last_success: Optional[str] = None
        self._last_error: Optional[str] = None
        self._source_url: Optional[str] = None
        self.load_cache()

    @staticmethod
    def _season_parts(season_code: str) -> tuple[int, int]:
        start = 2000 + int(season_code[:2])
        return start, start + 1

    def _url(self, now: datetime) -> str:
        season_code = current_season_code(now.date())
        start, end = self._season_parts(season_code)
        return OPENFOOTBALL_URL.format(season=f"{start}-{str(end)[-2:]}")

    def available(self) -> bool:
        return bool(self._fixtures)

    def load_cache(self) -> list[Fixture]:
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self._last_success = payload.get("fetched_at")
            self._source_url = payload.get("source_url")
            self._fixtures = [
                Fixture(
                    played_on=date.fromisoformat(item["date"]),
                    season=item["season"],
                    home_team=item["home_team"],
                    away_team=item["away_team"],
                    kickoff=item.get("kickoff"),
                )
                for item in payload.get("fixtures", [])
            ]
        except (OSError, TypeError, ValueError, KeyError):
            self._fixtures = []
        return list(self._fixtures)

    def fetch_upcoming(
        self, now: Optional[datetime] = None, known_teams: Optional[set[str]] = None
    ) -> list[Fixture]:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        url = self._url(now)
        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": "EPL-Predictor/3.0"},
            )
            response.raise_for_status()
            payload = response.json()
            matches = payload.get("matches", [])
            fixtures: list[Fixture] = []
            season = current_season_code(now.date())
            today = now.date()
            for item in matches:
                # A score means the fixture has already been played.  Missing dates or
                # times are not guessed.
                if item.get("score") is not None:
                    continue
                if not item.get("date") or not item.get("time"):
                    continue
                played_on = date.fromisoformat(item["date"])
                if played_on < today:
                    continue
                home = normalize_team_name(item.get("team1", ""))
                away = normalize_team_name(item.get("team2", ""))
                if not home or not away or home == away:
                    continue
                if known_teams and (home not in known_teams or away not in known_teams):
                    continue
                fixtures.append(Fixture(played_on, season, home, away, item["time"]))
            if not fixtures:
                raise ValueError("fixture feed contained no valid future EPL fixtures")
            self._fixtures = sorted(
                fixtures, key=lambda item: (item.played_on, item.kickoff or "", item.home_team)
            )
            self._last_success = now.astimezone(timezone.utc).isoformat()
            self._last_error = None
            self._source_url = url
            self._write_cache()
        except (
            requests.RequestException,
            ValueError,
            TypeError,
            KeyError,
            OSError,
            RuntimeError,
        ) as exc:
            self._last_error = str(exc)
            self.load_cache()
        return list(self._fixtures)

    def _write_cache(self) -> None:
        payload = {
            "source": "openfootball/football.json",
            "source_url": self._source_url,
            "license": "Public Domain",
            "fetched_at": self._last_success,
            "fixtures": [
                {
                    "date": item.played_on.isoformat(),
                    "season": item.season,
                    "kickoff": item.kickoff,
                    "home_team": item.home_team,
                    "away_team": item.away_team,
                }
                for item in self._fixtures
            ],
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.cache_path)

    def metadata(self) -> dict:
        return {
            "available": self.available(),
            "provider": "openfootball/football.json" if self.available() else None,
            "license": "Public Domain",
            "source_url": self._source_url,
            "cached_fixtures": len(self._fixtures),
            "last_success": self._last_success,
            "last_error": self._last_error,
            "limitation": (
                "The provider is optional and serves its last valid cache when the public "
                "feed is unavailable; dates and kickoff times are never fabricated."
            ),
        }


class AllCompetitionScheduleProvider(ABC):
    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def team_dates_before(self, team: str, as_of: date) -> list[date]:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict:
        raise NotImplementedError


class NullAllCompetitionScheduleProvider(AllCompetitionScheduleProvider):
    def available(self) -> bool:
        return False

    def team_dates_before(self, team: str, as_of: date) -> list[date]:
        return []

    def metadata(self) -> dict:
        return {
            "available": False,
            "congestion_scope": "EPL only",
            "provider": None,
            "limitation": (
                "No sufficiently complete, stable, public all-competition schedule source "
                "is enabled; cup and European matches are not invented."
            ),
        }


class PlayerAvailabilityProvider(ABC):
    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def features(self, team: str, as_of: date) -> dict:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict:
        raise NotImplementedError


class NullPlayerAvailabilityProvider(PlayerAvailabilityProvider):
    def available(self) -> bool:
        return False

    def features(self, team: str, as_of: date) -> dict:
        return {}

    def metadata(self) -> dict:
        return {
            "available": False,
            "provider": None,
            "limitation": (
                "No reliable free structured injuries, suspensions and confirmed-lineup "
                "source is enabled; rumours and subjective player scores are excluded."
            ),
        }
