from __future__ import annotations

import csv
import io
import json
import os
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests


FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
REFRESH_SOURCE = "Football-Data.co.uk"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REFRESH_STATE_FILE = ".refresh_status.json"
SNAPSHOT_STATE_FILE = "snapshot_metadata.json"

BASE_FIELDS = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
OPTIONAL_FIELDS = [
    "HS", "AS", "HST", "AST", "HxG", "AxG",
    "B365H", "B365D", "B365A", "AvgH", "AvgD", "AvgA", "PSH", "PSD", "PSA",
    "HPossession", "APossession", "PossessionH", "PossessionA",
]

_STARTUP_REFRESH_LOCK = threading.Lock()
_STARTUP_REFRESHED_DIRS: set[str] = set()


def _optional_float(value: Optional[str]) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Match:
    played_on: date
    season: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    kickoff: Optional[str] = None
    home_shots: Optional[float] = None
    away_shots: Optional[float] = None
    home_shots_on_target: Optional[float] = None
    away_shots_on_target: Optional[float] = None
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    home_odds: Optional[float] = None
    draw_odds: Optional[float] = None
    away_odds: Optional[float] = None

    @property
    def result(self) -> str:
        if self.home_goals > self.away_goals:
            return "H"
        if self.home_goals < self.away_goals:
            return "A"
        return "D"

    @property
    def identity(self) -> tuple[str, date, str, str]:
        return (self.season, self.played_on, self.home_team, self.away_team)


@dataclass(frozen=True)
class Fixture:
    played_on: date
    season: str
    home_team: str
    away_team: str
    kickoff: Optional[str] = None


def current_season_code(today: Optional[date] = None) -> str:
    today = today or date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_codes(count: int, today: Optional[date] = None) -> list[str]:
    active = current_season_code(today)
    start_year = int(active[:2])
    return [
        f"{(start_year - offset) % 100:02d}{(start_year - offset + 1) % 100:02d}"
        for offset in reversed(range(count))
    ]


def _parse_date(value: str) -> date:
    for pattern in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(value.strip(), pattern).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date: {value}")


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


class MatchRepository:
    def __init__(
        self,
        history_seasons: int = 6,
        timeout: int = 12,
        data_dir: Optional[Path] = None,
    ) -> None:
        configured_dir = os.environ.get("EPL_DATA_DIR")
        self.data_dir = data_dir or (Path(configured_dir) if configured_dir else DEFAULT_DATA_DIR)
        self.history_seasons = history_seasons
        self.timeout = timeout
        self.matches: list[Match] = []
        self.fixtures: list[Fixture] = []
        self.loaded_files: list[Path] = []
        self.last_updated: Optional[datetime] = None
        self.last_refresh_attempt: Optional[datetime] = None
        self.last_refresh_success: Optional[datetime] = None
        self.last_refresh_error: Optional[str] = None
        self.data_stale = False
        self._load_refresh_state()

    @property
    def data_through(self) -> Optional[date]:
        return self.matches[-1].played_on if self.matches else None

    @property
    def active_season(self) -> Optional[str]:
        return max((match.season for match in self.matches), default=None)

    def load(self) -> list[Match]:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        matches: list[Match] = []
        fixtures: list[Fixture] = []
        loaded_files: list[Path] = []

        for path in sorted(self.data_dir.glob("*_E0.csv")):
            season = path.name.split("_", 1)[0]
            file_matches, file_fixtures = self._read_file(path, season)
            if file_matches or file_fixtures:
                matches.extend(file_matches)
                fixtures.extend(file_fixtures)
                loaded_files.append(path)

        if not matches:
            raise RuntimeError(
                "No EPL match data found. Restore the committed data snapshot before starting."
            )

        self.matches = sorted(
            matches, key=lambda item: (item.played_on, item.kickoff or "", item.home_team)
        )
        self.fixtures = sorted(
            fixtures, key=lambda item: (item.played_on, item.kickoff or "", item.home_team)
        )
        self.loaded_files = loaded_files
        self.last_updated = datetime.fromtimestamp(
            max(path.stat().st_mtime for path in loaded_files), tz=timezone.utc
        )
        if self.last_refresh_success is None:
            self.last_refresh_success = self.last_updated
        return self.matches

    def ensure_fresh_once(
        self, max_age_hours: float = 12.0, now: Optional[datetime] = None
    ) -> bool:
        """Perform at most one startup freshness check per data directory and process."""
        key = str(self.data_dir.resolve())
        with _STARTUP_REFRESH_LOCK:
            if key in _STARTUP_REFRESHED_DIRS:
                self._update_stale_flag(max_age_hours, now)
                return False
            _STARTUP_REFRESHED_DIRS.add(key)
            return self.ensure_fresh(max_age_hours=max_age_hours, now=now)

    def ensure_fresh(
        self, max_age_hours: float = 12.0, now: Optional[datetime] = None
    ) -> bool:
        now = now or datetime.now(timezone.utc)
        self._update_stale_flag(max_age_hours, now)
        if not self.data_stale:
            return False

        self.last_refresh_attempt = now
        self.last_refresh_error = None
        try:
            self.refresh(now=now)
            self.data_stale = False
            return True
        except Exception as exc:  # startup must keep serving the committed snapshot
            self.last_refresh_error = str(exc)
            self.data_stale = True
            self._write_refresh_state()
            return False

    def refresh(self, now: Optional[datetime] = None) -> dict:
        now = now or datetime.now(timezone.utc)
        self.last_refresh_attempt = now
        self.data_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[str] = []
        failures: list[str] = []
        staged: list[tuple[Path, bytes]] = []

        headers = {
            "User-Agent": "EPL-Predictor/2.0 (+https://github.com/nelson10tech/epl-predictor)"
        }
        seasons = season_codes(self.history_seasons)
        with requests.Session() as session:
            for season in seasons:
                url = FOOTBALL_DATA_URL.format(season=season)
                try:
                    response = session.get(url, headers=headers, timeout=self.timeout)
                    response.raise_for_status()
                    normalized = self._normalized_csv(response.content)
                    destination = self.data_dir / f"{season}_E0.csv"
                    staged.append((destination, normalized))
                    downloaded.append(season)
                except (requests.RequestException, OSError, ValueError) as exc:
                    failures.append(f"{season}: {exc}")

        if not downloaded:
            raise RuntimeError("Unable to download EPL data from Football-Data.co.uk")
        if seasons[-1] not in downloaded:
            raise RuntimeError(
                f"Unable to refresh current EPL season {seasons[-1]}; keeping cached data"
            )

        for destination, content in staged:
            temporary = destination.with_suffix(".csv.tmp")
            temporary.write_bytes(content)
            temporary.replace(destination)

        self.load()
        self.last_refresh_success = now
        self.last_refresh_error = None
        self.data_stale = False
        self._write_refresh_state(snapshot=True)
        return {
            "downloaded_seasons": downloaded,
            "failed_seasons": failures,
            "matches": len(self.matches),
            **self.freshness_metadata(),
        }

    def freshness_metadata(self) -> dict:
        return {
            "data_through": self.data_through.isoformat() if self.data_through else None,
            "last_refresh_attempt": self.last_refresh_attempt.isoformat()
            if self.last_refresh_attempt else None,
            "last_refresh_success": self.last_refresh_success.isoformat()
            if self.last_refresh_success else None,
            "data_stale": self.data_stale,
            "refresh_source": REFRESH_SOURCE,
            "refresh_error": self.last_refresh_error,
        }

    def active_teams(self) -> list[str]:
        latest_season = self.active_season
        names = {
            team for match in self.matches if match.season == latest_season
            for team in (match.home_team, match.away_team)
        }
        return sorted(names)

    def latest_results(self, limit: int = 30) -> list[Match]:
        latest = [match for match in self.matches if match.season == self.active_season]
        return sorted(latest, key=lambda item: item.played_on, reverse=True)[:limit]

    def upcoming_fixtures(self, limit: int = 30) -> list[Fixture]:
        today = date.today()
        return [fixture for fixture in self.fixtures if fixture.played_on >= today][:limit]

    def _update_stale_flag(
        self, max_age_hours: float, now: Optional[datetime] = None
    ) -> None:
        now = now or datetime.now(timezone.utc)
        reference = self.last_refresh_success or self.last_updated
        self.data_stale = reference is None or now - reference > timedelta(hours=max_age_hours)

    def _load_refresh_state(self) -> None:
        for filename in (REFRESH_STATE_FILE, SNAPSHOT_STATE_FILE):
            path = self.data_dir / filename
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.last_refresh_attempt = _parse_iso_datetime(payload.get("last_refresh_attempt"))
                self.last_refresh_success = _parse_iso_datetime(payload.get("last_refresh_success"))
                self.last_refresh_error = payload.get("refresh_error")
                return
            except (OSError, ValueError, TypeError):
                continue

    def _write_refresh_state(self, snapshot: bool = False) -> None:
        payload = {
            "last_refresh_attempt": self.last_refresh_attempt.isoformat()
            if self.last_refresh_attempt else None,
            "last_refresh_success": self.last_refresh_success.isoformat()
            if self.last_refresh_success else None,
            "refresh_error": self.last_refresh_error,
            "refresh_source": REFRESH_SOURCE,
        }
        filenames = [REFRESH_STATE_FILE]
        if snapshot:
            filenames.append(SNAPSHOT_STATE_FILE)
        for filename in filenames:
            path = self.data_dir / filename
            temporary = path.with_suffix(path.suffix + ".tmp")
            try:
                temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                temporary.replace(path)
            except OSError:
                continue

    @staticmethod
    def _read_file(path: Path, season: str) -> tuple[list[Match], list[Fixture]]:
        matches: list[Match] = []
        fixtures: list[Fixture] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if not row.get("Date") or not row.get("HomeTeam") or not row.get("AwayTeam"):
                    continue
                try:
                    played_on = _parse_date(row["Date"])
                except ValueError:
                    continue

                home_team = row["HomeTeam"].strip()
                away_team = row["AwayTeam"].strip()
                kickoff = row.get("Time", "").strip() or None
                if row.get("FTHG", "").strip() == "" or row.get("FTAG", "").strip() == "":
                    fixtures.append(Fixture(played_on, season, home_team, away_team, kickoff))
                    continue

                try:
                    odds = MatchRepository._extract_odds(row)
                    matches.append(Match(
                        played_on=played_on,
                        season=season,
                        home_team=home_team,
                        away_team=away_team,
                        home_goals=int(float(row["FTHG"])),
                        away_goals=int(float(row["FTAG"])),
                        kickoff=kickoff,
                        home_shots=_optional_float(row.get("HS")),
                        away_shots=_optional_float(row.get("AS")),
                        home_shots_on_target=_optional_float(row.get("HST")),
                        away_shots_on_target=_optional_float(row.get("AST")),
                        home_xg=_optional_float(row.get("HxG")),
                        away_xg=_optional_float(row.get("AxG")),
                        home_odds=odds[0], draw_odds=odds[1], away_odds=odds[2],
                    ))
                except (TypeError, ValueError):
                    continue
        return matches, fixtures

    @staticmethod
    def _extract_odds(row: dict) -> tuple[Optional[float], Optional[float], Optional[float]]:
        for columns in (
            ("B365H", "B365D", "B365A"),
            ("AvgH", "AvgD", "AvgA"),
            ("PSH", "PSD", "PSA"),
        ):
            values = tuple(_optional_float(row.get(column)) for column in columns)
            if all(value is not None and value > 1.0 for value in values):
                return values  # type: ignore[return-value]
        return None, None, None

    @staticmethod
    def _normalized_csv(content: bytes) -> bytes:
        """Retain only detected fields that are useful to the V2 feature pipeline."""
        source = io.StringIO(content.decode("utf-8-sig", errors="replace"))
        reader = csv.DictReader(source)
        available = set(reader.fieldnames or [])
        if not {"Date", "HomeTeam", "AwayTeam"}.issubset(available):
            raise ValueError("download did not contain required EPL match columns")

        fields = [field for field in BASE_FIELDS + OPTIONAL_FIELDS if field in available]
        destination = io.StringIO(newline="")
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            writer.writerow({field: row.get(field, "") for field in fields})
        return destination.getvalue().encode("utf-8")


def reset_startup_refresh_state_for_tests() -> None:
    with _STARTUP_REFRESH_LOCK:
        _STARTUP_REFRESHED_DIRS.clear()
