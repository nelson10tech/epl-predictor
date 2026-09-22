from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import requests


FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass(frozen=True)
class Match:
    played_on: date
    season: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int

    @property
    def result(self) -> str:
        if self.home_goals > self.away_goals:
            return "H"
        if self.home_goals < self.away_goals:
            return "A"
        return "D"


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


class MatchRepository:
    def __init__(self, history_seasons: int = 6, timeout: int = 20) -> None:
        configured_dir = os.environ.get("EPL_DATA_DIR")
        self.data_dir = Path(configured_dir) if configured_dir else DEFAULT_DATA_DIR
        self.history_seasons = history_seasons
        self.timeout = timeout
        self.matches: list[Match] = []
        self.loaded_files: list[Path] = []
        self.last_updated: Optional[datetime] = None

    def load(self) -> list[Match]:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        matches: list[Match] = []
        loaded_files: list[Path] = []

        for path in sorted(self.data_dir.glob("*_E0.csv")):
            season = path.name.split("_", 1)[0]
            parsed = list(self._read_file(path, season))
            if parsed:
                matches.extend(parsed)
                loaded_files.append(path)

        if not matches:
            raise RuntimeError(
                "No EPL match data found. Run `python manage.py refresh` to download it."
            )

        self.matches = sorted(matches, key=lambda item: item.played_on)
        self.loaded_files = loaded_files
        self.last_updated = datetime.fromtimestamp(
            max(path.stat().st_mtime for path in loaded_files), tz=timezone.utc
        )
        return self.matches

    def refresh(self) -> dict:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[str] = []
        failures: list[str] = []

        headers = {"User-Agent": "EPL-Predictor/1.0 (+https://github.com/nelson10tech/epl-predictor)"}
        with requests.Session() as session:
            for season in season_codes(self.history_seasons):
                url = FOOTBALL_DATA_URL.format(season=season)
                try:
                    response = session.get(url, headers=headers, timeout=self.timeout)
                    response.raise_for_status()
                    content = response.content
                    header = content[:300].decode("utf-8-sig", errors="replace")
                    if "HomeTeam" not in header or "AwayTeam" not in header:
                        raise ValueError("download did not contain EPL match columns")

                    destination = self.data_dir / f"{season}_E0.csv"
                    temporary = destination.with_suffix(".csv.tmp")
                    temporary.write_bytes(self._minimal_csv(content))
                    temporary.replace(destination)
                    downloaded.append(season)
                except (requests.RequestException, OSError, ValueError) as exc:
                    failures.append(f"{season}: {exc}")

        if not downloaded:
            raise RuntimeError("Unable to download any EPL seasons from Football-Data.co.uk")

        self.load()
        return {
            "downloaded_seasons": downloaded,
            "failed_seasons": failures,
            "matches": len(self.matches),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }

    def active_teams(self) -> list[str]:
        latest_season = max(match.season for match in self.matches)
        names = {
            team
            for match in self.matches
            if match.season == latest_season
            for team in (match.home_team, match.away_team)
        }
        return sorted(names)

    def latest_fixtures(self, limit: int = 30) -> list[Match]:
        latest_season = max(match.season for match in self.matches)
        latest = [match for match in self.matches if match.season == latest_season]
        return sorted(latest, key=lambda item: item.played_on, reverse=True)[:limit]

    @staticmethod
    def _read_file(path: Path, season: str) -> Iterable[Match]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if not row.get("Date") or not row.get("HomeTeam") or not row.get("AwayTeam"):
                    continue
                if row.get("FTHG", "").strip() == "" or row.get("FTAG", "").strip() == "":
                    continue
                try:
                    yield Match(
                        played_on=_parse_date(row["Date"]),
                        season=season,
                        home_team=row["HomeTeam"].strip(),
                        away_team=row["AwayTeam"].strip(),
                        home_goals=int(float(row["FTHG"])),
                        away_goals=int(float(row["FTAG"])),
                    )
                except (TypeError, ValueError):
                    continue

    @staticmethod
    def _minimal_csv(content: bytes) -> bytes:
        """Keep only the match fields used by V1; discard unrelated market columns."""
        fields = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
        source = io.StringIO(content.decode("utf-8-sig", errors="replace"))
        destination = io.StringIO(newline="")
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in csv.DictReader(source):
            writer.writerow({field: row.get(field, "") for field in fields})
        return destination.getvalue().encode("utf-8")
