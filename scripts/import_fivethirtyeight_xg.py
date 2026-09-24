#!/usr/bin/env python3
"""Import the final public FiveThirtyEight EPL xG archive into a small cache."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import requests


ARCHIVE_URL = (
    "https://web.archive.org/web/20230625093532if_/"
    "https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv"
)


def import_xg(destination: Path, source: str = ARCHIVE_URL) -> int:
    response = requests.get(source, timeout=90, headers={"User-Agent": "EPL-Predictor/3.0"})
    response.raise_for_status()
    reader = csv.DictReader(io.StringIO(response.text))
    rows = [
        {
            "date": row["date"],
            "home_team": row["team1"],
            "away_team": row["team2"],
            "home_xg": row["xg1"],
            "away_xg": row["xg2"],
            "source": "FiveThirtyEight Soccer SPI",
        }
        for row in reader
        if row.get("league_id") == "2411"
        and row.get("season") in {"2021", "2022"}
        and row.get("xg1")
        and row.get("xg2")
    ]
    if len(rows) != 760:
        raise RuntimeError(f"expected 760 complete EPL xG rows, received {len(rows)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/advanced/fivethirtyeight_epl_xg_2021_2023.csv"),
    )
    args = parser.parse_args()
    print(f"Imported {import_xg(args.output)} genuine xG rows into {args.output}")


if __name__ == "__main__":
    main()
