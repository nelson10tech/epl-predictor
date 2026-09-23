from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

from .data import Match
from .model import (
    DixonColesPredictor,
    EloPoissonPredictor,
    TemperatureCalibrator,
    bookmaker_probabilities,
    dixon_coles_score_matrix,
    outcome_probabilities,
)


RESULT_INDEX = {"H": 0, "D": 1, "A": 2}
PROBABILITY_KEYS = ("home_win", "draw", "away_win")


def run_walk_forward_backtest(
    matches: list[Match],
    evaluation_seasons: Optional[list[str]] = None,
    calibration_season: str = "2324",
) -> dict:
    matches = sorted(matches, key=lambda item: (item.played_on, item.home_team))
    available_seasons = sorted({match.season for match in matches})
    evaluation_seasons = evaluation_seasons or [
        season for season in ("2425", "2526", "2627") if season in available_seasons
    ]
    if calibration_season not in available_seasons:
        calibration_season = available_seasons[max(1, len(available_seasons) - 4)]

    calibration_raw = _walk_season(matches, calibration_season, temperature=1.0, include_v1=False)
    calibration_pool = [
        (record["v2_raw_probabilities"], record["result"])
        for record in calibration_raw
    ]

    all_records: list[dict] = []
    season_reports: list[dict] = []
    temperatures: dict[str, float] = {}
    for season in evaluation_seasons:
        calibrator = TemperatureCalibrator.fit(calibration_pool)
        temperatures[season] = calibrator.temperature
        records = _walk_season(
            matches,
            season,
            temperature=calibrator.temperature,
            include_v1=True,
        )
        all_records.extend(records)
        season_reports.append({
            "season": season,
            "sample_size": len(records),
            "calibration_temperature": calibrator.temperature,
            "v1": calculate_metrics(records, "v1_probabilities"),
            "v2": calculate_metrics(records, "v2_probabilities"),
            "bookmaker": calculate_metrics(records, "bookmaker_probabilities", optional=True),
        })
        calibration_pool.extend(
            (record["v2_raw_probabilities"], record["result"])
            for record in records
        )

    live_calibrator = TemperatureCalibrator.fit(calibration_pool)
    leakage_ok = all(
        record["training_through"] is None
        or record["training_through"] < record["match_date"]
        for record in all_records
    )
    v1_metrics = calculate_metrics(all_records, "v1_probabilities")
    v2_metrics = calculate_metrics(all_records, "v2_probabilities")
    bookmaker_metrics = calculate_metrics(
        all_records, "bookmaker_probabilities", optional=True
    )

    return {
        "model_version": "2.0",
        "generated_from_data_through": matches[-1].played_on.isoformat(),
        "training_start_season": available_seasons[0],
        "calibration_season": calibration_season,
        "evaluation_seasons": evaluation_seasons,
        "sample_size": len(all_records),
        "chronological_walk_forward": True,
        "same_day_results_withheld_until_day_complete": True,
        "leakage_check_passed": leakage_ok,
        "live_calibration_temperature": live_calibrator.temperature,
        "temperature_by_evaluation_season": temperatures,
        "models": {
            "v1": {"name": "Elo + Poisson V1", **v1_metrics},
            "v2": {"name": "Enhanced Dixon-Coles V2", **v2_metrics},
            "bookmaker": {
                "name": "Normalized bookmaker implied probabilities (benchmark only)",
                **bookmaker_metrics,
            },
        },
        "calibration": calibration_bins(all_records, "v2_probabilities"),
        "by_season": season_reports,
    }


def _walk_season(
    matches: list[Match],
    target_season: str,
    temperature: float,
    include_v1: bool,
) -> list[dict]:
    target = [match for match in matches if match.season == target_season]
    if not target:
        return []
    first_date = min(match.played_on for match in target)
    history = [match for match in matches if match.played_on < first_date]
    grouped: defaultdict[date, list[Match]] = defaultdict(list)
    for match in target:
        grouped[match.played_on].append(match)

    records: list[dict] = []
    calibrator = TemperatureCalibrator(temperature)
    for match_date in sorted(grouped):
        if not history:
            continue
        v1 = EloPoissonPredictor(history) if include_v1 else None
        v2 = DixonColesPredictor(history, calibration_temperature=1.0)
        training_through = max(match.played_on for match in history)
        for match in grouped[match_date]:
            snapshot = v2.features.snapshot(match.home_team, match.away_team, match_date)
            expected_home, expected_away = v2.expected_goals(snapshot)
            raw_matrix = dixon_coles_score_matrix(expected_home, expected_away, rho=v2.RHO)
            v2_raw = outcome_probabilities(raw_matrix)
            record = {
                "season": target_season,
                "match_date": match_date,
                "training_through": training_through,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "result": match.result,
                "v2_raw_probabilities": v2_raw,
                "v2_probabilities": calibrator.apply(v2_raw),
                "bookmaker_probabilities": bookmaker_probabilities(match),
            }
            if v1 is not None:
                record["v1_probabilities"] = v1.predict(
                    match.home_team, match.away_team, allow_unseen=True
                )["probabilities"]
            records.append(record)

        # Results from simultaneous/same-day matches become available only after all
        # predictions for that date have been recorded.
        history.extend(grouped[match_date])
    return records


def calculate_metrics(
    records: list[dict], probability_field: str, optional: bool = False
) -> dict:
    eligible = [record for record in records if record.get(probability_field)]
    if not eligible:
        return {
            "sample_size": 0,
            "accuracy": None,
            "log_loss": None,
            "brier_score": None,
            "ranked_probability_score": None,
        }

    correct = 0
    log_loss = 0.0
    brier = 0.0
    rps = 0.0
    for record in eligible:
        probabilities = record[probability_field]
        vector = [probabilities[key] for key in PROBABILITY_KEYS]
        actual_index = RESULT_INDEX[record["result"]]
        actual = [1.0 if index == actual_index else 0.0 for index in range(3)]
        correct += int(max(range(3), key=lambda index: vector[index]) == actual_index)
        log_loss -= math.log(max(vector[actual_index], 1e-12))
        brier += sum((vector[index] - actual[index]) ** 2 for index in range(3)) / 3.0
        rps += 0.5 * (
            (vector[0] - actual[0]) ** 2
            + (vector[0] + vector[1] - actual[0] - actual[1]) ** 2
        )

    count = len(eligible)
    return {
        "sample_size": count,
        "accuracy": correct / count,
        "log_loss": log_loss / count,
        "brier_score": brier / count,
        "ranked_probability_score": rps / count,
    }


def calibration_bins(records: list[dict], probability_field: str) -> dict:
    result = {}
    for outcome, result_code in zip(PROBABILITY_KEYS, ("H", "D", "A")):
        bins = []
        for lower_int in range(0, 10):
            lower = lower_int / 10.0
            upper = (lower_int + 1) / 10.0
            selected = [
                record for record in records
                if lower <= record[probability_field][outcome]
                < upper + (1e-12 if upper == 1.0 else 0.0)
            ]
            if not selected:
                continue
            bins.append({
                "range": f"{lower:.1f}-{upper:.1f}",
                "predicted_probability": sum(
                    record[probability_field][outcome] for record in selected
                ) / len(selected),
                "actual_frequency": sum(
                    1 for record in selected if record["result"] == result_code
                ) / len(selected),
                "sample_count": len(selected),
            })
        result[outcome] = bins
    return result


def write_backtest_report(report: dict, report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "backtest_v2.json"
    csv_path = report_dir / "backtest_v2.csv"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "scope", "model", "sample_size", "accuracy", "log_loss",
            "brier_score", "ranked_probability_score",
        ])
        for key in ("v1", "v2", "bookmaker"):
            metrics = report["models"][key]
            writer.writerow([
                "all", key, metrics["sample_size"], metrics["accuracy"],
                metrics["log_loss"], metrics["brier_score"],
                metrics["ranked_probability_score"],
            ])
        for season in report["by_season"]:
            for key in ("v1", "v2", "bookmaker"):
                metrics = season[key]
                writer.writerow([
                    season["season"], key, metrics["sample_size"], metrics["accuracy"],
                    metrics["log_loss"], metrics["brier_score"],
                    metrics["ranked_probability_score"],
                ])
    return json_path, csv_path
