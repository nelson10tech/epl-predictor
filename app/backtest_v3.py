from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.inspection import permutation_importance

from .backtest import (
    PROBABILITY_KEYS,
    _walk_season,
    calculate_metrics,
    calibration_bins,
)
from .data import Match
from .feature_store import ML_FEATURE_NAMES, ChronologicalFeatureStore, FeatureRow
from .ml import (
    RESULT_LABEL,
    blend_probabilities,
    classifier_probabilities,
    fit_classifier,
)
from .model import TemperatureCalibrator


def _record_key(record: dict) -> tuple:
    return (record["match_date"], record["home_team"], record["away_team"])


def _walk_ml(
    rows: list[FeatureRow],
    target_seasons: list[str],
    max_iter: int,
) -> list[dict]:
    targets = [row for row in rows if row.match.season in target_seasons]
    grouped: defaultdict[date, list[FeatureRow]] = defaultdict(list)
    for row in targets:
        grouped[row.match.played_on].append(row)
    if not grouped:
        return []
    history = [row for row in rows if row.match.played_on < min(grouped)]
    records: list[dict] = []
    for match_date in sorted(grouped):
        classifier = fit_classifier(history, max_iter=max_iter)
        training_through = max(row.match.played_on for row in history)
        for row in grouped[match_date]:
            values = row.snapshot.values
            records.append({
                "season": row.match.season,
                "match_date": match_date,
                "training_through": training_through,
                "home_team": row.match.home_team,
                "away_team": row.match.away_team,
                "result": row.match.result,
                "ml_raw_probabilities": classifier_probabilities(
                    classifier, row.vector()
                ),
                "early_season": bool(values["early_season"]),
                "promoted_team_proxy": bool(
                    min(values["home_sample_count"], values["away_sample_count"]) < 38
                ),
            })
        history.extend(grouped[match_date])
    return records


def _apply_temperature(records: list[dict], source: str, target: str, temperature: float) -> None:
    calibrator = TemperatureCalibrator(temperature)
    for record in records:
        record[target] = calibrator.apply(record[source])


def _metrics_for_probability_pairs(pairs: list[tuple[dict, str]]) -> dict:
    records = [
        {"result": result, "probabilities": probabilities}
        for probabilities, result in pairs
    ]
    return calculate_metrics(records, "probabilities")


def _choose_ml_temperature(validation: list[dict]) -> tuple[float, dict]:
    samples = [
        (record["ml_raw_probabilities"], record["result"])
        for record in validation
    ]
    fitted = TemperatureCalibrator.fit(samples)
    candidates = {}
    for name, temperature in (("none", 1.0), ("temperature", fitted.temperature)):
        pairs = [
            (TemperatureCalibrator(temperature).apply(probabilities), result)
            for probabilities, result in samples
        ]
        candidates[name] = {
            "temperature": temperature,
            **_metrics_for_probability_pairs(pairs),
        }
    plain = candidates["none"]
    scaled = candidates["temperature"]
    chosen = fitted.temperature if (
        scaled["log_loss"] < plain["log_loss"]
        and scaled["brier_score"] <= plain["brier_score"] + 0.0005
    ) else 1.0
    return chosen, {
        "selected": "temperature" if chosen != 1.0 else "none",
        "ml_temperature": chosen,
        "candidates": candidates,
        "isotonic": {
            "selected": False,
            "reason": "Validation has fewer than 1,000 matches; isotonic was skipped to limit overfitting.",
        },
    }


def select_ensemble_weight(validation: list[dict]) -> tuple[float, list[dict]]:
    """Select only from the protected validation period, never final test rows."""
    candidates = []
    for step in range(11):
        weight = step / 10.0
        records = []
        for record in validation:
            records.append({
                "result": record["result"],
                "probabilities": blend_probabilities(
                    record["v2_probabilities"],
                    record["ml_probabilities"],
                    weight,
                ),
            })
        metrics = calculate_metrics(records, "probabilities")
        candidates.append({"v2_weight": weight, **metrics})
    best = min(candidates, key=lambda item: (item["log_loss"], item["brier_score"]))
    return float(best["v2_weight"]), candidates


def expected_calibration_error(records: list[dict], probability_field: str) -> Optional[float]:
    if not records:
        return None
    errors = []
    for outcome, result_code in zip(PROBABILITY_KEYS, ("H", "D", "A")):
        for lower_step in range(10):
            lower = lower_step / 10.0
            upper = (lower_step + 1) / 10.0
            selected = [
                record for record in records
                if lower <= record[probability_field][outcome]
                < upper + (1e-12 if upper == 1.0 else 0.0)
            ]
            if not selected:
                continue
            predicted = sum(item[probability_field][outcome] for item in selected) / len(selected)
            observed = sum(item["result"] == result_code for item in selected) / len(selected)
            errors.append((len(selected), abs(predicted - observed)))
    denominator = sum(count for count, _ in errors)
    return sum(count * error for count, error in errors) / denominator if denominator else None


def _attach_baselines(
    matches: list[Match],
    records: list[dict],
    seasons: list[str],
    temperatures: dict[str, float],
) -> None:
    baseline_by_key = {}
    for season in seasons:
        for record in _walk_season(
            matches,
            season,
            temperature=float(temperatures.get(season, 1.0)),
            include_v1=True,
        ):
            baseline_by_key[_record_key(record)] = record
    for record in records:
        baseline = baseline_by_key[_record_key(record)]
        for field in (
            "v1_probabilities",
            "v2_probabilities",
            "bookmaker_probabilities",
        ):
            record[field] = baseline.get(field)


def _model_metrics(records: list[dict], field: str, name: str) -> dict:
    result = calculate_metrics(records, field, optional=field == "bookmaker_probabilities")
    result["calibration_error"] = expected_calibration_error(
        [record for record in records if record.get(field)], field
    )
    return {"name": name, **result}


def _robustness(records: list[dict]) -> dict:
    groups = {
        "strong_favourites": [
            row for row in records if max(row["v2_probabilities"].values()) >= 0.60
        ],
        "balanced_matches": [
            row for row in records if max(row["v2_probabilities"].values()) <= 0.45
        ],
        "home_favourites": [
            row for row in records
            if row["v2_probabilities"]["home_win"]
            == max(row["v2_probabilities"].values())
        ],
        "away_favourites": [
            row for row in records
            if row["v2_probabilities"]["away_win"]
            == max(row["v2_probabilities"].values())
        ],
        "promoted_team_proxy": [row for row in records if row["promoted_team_proxy"]],
        "early_season": [row for row in records if row["early_season"]],
    }
    return {
        key: {
            "small_sample": len(selected) < 50,
            "v2": calculate_metrics(selected, "v2_probabilities"),
            "v3_ensemble": calculate_metrics(selected, "ensemble_probabilities"),
        }
        for key, selected in groups.items()
    }


def _feature_importance(rows: list[FeatureRow], max_iter: int) -> list[dict]:
    split = max(100, int(len(rows) * 0.8))
    validation = rows[split:][-200:]
    if len(validation) < 100:
        return []
    classifier = fit_classifier(rows[:split], max_iter=min(max_iter, 50))
    importance = permutation_importance(
        classifier,
        np.asarray([row.vector() for row in validation], dtype=float),
        np.asarray([RESULT_LABEL[row.result] for row in validation]),
        scoring="neg_log_loss",
        n_repeats=1,
        random_state=42,
    )
    ranked = sorted(
        zip(ML_FEATURE_NAMES, importance.importances_mean),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"feature": name, "importance": float(value), "method": "permutation_log_loss"}
        for name, value in ranked[:10]
    ]


def run_v3_backtest(
    matches: list[Match],
    v2_report: Optional[dict] = None,
    validation_season: str = "2324",
    evaluation_seasons: Optional[list[str]] = None,
    max_iter: int = 55,
) -> dict:
    ordered = sorted(matches, key=lambda item: (item.played_on, item.home_team))
    available = sorted({match.season for match in ordered})
    evaluation_seasons = evaluation_seasons or [
        season for season in ("2425", "2526", "2627") if season in available
    ]
    if validation_season not in available:
        raise ValueError("V3 requires a validation season before the final evaluation seasons")
    if not evaluation_seasons:
        raise ValueError("V3 found no final evaluation seasons")

    store = ChronologicalFeatureStore(ordered)
    validation = _walk_ml(store.rows, [validation_season], max_iter=max_iter)
    evaluation = _walk_ml(store.rows, evaluation_seasons, max_iter=max_iter)

    v2_report = v2_report or {}
    v2_temperatures = dict(v2_report.get("temperature_by_evaluation_season", {}))
    calibration_season = "2223" if "2223" in available else available[0]
    calibration_records = _walk_season(
        ordered, calibration_season, temperature=1.0, include_v1=False
    )
    prior_v2_calibrator = TemperatureCalibrator.fit([
        (row["v2_raw_probabilities"], row["result"])
        for row in calibration_records
    ])
    v2_temperatures[validation_season] = prior_v2_calibrator.temperature
    _attach_baselines(
        ordered,
        validation + evaluation,
        [validation_season] + evaluation_seasons,
        v2_temperatures,
    )

    ml_temperature, calibration_report = _choose_ml_temperature(validation)
    _apply_temperature(
        validation, "ml_raw_probabilities", "ml_probabilities", ml_temperature
    )
    _apply_temperature(
        evaluation, "ml_raw_probabilities", "ml_probabilities", ml_temperature
    )
    v2_weight, weight_candidates = select_ensemble_weight(validation)
    for record in validation + evaluation:
        record["ensemble_probabilities"] = blend_probabilities(
            record["v2_probabilities"], record["ml_probabilities"], v2_weight
        )

    models = {
        "v1": _model_metrics(evaluation, "v1_probabilities", "Elo + Poisson V1"),
        "v2": _model_metrics(evaluation, "v2_probabilities", "Enhanced Dixon-Coles V2"),
        "ml": _model_metrics(evaluation, "ml_probabilities", "V3 HistGradientBoosting ML"),
        "v3_ensemble": _model_metrics(
            evaluation, "ensemble_probabilities", "V3 Ensemble"
        ),
        "bookmaker": _model_metrics(
            evaluation,
            "bookmaker_probabilities",
            "Normalized bookmaker implied probabilities (benchmark only)",
        ),
    }
    by_season = []
    for season in evaluation_seasons:
        rows = [record for record in evaluation if record["season"] == season]
        by_season.append({
            "season": season,
            "sample_size": len(rows),
            "v1": _model_metrics(rows, "v1_probabilities", "Elo + Poisson V1"),
            "v2": _model_metrics(rows, "v2_probabilities", "Enhanced Dixon-Coles V2"),
            "ml": _model_metrics(rows, "ml_probabilities", "V3 HistGradientBoosting ML"),
            "v3_ensemble": _model_metrics(rows, "ensemble_probabilities", "V3 Ensemble"),
            "bookmaker": _model_metrics(
                rows, "bookmaker_probabilities", "Bookmaker benchmark"
            ),
        })

    v2 = models["v2"]
    ensemble = models["v3_ensemble"]
    calibration_regression = (
        ensemble["calibration_error"] is not None
        and v2["calibration_error"] is not None
        and ensemble["calibration_error"] > v2["calibration_error"] + 0.01
    )
    improvement_seasons = sum(
        season["v3_ensemble"]["log_loss"] < season["v2"]["log_loss"]
        and season["v3_ensemble"]["brier_score"] < season["v2"]["brier_score"]
        for season in by_season
    )
    qualifies = (
        ensemble["log_loss"] < v2["log_loss"]
        and ensemble["brier_score"] < v2["brier_score"]
        and not calibration_regression
    )
    leakage_ok = all(
        record["training_through"] < record["match_date"]
        for record in validation + evaluation
    )
    report = {
        "app_version": "3.0",
        "model_version": "3.0",
        "generated_from_data_through": ordered[-1].played_on.isoformat(),
        "training_seasons": [
            season for season in available if season < validation_season
        ],
        "validation_season": validation_season,
        "evaluation_seasons": evaluation_seasons,
        "sample_size": len(evaluation),
        "chronological_walk_forward": True,
        "same_day_results_withheld_until_day_complete": True,
        "leakage_check_passed": leakage_ok,
        "models": models,
        "by_season": by_season,
        "calibration": {
            **calibration_report,
            "evaluation_bins": calibration_bins(evaluation, "ensemble_probabilities"),
        },
        "ensemble": {
            "v2_weight": v2_weight,
            "ml_weight": 1.0 - v2_weight,
            "selection_season": validation_season,
            "selection_metric": "log_loss",
            "candidate_weights": weight_candidates,
            "final_test_used_for_selection": False,
        },
        "v3_vs_v2": {
            "log_loss_difference": ensemble["log_loss"] - v2["log_loss"],
            "brier_difference": ensemble["brier_score"] - v2["brier_score"],
            "rps_difference": (
                ensemble["ranked_probability_score"]
                - v2["ranked_probability_score"]
            ),
        },
        "promotion": {
            "eligible": qualifies,
            "live_model": "v2",
            "recommended_model": "v3" if qualifies else "v2",
            "serious_calibration_regression": calibration_regression,
            "evaluation_seasons_improved_on_both_primary_metrics": improvement_seasons,
            "rule": (
                "V3 must beat V2 on aggregate Log Loss and Brier without an obvious "
                "calibration regression. Initial deployment remains shadow-only."
            ),
        },
        "robustness": _robustness(evaluation),
        "feature_importance": _feature_importance(store.rows, max_iter),
        "data_provenance": store.metadata(),
    }
    return report


def run_v3_smoke_backtest(matches: list[Match], max_matches: int = 20) -> dict:
    ordered = sorted(matches, key=lambda item: (item.played_on, item.home_team))
    target_season = sorted({match.season for match in ordered})[-1]
    target_rows = [match for match in ordered if match.season == target_season][:max_matches]
    first_date = target_rows[0].played_on
    selected = [match for match in ordered if match.played_on < first_date][-500:] + target_rows
    store = ChronologicalFeatureStore(selected)
    records = _walk_ml(store.rows, [target_season], max_iter=12)
    return {
        "sample_size": len(records),
        "probabilities_normalized": all(
            abs(sum(row["ml_raw_probabilities"].values()) - 1.0) < 1e-9
            for row in records
        ),
        "leakage_check_passed": all(
            row["training_through"] < row["match_date"] for row in records
        ),
    }


def write_v3_backtest_report(report: dict, report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "backtest_v3.json"
    csv_path = report_dir / "backtest_v3.csv"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "scope", "model", "sample_size", "accuracy", "log_loss",
            "brier_score", "ranked_probability_score", "calibration_error",
        ])
        for key, metrics in report["models"].items():
            writer.writerow([
                "all", key, metrics["sample_size"], metrics["accuracy"],
                metrics["log_loss"], metrics["brier_score"],
                metrics["ranked_probability_score"], metrics["calibration_error"],
            ])
        for season in report["by_season"]:
            for key in ("v1", "v2", "ml", "v3_ensemble", "bookmaker"):
                metrics = season[key]
                writer.writerow([
                    season["season"], key, metrics["sample_size"], metrics["accuracy"],
                    metrics["log_loss"], metrics["brier_score"],
                    metrics["ranked_probability_score"], metrics["calibration_error"],
                ])
    return json_path, csv_path
