from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .backtest import calculate_metrics
from .data import Fixture, Match, MatchRepository
from .model import DixonColesPredictor


PREDICTION_SCHEMA_VERSION = 1
APP_VERSION = "2.1"
MODEL_VERSION = "2.0"


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _london_timezone():
    try:
        return ZoneInfo("Europe/London")
    except ZoneInfoNotFoundError:
        return timezone.utc


def _kickoff_utc(fixture: Fixture) -> Optional[datetime]:
    if not fixture.kickoff:
        return None
    try:
        hour, minute = (int(part) for part in fixture.kickoff.split(":", 1))
        local = datetime.combine(
            fixture.played_on,
            time(hour=hour, minute=minute),
            tzinfo=_london_timezone(),
        )
        return local.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _fixture_key(fixture: Fixture, model_version: str = MODEL_VERSION) -> str:
    return "|".join((
        fixture.played_on.isoformat(),
        fixture.kickoff or "",
        fixture.home_team,
        fixture.away_team,
        model_version,
    ))


def load_prediction_records(reports_dir: Path) -> list[dict]:
    records: list[dict] = []
    prediction_dir = reports_dir / "live_predictions"
    for path in sorted(prediction_dir.glob("[0-9][0-9][0-9][0-9]-*.json")):
        payload = _read_json(path, {})
        if isinstance(payload, dict) and isinstance(payload.get("predictions"), list):
            records.extend(item for item in payload["predictions"] if isinstance(item, dict))
    return records


def snapshot_predictions(
    repository: MatchRepository,
    predictor: DixonColesPredictor,
    reports_dir: Path,
    now: Optional[datetime] = None,
    dedupe_window_hours: float = 6.0,
) -> dict:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    london_today = now.astimezone(_london_timezone()).date()
    existing = load_prediction_records(reports_dir)
    dedupe_window = timedelta(hours=dedupe_window_hours)
    created: list[dict] = []
    skipped_duplicate = 0
    skipped_not_pre_match = 0
    errors: list[str] = []

    for fixture in sorted(
        repository.fixtures,
        key=lambda item: (item.played_on, item.kickoff or "", item.home_team),
    ):
        kickoff_utc = _kickoff_utc(fixture)
        if fixture.played_on < london_today:
            skipped_not_pre_match += 1
            continue
        if fixture.played_on == london_today:
            if kickoff_utc is None or kickoff_utc <= now:
                skipped_not_pre_match += 1
                continue

        key = _fixture_key(fixture)
        duplicate = False
        for record in existing + created:
            if record.get("fixture_key") != key:
                continue
            recorded_at = _parse_datetime(record.get("prediction_created_at"))
            if recorded_at and timedelta(0) <= now - recorded_at < dedupe_window:
                duplicate = True
                break
        if duplicate:
            skipped_duplicate += 1
            continue

        try:
            prediction = predictor.predict(
                fixture.home_team,
                fixture.away_team,
                as_of=fixture.played_on,
            )
        except ValueError as exc:
            errors.append(f"{fixture.home_team} vs {fixture.away_team}: {exc}")
            continue

        created_at = now.isoformat()
        prediction_id = hashlib.sha256(
            f"{key}|{created_at}".encode("utf-8")
        ).hexdigest()[:24]
        probabilities = prediction["probabilities"]
        expected = prediction["expected_goals"]
        score = prediction["most_likely_score"]
        created.append({
            "prediction_id": prediction_id,
            "fixture_key": key,
            "prediction_created_at": created_at,
            "fixture_date": fixture.played_on.isoformat(),
            "kickoff": fixture.kickoff,
            "kickoff_utc": kickoff_utc.isoformat() if kickoff_utc else None,
            "home_team": fixture.home_team,
            "away_team": fixture.away_team,
            "home_win_probability": probabilities["home_win"],
            "draw_probability": probabilities["draw"],
            "away_win_probability": probabilities["away_win"],
            "expected_home_goals": expected["home"],
            "expected_away_goals": expected["away"],
            "most_likely_score": f"{score['home']}-{score['away']}",
            "model": prediction["model"],
            "app_version": APP_VERSION,
            "model_version": prediction["version"],
            "data_through": prediction["data_through"],
            "xg_enabled": prediction["xg_enabled"],
        })

    output_path = reports_dir / "live_predictions" / f"{now.date().isoformat()}.json"
    if created:
        payload = _read_json(
            output_path,
            {"schema_version": PREDICTION_SCHEMA_VERSION, "predictions": []},
        )
        original = payload.get("predictions", []) if isinstance(payload, dict) else []
        payload = {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "predictions": original + created,
        }
        _write_json(output_path, payload)

    return {
        "created": len(created),
        "skipped_duplicate": skipped_duplicate,
        "skipped_not_pre_match": skipped_not_pre_match,
        "errors": errors,
        "output": str(output_path) if created else None,
    }


def _season_code(played_on: date) -> str:
    start = played_on.year if played_on.month >= 7 else played_on.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def _metric_records(predictions: list[dict], results: dict[str, dict]) -> list[dict]:
    records = []
    for prediction in predictions:
        result = results.get(prediction.get("prediction_id"))
        if not result or not _was_created_before_kickoff(prediction):
            continue
        records.append({
            "fixture_date": prediction["fixture_date"],
            "prediction_created_at": prediction["prediction_created_at"],
            "result": result["actual_result"],
            "probabilities": {
                "home_win": prediction["home_win_probability"],
                "draw": prediction["draw_probability"],
                "away_win": prediction["away_win_probability"],
            },
        })
    return sorted(
        records,
        key=lambda item: (item["fixture_date"], item["prediction_created_at"]),
    )


def _was_created_before_kickoff(prediction: dict) -> bool:
    created_at = _parse_datetime(prediction.get("prediction_created_at"))
    if created_at is None:
        return False
    kickoff_at = _parse_datetime(prediction.get("kickoff_utc"))
    if kickoff_at is not None:
        return created_at < kickoff_at
    try:
        fixture_date = date.fromisoformat(prediction["fixture_date"])
    except (KeyError, TypeError, ValueError):
        return False
    return created_at.astimezone(_london_timezone()).date() < fixture_date


def _empty_metrics() -> dict:
    return {
        "sample_size": 0,
        "accuracy": None,
        "log_loss": None,
        "brier_score": None,
        "ranked_probability_score": None,
    }


def _metrics(records: list[dict]) -> dict:
    return calculate_metrics(records, "probabilities") if records else _empty_metrics()


def _performance_status(sample_size: int, log_loss, historical_report: dict) -> dict:
    if sample_size < 50:
        return {
            "level": "insufficient",
            "message": "Small sample — live performance is not yet statistically reliable.",
        }
    baseline = historical_report.get("models", {}).get("v2", {}).get("log_loss")
    if baseline is not None and log_loss is not None and log_loss > baseline + 0.05:
        return {
            "level": "watch",
            "message": "Live Log Loss has worsened versus the historical baseline.",
        }
    return {
        "level": "normal",
        "message": "No material live deterioration is visible at the current sample size.",
    }


def score_predictions(
    repository: MatchRepository,
    reports_dir: Path,
    historical_report: Optional[dict] = None,
    now: Optional[datetime] = None,
) -> dict:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    historical_report = historical_report or {}
    predictions = load_prediction_records(reports_dir)
    results_path = reports_dir / "live_results.json"
    result_payload = _read_json(
        results_path,
        {"schema_version": PREDICTION_SCHEMA_VERSION, "results": []},
    )
    original_results = (
        result_payload.get("results", []) if isinstance(result_payload, dict) else []
    )
    results_by_id = {
        item.get("prediction_id"): item
        for item in original_results
        if isinstance(item, dict) and item.get("prediction_id")
    }
    completed: dict[tuple[str, str, str], Match] = {
        (match.played_on.isoformat(), match.home_team, match.away_team): match
        for match in repository.matches
    }
    new_results: list[dict] = []
    for prediction in predictions:
        prediction_id = prediction.get("prediction_id")
        if (
            not prediction_id
            or prediction_id in results_by_id
            or not _was_created_before_kickoff(prediction)
        ):
            continue
        match = completed.get((
            prediction.get("fixture_date"),
            prediction.get("home_team"),
            prediction.get("away_team"),
        ))
        if not match:
            continue
        result = {
            "prediction_id": prediction_id,
            "settled_at": now.isoformat(),
            "actual_result": match.result,
            "actual_home_goals": match.home_goals,
            "actual_away_goals": match.away_goals,
        }
        new_results.append(result)
        results_by_id[prediction_id] = result

    if new_results:
        _write_json(results_path, {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "results": original_results + new_results,
        })

    metric_records = _metric_records(predictions, results_by_id)
    overall = _metrics(metric_records)
    latest_season = max(
        (_season_code(date.fromisoformat(item["fixture_date"])) for item in metric_records),
        default=None,
    )
    season_records = [
        item for item in metric_records
        if latest_season and _season_code(date.fromisoformat(item["fixture_date"])) == latest_season
    ]
    report = {
        "performance_type": "live_prospective",
        "historical_backtest_included": False,
        "predictions_recorded": len(predictions),
        "settled_predictions": len(metric_records),
        "overall": overall,
        "rolling": {
            "last_20": _metrics(metric_records[-20:]),
            "last_50": _metrics(metric_records[-50:]),
        },
        "season_to_date": {
            "season": latest_season,
            **_metrics(season_records),
        },
        "status": _performance_status(
            overall["sample_size"], overall["log_loss"], historical_report
        ),
    }
    report_path = reports_dir / "live_performance.json"
    previous = _read_json(report_path, {})
    comparable_previous = dict(previous) if isinstance(previous, dict) else {}
    comparable_previous.pop("updated_at", None)
    if comparable_previous != report:
        report["updated_at"] = now.isoformat()
        _write_json(report_path, report)
    elif isinstance(previous, dict) and previous.get("updated_at"):
        report["updated_at"] = previous["updated_at"]

    return {
        "newly_settled": len(new_results),
        "report": report,
        "results_path": str(results_path),
        "report_path": str(report_path),
    }


def load_live_performance(reports_dir: Path) -> dict:
    path = reports_dir / "live_performance.json"
    report = _read_json(path, {})
    if isinstance(report, dict) and report:
        return report
    empty = _empty_metrics()
    return {
        "performance_type": "live_prospective",
        "historical_backtest_included": False,
        "predictions_recorded": 0,
        "settled_predictions": 0,
        "overall": empty,
        "rolling": {"last_20": empty, "last_50": empty},
        "season_to_date": {"season": None, **empty},
        "status": {
            "level": "insufficient",
            "message": "Small sample — live performance is not yet statistically reliable.",
        },
    }
