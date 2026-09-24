from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from flask import Flask

from .data import MatchRepository
from .providers import OpenFootballFixtureProvider
from .routes import web
from .runtime import PredictionRuntime


def create_app(test_config: Optional[dict] = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        JSON_SORT_KEYS=False,
        HISTORY_SEASONS=int(os.environ.get("HISTORY_SEASONS", "6")),
        REQUEST_TIMEOUT_SECONDS=int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "12")),
        DATA_MAX_AGE_HOURS=float(os.environ.get("DATA_MAX_AGE_HOURS", "12")),
        REFRESH_CHECK_INTERVAL_SECONDS=float(
            os.environ.get("REFRESH_CHECK_INTERVAL_SECONDS", "300")
        ),
        AUTO_REFRESH=os.environ.get("AUTO_REFRESH", "true").lower() not in {"0", "false", "no"},
        STARTUP_REFRESH=True,
        DATA_DIR=os.environ.get("EPL_DATA_DIR"),
        REPORTS_DIR=os.environ.get("EPL_REPORTS_DIR"),
    )
    if test_config:
        app.config.update(test_config)

    repository = MatchRepository(
        history_seasons=int(app.config["HISTORY_SEASONS"]),
        timeout=int(app.config["REQUEST_TIMEOUT_SECONDS"]),
        data_dir=Path(app.config["DATA_DIR"]) if app.config.get("DATA_DIR") else None,
    )
    repository.load()
    if app.config["AUTO_REFRESH"] and app.config["STARTUP_REFRESH"]:
        repository.ensure_fresh_once(float(app.config["DATA_MAX_AGE_HOURS"]))

    reports_dir = (
        Path(app.config["REPORTS_DIR"])
        if app.config.get("REPORTS_DIR")
        else Path(__file__).resolve().parent.parent / "reports"
    )
    v2_report_path = reports_dir / "backtest_v2.json"
    v3_report_path = reports_dir / "backtest_v3.json"
    v2_report = {}
    v3_report = {}
    try:
        v2_report = json.loads(v2_report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    try:
        v3_report = json.loads(v3_report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    temperature = float(v2_report.get("live_calibration_temperature", 1.0))
    v3_config = {
        "v2_weight": v3_report.get("ensemble", {}).get("v2_weight", 1.0),
        "ml_temperature": v3_report.get("calibration", {}).get("ml_temperature", 1.0),
        "feature_importance": v3_report.get("feature_importance", []),
        "max_iter": 80,
    }
    app.extensions["prediction_runtime"] = PredictionRuntime(
        repository=repository,
        calibration_temperature=temperature,
        max_age_hours=float(app.config["DATA_MAX_AGE_HOURS"]),
        check_interval_seconds=float(app.config["REFRESH_CHECK_INTERVAL_SECONDS"]),
        auto_refresh=bool(app.config["AUTO_REFRESH"]),
        v3_config=v3_config,
    )
    app.extensions["backtest_report"] = v2_report
    app.extensions["backtest_v2_report"] = v2_report
    app.extensions["backtest_v3_report"] = v3_report
    app.extensions["reports_dir"] = reports_dir
    app.extensions["fixture_provider"] = OpenFootballFixtureProvider(
        timeout=int(app.config["REQUEST_TIMEOUT_SECONDS"])
    )
    app.register_blueprint(web)
    return app
