from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from flask import Flask

from .data import MatchRepository
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
    report_path = reports_dir / "backtest_v2.json"
    report = {}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    temperature = float(report.get("live_calibration_temperature", 1.0))
    app.extensions["prediction_runtime"] = PredictionRuntime(
        repository=repository,
        calibration_temperature=temperature,
        max_age_hours=float(app.config["DATA_MAX_AGE_HOURS"]),
        check_interval_seconds=float(app.config["REFRESH_CHECK_INTERVAL_SECONDS"]),
        auto_refresh=bool(app.config["AUTO_REFRESH"]),
    )
    app.extensions["backtest_report"] = report
    app.extensions["reports_dir"] = reports_dir
    app.register_blueprint(web)
    return app
