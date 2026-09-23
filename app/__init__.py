from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional

from flask import Flask

from .data import MatchRepository
from .model import DixonColesPredictor, EloPoissonPredictor
from .routes import web


def create_app(test_config: Optional[dict] = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        JSON_SORT_KEYS=False,
        HISTORY_SEASONS=int(os.environ.get("HISTORY_SEASONS", "6")),
        REQUEST_TIMEOUT_SECONDS=int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "12")),
        DATA_MAX_AGE_HOURS=float(os.environ.get("DATA_MAX_AGE_HOURS", "12")),
        AUTO_REFRESH=os.environ.get("AUTO_REFRESH", "true").lower() not in {"0", "false", "no"},
        DATA_DIR=os.environ.get("EPL_DATA_DIR"),
    )
    if test_config:
        app.config.update(test_config)

    repository = MatchRepository(
        history_seasons=int(app.config["HISTORY_SEASONS"]),
        timeout=int(app.config["REQUEST_TIMEOUT_SECONDS"]),
        data_dir=Path(app.config["DATA_DIR"]) if app.config.get("DATA_DIR") else None,
    )
    repository.load()
    if app.config["AUTO_REFRESH"]:
        repository.ensure_fresh_once(float(app.config["DATA_MAX_AGE_HOURS"]))

    app.extensions["match_repository"] = repository
    report_path = Path(__file__).resolve().parent.parent / "reports" / "backtest_v2.json"
    report = {}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    temperature = float(report.get("live_calibration_temperature", 1.0))
    app.extensions["predictors"] = {
        "v1": EloPoissonPredictor(repository.matches),
        "v2": DixonColesPredictor(repository.matches, calibration_temperature=temperature),
    }
    app.extensions["backtest_report"] = report
    app.register_blueprint(web)
    return app
