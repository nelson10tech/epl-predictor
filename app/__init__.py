from __future__ import annotations

import os
from typing import Optional

from flask import Flask

from .data import MatchRepository
from .model import EloPoissonPredictor
from .routes import web


def create_app(test_config: Optional[dict] = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        JSON_SORT_KEYS=False,
        HISTORY_SEASONS=int(os.environ.get("HISTORY_SEASONS", "6")),
        REQUEST_TIMEOUT_SECONDS=int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "20")),
    )
    if test_config:
        app.config.update(test_config)

    repository = MatchRepository(
        history_seasons=int(app.config["HISTORY_SEASONS"]),
        timeout=int(app.config["REQUEST_TIMEOUT_SECONDS"]),
    )
    repository.load()

    app.extensions["match_repository"] = repository
    app.extensions["predictor"] = EloPoissonPredictor(repository.matches)
    app.register_blueprint(web)
    return app
