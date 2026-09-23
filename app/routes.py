from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template, request

from .live import load_live_performance
from .model import DixonColesPredictor
from .runtime import APP_VERSION, MODEL_VERSION


web = Blueprint("web", __name__)


def _runtime():
    return current_app.extensions["prediction_runtime"]


def _state():
    return _runtime().snapshot()


@web.before_app_request
def check_data_freshness():
    if request.endpoint != "static":
        _runtime().maybe_refresh()


@web.get("/")
def index():
    repository = _state().repository
    teams = repository.active_teams()
    default_home = "Arsenal" if "Arsenal" in teams else teams[0]
    default_away = "Liverpool" if "Liverpool" in teams else teams[1]
    return render_template(
        "index.html",
        teams=teams,
        default_home=default_home,
        default_away=default_away,
        last_updated=repository.last_updated,
        freshness=_runtime().freshness_metadata(),
    )


@web.get("/fixtures")
def fixtures():
    repository = _state().repository
    return render_template(
        "fixtures.html",
        upcoming=repository.upcoming_fixtures(),
        results=repository.latest_results(),
        last_updated=repository.last_updated,
        freshness=_runtime().freshness_metadata(),
    )


@web.get("/model")
def model_info():
    state = _state()
    repository = state.repository
    predictor: DixonColesPredictor = state.predictors["v2"]
    live_report = load_live_performance(current_app.extensions["reports_dir"])
    return render_template(
        "model.html",
        report=current_app.extensions.get("backtest_report", {}),
        matches=len(repository.matches),
        active_season=repository.active_season,
        live_model="Enhanced Dixon-Coles V2",
        xg=predictor.xg_provider.metadata(),
        freshness=_runtime().freshness_metadata(),
        live_report=live_report,
        app_version=APP_VERSION,
        model_version=MODEL_VERSION,
    )


@web.get("/api/predict")
def predict():
    home_team = request.args.get("home", "").strip()
    away_team = request.args.get("away", "").strip()
    if not home_team or not away_team:
        return jsonify({"error": "home and away query parameters are required"}), 400
    model_key = request.args.get("model", "v2").lower()
    state = _state()
    if model_key not in state.predictors:
        return jsonify({"error": "model must be v1 or v2"}), 400
    try:
        prediction = state.predictors[model_key].predict(home_team, away_team)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    freshness = _runtime().freshness_metadata()
    prediction["app_version"] = APP_VERSION
    prediction["model_version"] = prediction["version"]
    prediction["data_stale"] = freshness["data_stale"]
    prediction["metadata"]["freshness"] = freshness
    prediction["metadata"]["live_model"] = model_key == "v2"
    return jsonify(prediction)


@web.get("/api/fixtures")
def api_fixtures():
    limit = min(max(request.args.get("limit", 20, type=int), 1), 100)
    repository = _state().repository
    matches = repository.latest_results(limit)
    upcoming = repository.upcoming_fixtures(limit)
    return jsonify(
        {
            "upcoming": [
                {
                    "date": fixture.played_on.isoformat(),
                    "kickoff": fixture.kickoff,
                    "home_team": fixture.home_team,
                    "away_team": fixture.away_team,
                }
                for fixture in upcoming
            ],
            "results": [
                {
                    "date": match.played_on.isoformat(),
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "home_goals": match.home_goals,
                    "away_goals": match.away_goals,
                    "result": match.result,
                }
                for match in matches
            ]
        }
    )


@web.get("/api/model")
def api_model():
    state = _state()
    repository = state.repository
    predictor: DixonColesPredictor = state.predictors["v2"]
    freshness = _runtime().freshness_metadata()
    return jsonify({
        "version": APP_VERSION,
        "app_version": APP_VERSION,
        "model_version": MODEL_VERSION,
        "live_model": "Enhanced Dixon-Coles V2",
        "matches": len(repository.matches),
        "active_season": repository.active_season,
        "xg": predictor.xg_provider.metadata(),
        "historical_backtest": current_app.extensions.get("backtest_report", {}),
        "live_performance": load_live_performance(current_app.extensions["reports_dir"]),
        "freshness": freshness,
        **freshness,
    })


@web.get("/health")
def health():
    repository = _state().repository
    return jsonify({
        "status": "ok",
        "app_version": APP_VERSION,
        "model_version": MODEL_VERSION,
        "matches": len(repository.matches),
        "active_season": repository.active_season,
        **_runtime().freshness_metadata(),
    })
