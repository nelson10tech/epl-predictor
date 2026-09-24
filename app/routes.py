from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template, request

from .live import load_live_performance
from .model import DixonColesPredictor
from .runtime import APP_VERSION, CHALLENGER_MODEL_VERSION, MODEL_VERSION


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
    provider = current_app.extensions["fixture_provider"]
    upcoming = provider.load_cache()
    if not upcoming:
        upcoming = repository.upcoming_fixtures()
    upcoming = upcoming[:30]
    return render_template(
        "fixtures.html",
        upcoming=upcoming,
        results=repository.latest_results(),
        last_updated=repository.last_updated,
        freshness=_runtime().freshness_metadata(),
        fixture_source=provider.metadata(),
    )


@web.get("/model")
def model_info():
    state = _state()
    repository = state.repository
    predictor: DixonColesPredictor = state.predictors["v2"]
    live_report = load_live_performance(current_app.extensions["reports_dir"])
    v3_report = current_app.extensions.get("backtest_v3_report", {})
    ml = state.predictors.get("ml")
    return render_template(
        "model.html",
        report=current_app.extensions.get("backtest_v2_report", {}),
        v3_report=v3_report,
        matches=len(repository.matches),
        active_season=repository.active_season,
        live_model="Enhanced Dixon-Coles V2",
        xg=predictor.xg_provider.metadata(),
        freshness=_runtime().freshness_metadata(),
        live_report=live_report,
        app_version=APP_VERSION,
        model_version=MODEL_VERSION,
        challenger_version=CHALLENGER_MODEL_VERSION,
        feature_importance=getattr(ml, "feature_importance", []),
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
        return jsonify({"error": "model must be v1, v2, ml or v3"}), 400
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
    prediction["metadata"]["shadow_model"] = model_key in {"ml", "v3"}
    return jsonify(prediction)


@web.get("/api/fixtures")
def api_fixtures():
    limit = min(max(request.args.get("limit", 20, type=int), 1), 100)
    repository = _state().repository
    matches = repository.latest_results(limit)
    provider = current_app.extensions["fixture_provider"]
    upcoming = provider.load_cache()[:limit] or repository.upcoming_fixtures(limit)
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
        "challenger_model_version": CHALLENGER_MODEL_VERSION,
        "live_model": "Enhanced Dixon-Coles V2",
        "models_available": sorted(state.predictors),
        "matches": len(repository.matches),
        "active_season": repository.active_season,
        "xg": predictor.xg_provider.metadata(),
        "historical_backtest": current_app.extensions.get("backtest_report", {}),
        "historical_backtest_v3": current_app.extensions.get("backtest_v3_report", {}),
        "live_performance": load_live_performance(current_app.extensions["reports_dir"]),
        "freshness": freshness,
        **freshness,
    })


@web.get("/api/data")
def api_data():
    state = _state()
    repository = state.repository
    fixture_provider = current_app.extensions["fixture_provider"]
    ml = state.predictors.get("ml")
    store_metadata = ml.store.metadata() if ml is not None else {}
    xg = store_metadata.get("xg", predictor_xg_metadata(state))
    schedule = store_metadata.get("all_competition_schedule", {})
    player = store_metadata.get("player_availability", {})
    freshness = _runtime().freshness_metadata()
    return jsonify({
        "historical_results_source": "Football-Data.co.uk EPL CSV",
        "fixture_source": fixture_provider.metadata(),
        "xg_source": xg,
        "congestion_scope": schedule.get("congestion_scope", "EPL only"),
        "all_competition_schedule": schedule,
        "player_availability": player,
        "xg_matches_available": xg.get("matches_with_xg", 0),
        "data_through": freshness.get("data_through"),
        "last_refresh_success": freshness.get("last_refresh_success"),
    })


def predictor_xg_metadata(state):
    predictor: DixonColesPredictor = state.predictors["v2"]
    return predictor.xg_provider.metadata()


@web.get("/health")
def health():
    repository = _state().repository
    return jsonify({
        "status": "ok",
        "app_version": APP_VERSION,
        "model_version": MODEL_VERSION,
        "challenger_model_version": CHALLENGER_MODEL_VERSION,
        "live_model": "v2",
        "matches": len(repository.matches),
        "active_season": repository.active_season,
        **_runtime().freshness_metadata(),
    })
