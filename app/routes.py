from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template, request

from .model import DixonColesPredictor


web = Blueprint("web", __name__)


def _repository():
    return current_app.extensions["match_repository"]


def _predictors():
    return current_app.extensions["predictors"]


@web.get("/")
def index():
    repository = _repository()
    teams = repository.active_teams()
    default_home = "Arsenal" if "Arsenal" in teams else teams[0]
    default_away = "Liverpool" if "Liverpool" in teams else teams[1]
    return render_template(
        "index.html",
        teams=teams,
        default_home=default_home,
        default_away=default_away,
        last_updated=repository.last_updated,
        freshness=repository.freshness_metadata(),
    )


@web.get("/fixtures")
def fixtures():
    repository = _repository()
    return render_template(
        "fixtures.html",
        upcoming=repository.upcoming_fixtures(),
        results=repository.latest_results(),
        last_updated=repository.last_updated,
        freshness=repository.freshness_metadata(),
    )


@web.get("/model")
def model_info():
    repository = _repository()
    predictor: DixonColesPredictor = _predictors()["v2"]
    return render_template(
        "model.html",
        report=current_app.extensions.get("backtest_report", {}),
        matches=len(repository.matches),
        active_season=repository.active_season,
        live_model="Enhanced Dixon-Coles V2",
        xg=predictor.xg_provider.metadata(),
        freshness=repository.freshness_metadata(),
    )


@web.get("/api/predict")
def predict():
    home_team = request.args.get("home", "").strip()
    away_team = request.args.get("away", "").strip()
    if not home_team or not away_team:
        return jsonify({"error": "home and away query parameters are required"}), 400
    model_key = request.args.get("model", "v2").lower()
    if model_key not in _predictors():
        return jsonify({"error": "model must be v1 or v2"}), 400
    try:
        prediction = _predictors()[model_key].predict(home_team, away_team)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    freshness = _repository().freshness_metadata()
    prediction["data_stale"] = freshness["data_stale"]
    prediction["metadata"]["freshness"] = freshness
    prediction["metadata"]["live_model"] = model_key == "v2"
    return jsonify(prediction)


@web.get("/api/fixtures")
def api_fixtures():
    limit = min(max(request.args.get("limit", 20, type=int), 1), 100)
    repository = _repository()
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
    repository = _repository()
    predictor: DixonColesPredictor = _predictors()["v2"]
    return jsonify({
        "version": "2.0",
        "live_model": "Enhanced Dixon-Coles V2",
        "matches": len(repository.matches),
        "active_season": repository.active_season,
        "xg": predictor.xg_provider.metadata(),
        "backtest": current_app.extensions.get("backtest_report", {}),
        "freshness": repository.freshness_metadata(),
    })


@web.get("/health")
def health():
    repository = _repository()
    return jsonify({
        "status": "ok",
        "model_version": "2.0",
        "matches": len(repository.matches),
        "active_season": repository.active_season,
        **repository.freshness_metadata(),
    })
