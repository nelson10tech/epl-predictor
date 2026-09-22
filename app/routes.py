from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template, request

from .model import EloPoissonPredictor


web = Blueprint("web", __name__)


def _repository():
    return current_app.extensions["match_repository"]


def _predictor() -> EloPoissonPredictor:
    return current_app.extensions["predictor"]


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
    )


@web.get("/fixtures")
def fixtures():
    repository = _repository()
    return render_template(
        "fixtures.html",
        fixtures=repository.latest_fixtures(),
        last_updated=repository.last_updated,
    )


@web.get("/api/predict")
def predict():
    home_team = request.args.get("home", "").strip()
    away_team = request.args.get("away", "").strip()
    if not home_team or not away_team:
        return jsonify({"error": "home and away query parameters are required"}), 400
    try:
        prediction = _predictor().predict(home_team, away_team)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(prediction)


@web.get("/api/fixtures")
def api_fixtures():
    limit = min(max(request.args.get("limit", 20, type=int), 1), 100)
    matches = _repository().latest_fixtures(limit)
    return jsonify(
        {
            "fixtures": [
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


@web.post("/api/refresh")
def refresh():
    repository = _repository()
    try:
        summary = repository.refresh()
        current_app.extensions["predictor"] = EloPoissonPredictor(repository.matches)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502
    return jsonify({"status": "ok", **summary})


@web.get("/health")
def health():
    repository = _repository()
    return jsonify(
        {
            "status": "ok",
            "matches": len(repository.matches),
            "last_updated": repository.last_updated.isoformat()
            if repository.last_updated
            else None,
        }
    )

