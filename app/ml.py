from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Optional

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from .data import Match
from .feature_store import (
    ML_FEATURE_NAMES,
    ChronologicalFeatureStore,
    FeatureRow,
    feature_vector,
)
from .model import (
    OUTCOMES,
    DixonColesPredictor,
    TemperatureCalibrator,
    dixon_coles_score_matrix,
    normalize_matrix,
    outcome_probabilities,
    prediction_payload,
)


RESULT_LABEL = {"H": 0, "D": 1, "A": 2}


def new_classifier(max_iter: int = 80) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.045,
        max_iter=max_iter,
        max_leaf_nodes=12,
        min_samples_leaf=25,
        l2_regularization=1.5,
        early_stopping=False,
        random_state=42,
    )


def fit_classifier(rows: list[FeatureRow], max_iter: int = 80):
    if len(rows) < 100:
        raise ValueError("V3 ML requires at least 100 chronological training matches")
    classifier = new_classifier(max_iter=max_iter)
    classifier.fit(
        np.asarray([row.vector() for row in rows], dtype=float),
        np.asarray([RESULT_LABEL[row.result] for row in rows], dtype=int),
    )
    return classifier


def classifier_probabilities(classifier, vector: list[float]) -> dict[str, float]:
    predicted = classifier.predict_proba(np.asarray([vector], dtype=float))[0]
    by_class = {int(label): float(value) for label, value in zip(classifier.classes_, predicted)}
    values = [max(by_class.get(index, 0.0), 1e-12) for index in range(3)]
    total = sum(values)
    return dict(zip(OUTCOMES, (value / total for value in values)))


def blend_probabilities(v2: dict, ml: dict, v2_weight: float) -> dict[str, float]:
    weight = max(0.0, min(1.0, float(v2_weight)))
    blended = {
        key: weight * v2[key] + (1.0 - weight) * ml[key]
        for key in OUTCOMES
    }
    total = sum(blended.values())
    return {key: value / total for key, value in blended.items()}


def reweight_score_matrix(matrix, probabilities: dict[str, float]):
    raw = outcome_probabilities(matrix)
    adjusted = []
    for home, away, probability in matrix:
        outcome = "home_win" if home > away else "away_win" if home < away else "draw"
        adjusted.append(
            (home, away, probability * probabilities[outcome] / max(raw[outcome], 1e-12))
        )
    return normalize_matrix(adjusted)


class V3MLPredictor:
    """Lightweight challenger using only chronological football-strength features."""

    def __init__(
        self,
        matches: list[Match],
        v2_predictor: DixonColesPredictor,
        calibration_temperature: float = 1.0,
        max_iter: int = 80,
        feature_importance: Optional[list[dict]] = None,
        calculate_importance: bool = False,
    ) -> None:
        self.matches = sorted(matches, key=lambda item: item.played_on)
        self.v2_predictor = v2_predictor
        self.store = ChronologicalFeatureStore(self.matches)
        self.classifier = fit_classifier(self.store.rows, max_iter=max_iter)
        self.calibrator = TemperatureCalibrator(calibration_temperature)
        self.feature_importance = list(feature_importance or [])
        if calculate_importance and not self.feature_importance:
            self.feature_importance = self._calculate_feature_importance(max_iter)

    def _calculate_feature_importance(self, max_iter: int) -> list[dict]:
        split = max(100, int(len(self.store.rows) * 0.8))
        if len(self.store.rows) - split < 100:
            return []
        training = self.store.rows[:split]
        validation = self.store.rows[split:]
        probe = fit_classifier(training, max_iter=min(max_iter, 60))
        x_validation = np.asarray([row.vector() for row in validation], dtype=float)
        y_validation = np.asarray([RESULT_LABEL[row.result] for row in validation])
        importance = permutation_importance(
            probe,
            x_validation,
            y_validation,
            scoring="neg_log_loss",
            n_repeats=3,
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

    def raw_probabilities(self, home_team: str, away_team: str, as_of: date) -> tuple[dict, object]:
        if home_team == away_team:
            raise ValueError("Home and away teams must be different")
        known = {team for match in self.matches for team in (match.home_team, match.away_team)}
        if home_team not in known or away_team not in known:
            raise ValueError("Both teams must exist in the loaded EPL data")
        snapshot = self.store.live_snapshot(home_team, away_team, as_of)
        probabilities = classifier_probabilities(
            self.classifier, feature_vector(snapshot.values)
        )
        return probabilities, snapshot

    def predict(
        self,
        home_team: str,
        away_team: str,
        as_of: Optional[date] = None,
    ) -> dict:
        as_of = as_of or (self.matches[-1].played_on + timedelta(days=1))
        raw, snapshot = self.raw_probabilities(home_team, away_team, as_of)
        probabilities = self.calibrator.apply(raw)
        v2 = self.v2_predictor.predict(home_team, away_team, as_of=as_of)
        expected_home = v2["expected_goals"]["home"]
        expected_away = v2["expected_goals"]["away"]
        matrix = reweight_score_matrix(
            dixon_coles_score_matrix(expected_home, expected_away), probabilities
        )
        return prediction_payload(
            home_team,
            away_team,
            expected_home,
            expected_away,
            matrix,
            model="V3 HistGradientBoosting ML",
            version="3.0-ml",
            data_through=self.matches[-1].played_on,
            xg_enabled=self.store.xg_provider.available(),
            factors=[
                {
                    "label": "ML feature store",
                    "value": "Pre-match Elo, form, venue, xG where genuine, and context inputs",
                }
            ],
            extra_metadata={
                "training_through": snapshot.training_through.isoformat()
                if snapshot.training_through else None,
                "calibration_temperature": self.calibrator.temperature,
                "feature_importance_method": "chronological holdout permutation log loss",
                "feature_importance": self.feature_importance,
                "data_provenance": self.store.metadata(),
            },
        )


class V3EnsemblePredictor:
    def __init__(
        self,
        v2_predictor: DixonColesPredictor,
        ml_predictor: V3MLPredictor,
        v2_weight: float,
    ) -> None:
        self.v2_predictor = v2_predictor
        self.ml_predictor = ml_predictor
        self.v2_weight = max(0.0, min(1.0, float(v2_weight)))
        self.matches = ml_predictor.matches

    def predict(
        self,
        home_team: str,
        away_team: str,
        as_of: Optional[date] = None,
    ) -> dict:
        as_of = as_of or (self.matches[-1].played_on + timedelta(days=1))
        v2 = self.v2_predictor.predict(home_team, away_team, as_of=as_of)
        ml = self.ml_predictor.predict(home_team, away_team, as_of=as_of)
        probabilities = blend_probabilities(
            v2["probabilities"], ml["probabilities"], self.v2_weight
        )
        expected_home = v2["expected_goals"]["home"]
        expected_away = v2["expected_goals"]["away"]
        matrix = reweight_score_matrix(
            dixon_coles_score_matrix(expected_home, expected_away), probabilities
        )
        factors = list(v2.get("factors", []))
        factors.append({
            "label": "Challenger ensemble",
            "value": (
                f"V2 {self.v2_weight:.0%} · calibrated ML {1.0 - self.v2_weight:.0%}"
            ),
        })
        return prediction_payload(
            home_team,
            away_team,
            expected_home,
            expected_away,
            matrix,
            model="V3 Ensemble",
            version="3.0",
            data_through=self.matches[-1].played_on,
            xg_enabled=self.ml_predictor.store.xg_provider.available(),
            factors=factors,
            extra_metadata={
                "challenger": True,
                "live_default": False,
                "v2_weight": self.v2_weight,
                "ml_weight": 1.0 - self.v2_weight,
                "ml_calibration_temperature": self.ml_predictor.calibrator.temperature,
                "feature_importance": self.ml_predictor.feature_importance,
                "data_provenance": self.ml_predictor.store.metadata(),
            },
        )
