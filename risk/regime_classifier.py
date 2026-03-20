from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

from .risk_types import RegimeFeatures, RegimeSnapshot, RegimeState


DEFAULT_COMPOSITE_WEIGHTS = {
    "trend_score": 0.28,
    "vol_score": 0.18,
    "corr_score": 0.16,
    "breadth_score": 0.18,
    "dispersion_score": 0.10,
    "vix_score": 0.10,
}


class RegimeClassifier:
    def __init__(self, composite_weights: Optional[Mapping[str, float]] = None):
        self.composite_weights = dict(DEFAULT_COMPOSITE_WEIGHTS)
        if composite_weights:
            self.composite_weights.update(composite_weights)

    def build_snapshot(self, date: pd.Timestamp, features: RegimeFeatures) -> RegimeSnapshot:
        composite = self.compute_composite(features)
        state = self.classify_state(features, composite)
        confidence = self.compute_confidence(features, state)
        exposure_multiplier = self.map_exposure(state, confidence, composite)

        return RegimeSnapshot(
            date=date,
            state=state,
            confidence=float(confidence),
            composite_score=float(composite),
            features=features,
            exposure_multiplier=float(exposure_multiplier),
        )

    def compute_composite(self, features: RegimeFeatures) -> float:
        score_keys = list(self.composite_weights.keys())
        values = [float(features[key]) for key in score_keys]
        weights = [float(self.composite_weights[key]) for key in score_keys]
        return float(np.average(values, weights=weights))

    def classify_state(self, features: RegimeFeatures, composite: float) -> RegimeState:
        market_ret_63d = float(features.get("market_63d_return", 0.0))
        stress = np.mean(
            [
                1.0 - float(features["vol_score"]),
                1.0 - float(features["corr_score"]),
                1.0 - float(features["vix_score"]),
            ]
        )

        if stress > 0.60 and float(features["breadth_score"]) < 0.40 and market_ret_63d < 0:
            return RegimeState.RISK_OFF

        if (
            composite >= 0.72
            and float(features["trend_score"]) >= 0.70
            and float(features["breadth_score"]) >= 0.60
            and float(features["corr_score"]) >= 0.50
        ):
            return RegimeState.TREND

        if market_ret_63d > 0 and float(features["breadth_score"]) >= 0.55 and stress < 0.45:
            return RegimeState.RISK_ON

        return RegimeState.TRANSITION

    def compute_confidence(self, features: RegimeFeatures, state: RegimeState) -> float:
        values = np.array(
            [
                float(features["trend_score"]),
                float(features["vol_score"]),
                float(features["corr_score"]),
                float(features["breadth_score"]),
                float(features["dispersion_score"]),
                float(features["vix_score"]),
            ],
            dtype=float,
        )

        dispersion = values.std()
        base = values.mean()

        if state == RegimeState.TRANSITION:
            confidence = max(0.15, 0.55 - dispersion)
        else:
            confidence = min(1.0, base * (1.0 - 0.5 * dispersion))

        return float(np.clip(confidence, 0.05, 1.0))

    def map_exposure(self, state: RegimeState, confidence: float, composite: float) -> float:
        if state == RegimeState.TREND:
            multiplier = 0.90 + 0.30 * confidence
        elif state == RegimeState.RISK_ON:
            multiplier = 0.65 + 0.25 * confidence
        elif state == RegimeState.RISK_OFF:
            multiplier = 0.10 + 0.20 * confidence
        else:
            multiplier = 0.20 + 0.20 * confidence

        multiplier *= np.clip(composite, 0.25, 1.0)
        return float(np.clip(multiplier, 0.0, 1.10))
