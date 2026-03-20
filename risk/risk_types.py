from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class RegimeState(Enum):
    TREND = "TREND"
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    TRANSITION = "TRANSITION"


@dataclass(frozen=True)
class RegimeFeatures:
    trend_score: float
    vol_score: float
    corr_score: float
    breadth_score: float
    dispersion_score: float
    vix_score: float
    vol_ratio_20_63: float
    avg_corr_42d: float
    breadth_200d: float
    dispersion_20d: float
    vix_level: float = float("nan")
    vix_5d_change: float = float("nan")
    market_63d_return: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "trend_score": float(self.trend_score),
            "vol_score": float(self.vol_score),
            "corr_score": float(self.corr_score),
            "breadth_score": float(self.breadth_score),
            "dispersion_score": float(self.dispersion_score),
            "vix_score": float(self.vix_score),
            "vol_ratio_20_63": float(self.vol_ratio_20_63),
            "avg_corr_42d": float(self.avg_corr_42d),
            "breadth_200d": float(self.breadth_200d),
            "dispersion_20d": float(self.dispersion_20d),
            "vix_level": float(self.vix_level),
            "vix_5d_change": float(self.vix_5d_change),
            "market_63d_return": float(self.market_63d_return),
        }

    def __getitem__(self, key: str) -> float:
        return self.as_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)

    def items(self):
        return self.as_dict().items()

    def keys(self):
        return self.as_dict().keys()

    def values(self):
        return self.as_dict().values()

    def __iter__(self):
        return iter(self.as_dict())


@dataclass(frozen=True)
class RegimeSnapshot:
    date: pd.Timestamp
    state: RegimeState
    confidence: float
    composite_score: float
    features: RegimeFeatures
    exposure_multiplier: float


@dataclass(frozen=True)
class RegimeOverlayDecision:
    state: str
    scale: float
    active: bool
    reason: str
    confidence: float = 0.0
    composite_score: float = 0.0
