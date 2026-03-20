from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .regime_classifier import RegimeClassifier
from .regime_features import RegimeFeatureCalculator
from .risk_types import RegimeOverlayDecision, RegimeSnapshot, RegimeState

try:
    from config import (
        ENABLE_MARKET_OVERLAY,
        MARKET_OVERLAY_MODE,
        RISK_OFF_MAX_SCALE,
        RISK_OFF_MIN_SCALE,
        TRANSITION_OVERLAY_ENABLED,
    )
except ImportError:
    ENABLE_MARKET_OVERLAY = True
    MARKET_OVERLAY_MODE = "risk_off_only"
    TRANSITION_OVERLAY_ENABLED = False
    RISK_OFF_MIN_SCALE = 0.15
    RISK_OFF_MAX_SCALE = 0.35


class RegimeEngine:
    def __init__(
        self,
        vix_series: Optional[pd.Series] = None,
        feature_calculator: Optional[RegimeFeatureCalculator] = None,
        classifier: Optional[RegimeClassifier] = None,
        track_history: bool = True,
    ):
        self.feature_calculator = feature_calculator or RegimeFeatureCalculator(
            vix_series=vix_series,
        )
        self.classifier = classifier or RegimeClassifier()
        self.track_history = track_history
        self._history: list[RegimeSnapshot] = []

    def compute(self, prices: pd.DataFrame) -> Optional[RegimeSnapshot]:
        features = self.feature_calculator.compute(prices)
        if features is None:
            return None

        snapshot = self.classifier.build_snapshot(prices.index[-1], features)
        if self.track_history:
            self._history.append(snapshot)
        return snapshot

    @property
    def history(self) -> tuple[RegimeSnapshot, ...]:
        return tuple(self._history)

    def history_frame(self, prefix: str = "market_") -> pd.DataFrame:
        rows = []
        for snapshot in self._history:
            overlay = decide_event_driven_overlay(snapshot)
            row = {
                "date": snapshot.date,
                f"{prefix}regime_state": snapshot.state.value,
                f"{prefix}regime_confidence": snapshot.confidence,
                f"{prefix}regime_score": snapshot.composite_score,
                f"{prefix}exposure_multiplier": snapshot.exposure_multiplier,
                f"{prefix}overlay_scale": overlay.scale,
                f"{prefix}overlay_active": overlay.active,
                f"{prefix}overlay_reason": overlay.reason,
            }
            for name, value in snapshot.features.items():
                row[f"{prefix}{name}"] = value
            rows.append(row)
        return pd.DataFrame(rows)


def decide_event_driven_overlay(
    snapshot: Optional[RegimeSnapshot],
) -> RegimeOverlayDecision:
    if snapshot is None:
        return RegimeOverlayDecision(
            state="UNKNOWN",
            scale=1.0,
            active=False,
            reason="NO_SNAPSHOT",
        )

    if not ENABLE_MARKET_OVERLAY or str(MARKET_OVERLAY_MODE).lower() == "disabled":
        return RegimeOverlayDecision(
            state=snapshot.state.value,
            scale=1.0,
            active=False,
            reason="OVERLAY_DISABLED",
            confidence=snapshot.confidence,
            composite_score=snapshot.composite_score,
        )

    if snapshot.state == RegimeState.TRANSITION:
        mode = str(MARKET_OVERLAY_MODE).lower()
        if mode == "risk_off_only" or not TRANSITION_OVERLAY_ENABLED:
            return RegimeOverlayDecision(
                state=snapshot.state.value,
                scale=1.0,
                active=False,
                reason="TRANSITION_MONITOR_ONLY",
                confidence=snapshot.confidence,
                composite_score=snapshot.composite_score,
            )

        weak_transition = (
            snapshot.composite_score < 0.50
            or (
                snapshot.composite_score < 0.60
                and snapshot.confidence < 0.32
            )
        )
        if not weak_transition:
            return RegimeOverlayDecision(
                state=snapshot.state.value,
                scale=1.0,
                active=False,
                reason="TRANSITION_MONITOR_ONLY",
                confidence=snapshot.confidence,
                composite_score=snapshot.composite_score,
            )

        scale = float(
            np.clip(
                0.80 + 0.15 * snapshot.composite_score + 0.05 * snapshot.confidence,
                0.82,
                0.94,
            )
        )
        return RegimeOverlayDecision(
            state=snapshot.state.value,
            scale=scale,
            active=True,
            reason="WEAK_TRANSITION_REDUCTION",
            confidence=snapshot.confidence,
            composite_score=snapshot.composite_score,
        )

    if snapshot.state == RegimeState.RISK_OFF:
        scale = float(
            np.clip(
                0.10 + 0.25 * snapshot.confidence + 0.20 * snapshot.composite_score,
                RISK_OFF_MIN_SCALE,
                RISK_OFF_MAX_SCALE,
            )
        )
        return RegimeOverlayDecision(
            state=snapshot.state.value,
            scale=scale,
            active=True,
            reason="RISK_OFF_REDUCTION",
            confidence=snapshot.confidence,
            composite_score=snapshot.composite_score,
        )

    return RegimeOverlayDecision(
        state=snapshot.state.value,
        scale=1.0,
        active=False,
        reason="NO_OVERLAY",
        confidence=snapshot.confidence,
        composite_score=snapshot.composite_score,
    )


def build_regime_log_frame(
    frame: pd.DataFrame,
    regime_col: str = "market_regime_state",
    score_col: str = "market_regime_score",
    confidence_col: str = "market_regime_confidence",
) -> pd.DataFrame:
    if frame is None or frame.empty or regime_col not in frame.columns:
        return pd.DataFrame()

    base_columns = [
        "date",
        regime_col,
        score_col,
        confidence_col,
        "portfolio_value",
        "daily_return",
        "drawdown",
        "turnover",
    ]
    available_columns = [column for column in base_columns if column in frame.columns]
    regime_log = frame.loc[frame[regime_col].notna(), available_columns].copy()
    if regime_log.empty:
        return regime_log

    regime_log["regime_changed"] = regime_log[regime_col].ne(regime_log[regime_col].shift(1))
    regime_log.loc[regime_log.index[0], "regime_changed"] = False
    return regime_log


def summarize_regime_performance(
    frame: pd.DataFrame,
    regime_col: str = "market_regime_state",
    score_col: str = "market_regime_score",
    returns_col: str = "daily_return",
    turnover_col: str = "turnover",
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    if frame is None or frame.empty or regime_col not in frame.columns or returns_col not in frame.columns:
        return pd.DataFrame()

    working = frame.loc[frame[regime_col].notna()].copy()
    if working.empty:
        return pd.DataFrame()

    rf_daily = (1.0 + risk_free_rate) ** (1 / 252) - 1
    total_days = len(working)
    rows = []

    for regime_state, group in working.groupby(regime_col):
        returns = group[returns_col].astype(float)
        if returns.empty:
            continue

        days = len(group)
        total_return = float((1.0 + returns).prod() - 1.0)
        if days > 0 and total_return > -1.0:
            annualized_return = float((1.0 + total_return) ** (252 / days) - 1.0)
        else:
            annualized_return = float("nan")

        std = float(returns.std(ddof=0)) if days > 1 else 0.0
        annualized_vol = float(std * np.sqrt(252)) if days > 1 else 0.0
        if std > 1e-12:
            sharpe = float(((returns.mean() - rf_daily) / std) * np.sqrt(252))
        else:
            sharpe = float("nan")

        equity_curve = (1.0 + returns).cumprod()
        if equity_curve.empty:
            max_drawdown = float("nan")
        else:
            running_peak = equity_curve.cummax()
            max_drawdown = float((equity_curve / running_peak - 1.0).min())

        if pd.notna(annualized_return) and max_drawdown < 0:
            calmar = float(annualized_return / abs(max_drawdown))
        else:
            calmar = float("nan")

        rows.append(
            {
                "regime_state": regime_state,
                "days": int(days),
                "pct_time": float(days / total_days),
                "total_return": total_return,
                "annualized_return": annualized_return,
                "annualized_vol": annualized_vol,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "calmar": calmar,
                "hit_rate": float((returns > 0).mean()),
                "avg_turnover": float(group[turnover_col].mean()) if turnover_col in group.columns else float("nan"),
                "avg_regime_score": float(group[score_col].mean()) if score_col in group.columns else float("nan"),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    preferred_order = {
        "TREND": 0,
        "RISK_ON": 1,
        "TRANSITION": 2,
        "RISK_OFF": 3,
        "BULL": 4,
        "NORMAL": 5,
        "STRESS": 6,
        "CRISIS": 7,
    }
    summary["_order"] = summary["regime_state"].map(lambda value: preferred_order.get(value, 999))
    summary = summary.sort_values(["_order", "regime_state"]).drop(columns="_order").reset_index(drop=True)
    return summary
