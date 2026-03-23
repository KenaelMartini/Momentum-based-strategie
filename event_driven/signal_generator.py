# ============================================================
# MomentumSignalGenerator — signal à la date T (sous-ensemble research / legacy)
# ============================================================
# Le moteur principal utilise MomentumSignalGeneratorV2 (event_driven_risk).
# ============================================================
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import MOMENTUM_WEIGHTS
from risk import RegimeEngine, RegimeState

from .events import Signal, SignalEvent

logger = logging.getLogger(__name__)


class MomentumSignalGenerator:
    """
    Calcule le signal momentum sur les données disponibles à T.
    NE VOIT JAMAIS les données au-delà de T.

    Score_i = Σ w_k * log(P_{T-skip} / P_{T-skip-window_k})
    """

    def __init__(
        self,
        data_handler,
        momentum_weights=None,
        skip_days=10,
        long_quantile=0.70,
        short_quantile=0.30,
        regime_engine=None,
    ):
        self.data = data_handler
        self.weights = momentum_weights or MOMENTUM_WEIGHTS
        self.skip_days = skip_days
        self.long_q = long_quantile
        self.short_q = short_quantile
        self.max_window = max(self.weights.keys())
        self.regime_engine = regime_engine or RegimeEngine()

    def compute_signal(self, date):
        required = self.max_window + self.skip_days + 10
        prices = self.data.get_history(date, max(required, 300))
        if len(prices) < max(required, 200):
            return None

        regime_snapshot = self.regime_engine.compute(prices)
        if regime_snapshot is None:
            return None

        scores = pd.Series(0.0, index=prices.columns)
        for window, weight in self.weights.items():
            idx_end = -self.skip_days if self.skip_days > 0 else len(prices)
            idx_start = -(window + self.skip_days)
            p_end = prices.iloc[idx_end]
            p_start = prices.iloc[idx_start]
            ret = np.log(p_end / p_start).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            scores += weight * ret

        if regime_snapshot.state == RegimeState.RISK_OFF and regime_snapshot.confidence > 0.60:
            return SignalEvent(
                date=date,
                weights={},
                regime=regime_snapshot.composite_score,
                regime_state=regime_snapshot.state.value,
                regime_confidence=regime_snapshot.confidence,
                signal=Signal.FLAT,
            )

        valid = scores.dropna()
        q_high = valid.quantile(self.long_q)
        q_low = valid.quantile(self.short_q)

        longs = valid[valid >= q_high].index.tolist()
        shorts = valid[valid <= q_low].index.tolist()

        weights_dict = {}

        if regime_snapshot.state == RegimeState.TREND:
            short_multiplier = 0.70
        elif regime_snapshot.state == RegimeState.RISK_ON:
            short_multiplier = 0.40
        elif regime_snapshot.state == RegimeState.TRANSITION:
            short_multiplier = 0.20
        else:
            short_multiplier = 0.00

        base_weight = min(regime_snapshot.exposure_multiplier / max(len(longs), 1), 0.10)

        for t in longs:
            weights_dict[t] = base_weight

        for t in shorts:
            weights_dict[t] = -base_weight * short_multiplier

        signal = Signal.FLAT if len(weights_dict) == 0 else Signal.LONG

        return SignalEvent(
            date=date,
            weights=weights_dict,
            regime=regime_snapshot.composite_score,
            regime_state=regime_snapshot.state.value,
            regime_confidence=regime_snapshot.confidence,
            signal=signal,
        )

    def _compute_regime(self, prices: pd.DataFrame):
        scores = []
        avg_corr = 0.0
        if len(prices) >= 200:
            ma200 = prices.iloc[-200:].mean()
            scores.append((prices.iloc[-1] > ma200).mean())
        if len(prices) >= 63:
            rets = np.log(prices / prices.shift(1)).fillna(0)
            vol_10j = rets.iloc[-10:].std().mean()
            vol_63j = rets.iloc[-63:].std().mean()
            vol_ratio = vol_10j / (vol_63j + 1e-8)
            if vol_ratio >= 2.0:
                vol_score = 0.0
            elif vol_ratio >= 1.5:
                vol_score = 0.25
            elif vol_ratio >= 1.25:
                vol_score = float(np.interp(vol_ratio, [1.25, 1.5], [1.0, 0.25]))
            else:
                vol_score = 1.0
            scores.append(vol_score)
            corr = rets.iloc[-42:].corr()
            n = len(corr)
            avg_corr = (corr.sum().sum() - n) / (n * (n - 1) + 1e-8)
            if avg_corr > 0.45:
                corr_score = 0.10
            elif avg_corr > 0.35:
                corr_score = float(np.interp(avg_corr, [0.35, 0.45], [1.0, 0.10]))
            else:
                corr_score = 1.0
            scores.append(corr_score)
        if not scores:
            return 0.5, 0.5
        s = float(min(scores))

        if len(scores) >= 3:
            logger.info(
                f"  REGIME: trend={scores[0]:.2f} vol={scores[1]:.2f} corr={scores[2]:.2f} "
                f"avg_corr={avg_corr:.3f} s={s:.2f}"
            )

        if s >= 0.70:
            m = 1.0
        elif s >= 0.50:
            m = 0.5 + (s - 0.50) / 0.20 * 0.5
        elif s >= 0.30:
            m = 0.25 + (s - 0.30) / 0.20 * 0.25
        elif s >= 0.10:
            m = 0.25
        else:
            m = 0.0
        return s, m
