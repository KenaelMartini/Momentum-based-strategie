# ============================================================
# Portfolio — positions, cash, ordres, stats
# ============================================================
from __future__ import annotations

import numpy as np
import pandas as pd

from config import INITIAL_CAPITAL

from .events import FillEvent, OrderEvent, PortfolioStats, SignalEvent


class Portfolio:
    """
    Gère positions + cash.
    Ordres soumis à T, exécutés à T+1 (réaliste).
    """

    def __init__(self, initial_capital=INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.prices = {}
        self.history = []
        self.returns_history = []
        self.peak_value = initial_capital
        self.prev_value = initial_capital

    @property
    def positions_value(self):
        return sum(qty * self.prices.get(t, 0.0) for t, qty in self.positions.items())

    @property
    def portfolio_value(self):
        return self.cash + self.positions_value

    def update_prices(self, prices: pd.Series):
        for t in prices.index:
            v = prices[t]
            if not np.isnan(v):
                self.prices[t] = float(v)

    def generate_orders(
        self,
        signal: SignalEvent,
        prices: pd.Series,
        rebal_threshold: float = 0.015,
    ) -> list:
        orders = []
        pv = self.portfolio_value
        target_w = signal.weights

        for t in list(self.positions.keys()):
            if t not in target_w and self.positions[t] != 0:
                p = self.prices.get(t, 0.0)
                if p > 0:
                    orders.append(
                        OrderEvent(date=signal.date, ticker=t, quantity=-self.positions[t], price=p)
                    )

        for t, w in target_w.items():
            p = self.prices.get(t, 0.0)
            if p <= 0:
                continue
            current_w = self.positions.get(t, 0) * p / pv if pv > 0 else 0
            delta_w = w - current_w
            if abs(delta_w) < rebal_threshold:
                continue
            delta_val = delta_w * pv
            orders.append(OrderEvent(date=signal.date, ticker=t, quantity=delta_val / p, price=p))
        return orders

    def fill_order(self, fill: FillEvent):
        t = fill.ticker
        qty = fill.quantity
        cost = qty * fill.fill_price + fill.commission
        self.positions[t] = self.positions.get(t, 0) + qty
        if abs(self.positions[t]) < 0.01:
            del self.positions[t]
        self.cash -= cost

    def compute_stats(
        self,
        date,
        regime_score,
        turnover,
        regime_state="UNKNOWN",
        regime_confidence=0.0,
        trading_suspended=False,
        dd_max_stop=False,
        suspension_reason="",
        suspended_days=0,
        diagnostics=None,
        market_regime_feature="",
        market_regime_effective="",
        market_regime_align_reason="",
        risk_regime_name="",
        informed_tilt_scale=float("nan"),
        informed_tilt_reason="",
        trend_tilt_mult=float("nan"),
    ) -> PortfolioStats:
        diagnostics = diagnostics or {}
        pv = self.portfolio_value
        daily_ret = (pv / self.prev_value - 1) if self.prev_value > 0 else 0.0

        self.returns_history.append(daily_ret)
        self.prev_value = pv
        self.peak_value = max(self.peak_value, pv)
        drawdown = (pv - self.peak_value) / self.peak_value if self.peak_value > 0 else 0.0

        recent = self.returns_history[-21:]
        realized_vol = float(np.std(recent) * np.sqrt(252)) if len(recent) >= 5 else 0.0

        long_r = self.returns_history[-63:]
        long_vol = float(np.std(long_r) * np.sqrt(252)) if len(long_r) >= 21 else realized_vol
        expected_vol = 0.94 * long_vol + 0.06 * realized_vol

        stats = PortfolioStats(
            date=date,
            portfolio_value=pv,
            cash=self.cash,
            positions_value=self.positions_value,
            daily_return=daily_ret,
            realized_vol=realized_vol,
            expected_vol=expected_vol,
            drawdown=drawdown,
            regime_score=regime_score,
            regime_state=regime_state,
            regime_confidence=regime_confidence,
            trading_suspended=bool(trading_suspended),
            dd_max_stop=bool(dd_max_stop),
            suspension_reason=str(suspension_reason or ""),
            suspended_days=int(suspended_days or 0),
            positions=dict(self.positions),
            turnover=turnover,
            rebalancing_day=bool(diagnostics.get("rebalancing_day", False)),
            n_orders=int(diagnostics.get("n_orders", 0)),
            risk_scaling=float(diagnostics.get("risk_scaling", float("nan"))),
            rebal_threshold=float(diagnostics.get("rebal_threshold", float("nan"))),
            rebal_threshold_context=str(diagnostics.get("rebal_threshold_context", "")),
            signal_generation_reason=str(diagnostics.get("signal_generation_reason", "")),
            gross_signal_raw=float(diagnostics.get("gross_signal_raw", float("nan"))),
            gross_after_constraints=float(diagnostics.get("gross_after_constraints", float("nan"))),
            gross_after_risk_manager=float(diagnostics.get("gross_after_risk_manager", float("nan"))),
            gross_after_rebal_threshold=float(diagnostics.get("gross_after_rebal_threshold", float("nan"))),
            gross_after_old_regime_filter=float(diagnostics.get("gross_after_old_regime_filter", float("nan"))),
            gross_after_market_overlay=float(diagnostics.get("gross_after_market_overlay", float("nan"))),
            old_regime_filter_scale=float(diagnostics.get("old_regime_filter_scale", float("nan"))),
            applied_market_overlay_scale=float(diagnostics.get("applied_market_overlay_scale", float("nan"))),
            applied_market_overlay_active=bool(diagnostics.get("applied_market_overlay_active", False)),
            applied_market_overlay_reason=str(diagnostics.get("applied_market_overlay_reason", "")),
            final_turnover=float(diagnostics.get("final_turnover", float("nan"))),
            market_regime_feature=str(market_regime_feature or diagnostics.get("market_regime_feature", "") or ""),
            market_regime_effective=str(market_regime_effective or diagnostics.get("market_regime_effective", "") or ""),
            market_regime_align_reason=str(
                market_regime_align_reason or diagnostics.get("market_regime_align_reason", "") or ""
            ),
            risk_regime_name=str(risk_regime_name or diagnostics.get("risk_regime_name", "") or ""),
            informed_tilt_scale=float(diagnostics.get("informed_tilt_scale", informed_tilt_scale)),
            informed_tilt_reason=str(diagnostics.get("informed_tilt_reason", informed_tilt_reason) or ""),
            trend_tilt_mult=float(diagnostics.get("trend_tilt_mult", trend_tilt_mult)),
            defensive_flat_phase=str(diagnostics.get("defensive_flat_phase", "") or ""),
            defensive_flat_reason=str(diagnostics.get("defensive_flat_reason", "") or ""),
        )

        self.history.append(stats)
        return stats
