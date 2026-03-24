# ============================================================
# EventDrivenEngine — boucle principale du backtest event-driven
# ============================================================
from __future__ import annotations

import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from config import (
    BACKTEST_END,
    BACKTEST_START,
    EVENT_DRIVEN_BASELINE_JSON,
    INITIAL_CAPITAL,
    REBALANCE_FILL_SAME_BAR,
    RESEARCH_TRANSACTION_COST_STRESS_MULTIPLIER,
    RISK_FREE_RATE,
    SLIPPAGE_BPS,
    TRANSACTION_COST_BPS,
)
from risk import (
    DefensiveFlatController,
    RegimeEngine,
    align_market_regime_with_risk,
    apply_market_regime_overlay as apply_market_regime_overlay_helper,
    apply_regime_weight_filter as apply_regime_weight_filter_helper,
    apply_risk_informed_exposure_tilt as apply_risk_informed_exposure_tilt_helper,
    build_regime_log_frame,
    decide_event_driven_overlay,
    summarize_regime_performance,
)

from .baseline import compare_with_baseline_reference, evaluate_baseline_verdict
from .broker import SimulatedBroker
from .data_handler import DataHandler
from .events import FillEvent, OrderEvent, PortfolioStats, Signal, SignalEvent
from .portfolio import Portfolio
from .visualizer import LiveVisualizer3D

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

PerDayCallback = Optional[Callable[[PortfolioStats, "EventDrivenEngine"], None]]


class EventDrivenEngine:
    """
    Boucle principale :
    Pour chaque jour T :
      1. get_next_bar()     → MarketEvent
      2. execute_pending()  → FillEvent   (ordres de T-1)
      3. compute_signal()   → SignalEvent (mensuel uniquement)
      4. generate_orders()  → OrderEvent
      5. submit_order()     → pending     (exécutés à T+1)
      6. compute_stats()    → PortfolioStats
      7. update_visualizer()

    REBALANCEMENT MENSUEL :
    Le signal est recalculé une fois par mois seulement,
    aligné avec le backtest vectorisé (REBALANCING_FREQUENCY = "monthly").
    """

    def __init__(
        self,
        data_path,
        start_date=None,
        end_date=None,
        initial_capital=INITIAL_CAPITAL,
        live_viz=False,
        output_dir="./results/event_driven",
        per_day_callback: PerDayCallback = None,
        day_sleep_sec: float = 0.0,
        baseline_reference_path: Path | str | None = None,
        skip_baseline_comparison: bool = False,
        transaction_cost_stress_multiplier: float | None = None,
        rebalance_threshold: float | None = None,
        skip_strategy_benchmark_report: bool = False,
    ):
        if start_date is None:
            start_date = BACKTEST_START
        if end_date is None:
            end_date = BACKTEST_END

        self._skip_baseline_comparison = bool(skip_baseline_comparison)
        if skip_baseline_comparison:
            self._baseline_reference_path = None
        elif baseline_reference_path is not None:
            self._baseline_reference_path = Path(baseline_reference_path)
        else:
            self._baseline_reference_path = Path(EVENT_DRIVEN_BASELINE_JSON)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._data_path = Path(data_path)
        self._skip_strategy_benchmark_report = bool(skip_strategy_benchmark_report)
        self.live_viz = live_viz
        self.per_day_callback = per_day_callback
        self.day_sleep_sec = float(day_sleep_sec)

        from event_driven_risk import EventDrivenRiskManager, MomentumSignalGeneratorV2

        self.data_handler = DataHandler(data_path, start_date, end_date)
        self.risk_manager = EventDrivenRiskManager(initial_capital)
        self.signal_gen = MomentumSignalGeneratorV2(
            self.data_handler,
            self.risk_manager,
            rebalance_threshold=rebalance_threshold,
        )
        self.portfolio = Portfolio(initial_capital)
        _cm = (
            float(transaction_cost_stress_multiplier)
            if transaction_cost_stress_multiplier is not None
            else float(RESEARCH_TRANSACTION_COST_STRESS_MULTIPLIER)
        )
        _cm = max(_cm, 1e-9)
        self.broker = SimulatedBroker(
            commission_rate=(TRANSACTION_COST_BPS / 10_000) * _cm,
            slippage_rate=(SLIPPAGE_BPS / 10_000) * _cm,
        )
        self.visualizer = LiveVisualizer3D(update_every=21)
        self.market_regime_engine = RegimeEngine(track_history=True)
        self.defensive_flat_ctrl = DefensiveFlatController()
        self.n_trades = 0
        self.last_rebal_date = None
        self.last_market_regime_state = None
        self.last_aligned_market_regime = None
        self._day_regime_ctx: dict = {}
        self.final_metrics = {}
        self.rebal_diagnostics = []
        self.prev_prices = None
        self._prev_trading_suspended = False
        self._rebalance_immediately_after_reentry = False

    def run(self) -> dict:
        logger.info("\n" + "=" * 60)
        logger.info("  BACKTEST EVENT-DRIVEN — PHASE 3")
        logger.info("=" * 60)

        day_count = 0

        while self.data_handler.has_data:
            market_event = self.data_handler.get_next_bar()
            if market_event is None:
                break

            date = market_event.date
            prices = market_event.prices

            self.portfolio.update_prices(prices)

            fills = self.broker.execute_pending(prices)
            for fill in fills:
                self.portfolio.fill_order(fill)
                self.n_trades += 1

            turnover = 0.0

            risk_snapshot = self.risk_manager.update(
                date=date,
                prices=prices,
                portfolio_value=self.portfolio.portfolio_value,
                current_positions=self.portfolio.positions,
                entry_prices={t: self.portfolio.prices.get(t, 0) for t in self.portfolio.positions},
                prev_prices=self.prev_prices,
            )
            suspended_now = bool(getattr(risk_snapshot, "trading_suspended", False))
            if self._prev_trading_suspended and not suspended_now:
                self._rebalance_immediately_after_reentry = True
            self._prev_trading_suspended = suspended_now

            self.signal_gen.update_ewma_vol(prices, self.prev_prices)
            regime_score = risk_snapshot.regime_score
            risk_regime_name = getattr(getattr(risk_snapshot, "regime", None), "name", "NORMAL")
            daily_rebal_diagnostics = {
                "rebalancing_day": False,
                "n_orders": 0,
                "risk_scaling": float(getattr(risk_snapshot, "risk_scaling", float("nan"))),
                "rebal_threshold": float("nan"),
                "rebal_threshold_context": "",
                "signal_generation_reason": "",
                "gross_signal_raw": float("nan"),
                "gross_after_constraints": float("nan"),
                "gross_after_risk_manager": float("nan"),
                "gross_after_rebal_threshold": float("nan"),
                "gross_after_old_regime_filter": float("nan"),
                "gross_after_market_overlay": float("nan"),
                "old_regime_filter_scale": float("nan"),
                "applied_market_overlay_scale": float("nan"),
                "applied_market_overlay_active": False,
                "applied_market_overlay_reason": "",
                "final_turnover": float("nan"),
            }

            market_regime_snapshot = self.market_regime_engine.compute(
                self.data_handler.get_history(date, 300)
            )
            market_overlay_decision = decide_event_driven_overlay(market_regime_snapshot)

            feature_market_state = ""
            aligned_market_regime = ""
            align_reason = ""
            if market_regime_snapshot is not None:
                feature_market_state = str(market_regime_snapshot.state.value)
                aligned_market_regime, align_reason = align_market_regime_with_risk(
                    feature_market_state, risk_regime_name
                )
                if feature_market_state != self.last_market_regime_state or (
                    aligned_market_regime != self.last_aligned_market_regime
                ):
                    logger.info(
                        f"  Market regime {date.date()} | eff={aligned_market_regime} "
                        f"(model={feature_market_state}) | align={align_reason} | "
                        f"risk={risk_regime_name} | "
                        f"Score: {market_regime_snapshot.composite_score:.2f} | "
                        f"Conf: {market_regime_snapshot.confidence:.2f} | "
                        f"Expo: {market_regime_snapshot.exposure_multiplier:.2f}x"
                    )
                    self.last_market_regime_state = feature_market_state
                    self.last_aligned_market_regime = aligned_market_regime
            else:
                aligned_market_regime, align_reason = align_market_regime_with_risk("", risk_regime_name)

            self._day_regime_ctx = {
                "market_regime_feature": feature_market_state,
                "market_regime_effective": aligned_market_regime or feature_market_state,
                "market_regime_align_reason": align_reason,
                "risk_regime_name": risk_regime_name,
            }

            eff_for_df = self._day_regime_ctx["market_regime_effective"]
            dfr = self.defensive_flat_ctrl.step(
                date,
                eff_for_df,
                risk_regime_name,
                float(getattr(risk_snapshot, "current_drawdown", 0.0)),
                bool(risk_snapshot.trading_suspended),
            )
            self._day_regime_ctx["defensive_flat_phase"] = dfr.phase.value
            self._day_regime_ctx["defensive_flat_reason"] = dfr.reason
            if dfr.entered_today:
                logger.info(
                    f"  Defensive flat -> ENTER {date.date()} | eff={eff_for_df} | "
                    f"DD={float(getattr(risk_snapshot, 'current_drawdown', 0.0)):.1%}"
                )
            if dfr.exited_today:
                logger.info(f"  Defensive flat -> EXIT {date.date()} | reason={dfr.reason}")

            if dfr.should_hold_flat and not risk_snapshot.trading_suspended:
                for symbol, qty in list(self.portfolio.positions.items()):
                    if qty != 0:
                        p = self.portfolio.prices.get(symbol, 0)
                        if p > 0:
                            direction = 1 if -qty > 0 else -1
                            fill_price = p * (1 + direction * self.broker.slippage_rate * 2)
                            commission = abs(qty) * fill_price * self.broker.commission_rate
                            fill = FillEvent(
                                date=date,
                                ticker=symbol,
                                quantity=-qty,
                                fill_price=fill_price,
                                commission=commission,
                            )
                            self.portfolio.fill_order(fill)
                            self.n_trades += 1

            for symbol in risk_snapshot.positions_to_close:
                if symbol in self.portfolio.positions:
                    p = self.portfolio.prices.get(symbol, 0)
                    if p > 0:
                        self.broker.submit_order(
                            OrderEvent(
                                date=date,
                                ticker=symbol,
                                quantity=-self.portfolio.positions[symbol],
                                price=p,
                            )
                        )

            rebal_today = self.last_rebal_date is None or (
                (date.year * 12 + date.month) > (self.last_rebal_date.year * 12 + self.last_rebal_date.month)
            )
            if self._rebalance_immediately_after_reentry:
                rebal_today = True

            if rebal_today and not risk_snapshot.trading_suspended:
                if self._rebalance_immediately_after_reentry:
                    self._rebalance_immediately_after_reentry = False
                self.last_rebal_date = date

                eff_regime = aligned_market_regime or feature_market_state
                if self.defensive_flat_ctrl.is_flat():
                    weights = {}
                    signal_diagnostics = {
                        "signal_reason": "DEFENSIVE_FLAT",
                        "rebal_threshold": float("nan"),
                        "rebal_threshold_context": "",
                        "gross_signal_raw": 0.0,
                        "gross_after_constraints": 0.0,
                        "gross_after_risk_manager": 0.0,
                        "gross_after_rebal_threshold": 0.0,
                    }
                    old_regime_meta = {
                        "applied_scale": 0.0,
                        "trend_tilt_mult": 1.0,
                        "regime_name": risk_regime_name,
                        "regime_score": float(risk_snapshot.regime_score),
                        "hard_flat": True,
                    }
                    informed_meta = {"informed_tilt_scale": 1.0, "informed_tilt_reason": "defensive_flat"}
                    gross_after_old_regime = 0.0
                    gross_after_informed = 0.0
                    weights = apply_market_regime_overlay_helper(weights, market_overlay_decision)
                    gross_after_overlay = 0.0
                else:
                    weights = self.signal_gen.compute_weights(
                        date,
                        risk_snapshot,
                        market_regime_state=eff_regime,
                    )
                    signal_diagnostics = dict(getattr(self.signal_gen, "last_diagnostics", {}) or {})
                    weights, old_regime_meta = apply_regime_weight_filter_helper(
                        weights,
                        risk_snapshot,
                        return_meta=True,
                        aligned_market_regime=eff_regime,
                    )
                    gross_after_old_regime = sum(abs(w) for w in weights.values())
                    weights, informed_meta = apply_risk_informed_exposure_tilt_helper(
                        weights,
                        eff_regime,
                        risk_regime_name,
                        return_meta=True,
                    )
                    gross_after_informed = sum(abs(w) for w in weights.values())
                    weights = apply_market_regime_overlay_helper(weights, market_overlay_decision)
                    gross_after_overlay = sum(abs(w) for w in weights.values())
                if market_overlay_decision.active and gross_after_informed > 0:
                    logger.info(
                        f"  Overlay marche {date.date()} | {market_overlay_decision.state} | "
                        f"{market_overlay_decision.reason} | "
                        f"Gross: {gross_after_informed:.2f}x -> {gross_after_overlay:.2f}x"
                    )

                signal_event = SignalEvent(
                    date=date,
                    weights=weights,
                    regime=risk_snapshot.regime_score,
                    regime_state=getattr(getattr(risk_snapshot, "regime", None), "name", "UNKNOWN"),
                    regime_confidence=getattr(risk_snapshot, "confidence", 0.0),
                    signal=Signal.FLAT if not weights else Signal.HOLD,
                )

                orders = self.portfolio.generate_orders(signal_event, prices)

                pv = self.portfolio.portfolio_value
                if pv > 0 and orders:
                    turnover = sum(abs(o.quantity * o.price) for o in orders) / pv

                daily_rebal_diagnostics = {
                    "rebalancing_day": True,
                    "n_orders": int(len(orders)),
                    "risk_scaling": float(getattr(risk_snapshot, "risk_scaling", float("nan"))),
                    "rebal_threshold": float(signal_diagnostics.get("rebal_threshold", float("nan"))),
                    "rebal_threshold_context": str(signal_diagnostics.get("rebal_threshold_context", "")),
                    "signal_generation_reason": str(signal_diagnostics.get("signal_reason", "")),
                    "gross_signal_raw": float(signal_diagnostics.get("gross_signal_raw", float("nan"))),
                    "gross_after_constraints": float(
                        signal_diagnostics.get("gross_after_constraints", float("nan"))
                    ),
                    "gross_after_risk_manager": float(
                        signal_diagnostics.get("gross_after_risk_manager", float("nan"))
                    ),
                    "gross_after_rebal_threshold": float(
                        signal_diagnostics.get("gross_after_rebal_threshold", float("nan"))
                    ),
                    "gross_after_old_regime_filter": float(gross_after_old_regime),
                    "gross_after_informed_tilt": float(gross_after_informed),
                    "gross_after_market_overlay": float(gross_after_overlay),
                    "old_regime_filter_scale": float(old_regime_meta.get("applied_scale", float("nan"))),
                    "trend_tilt_mult": float(old_regime_meta.get("trend_tilt_mult", float("nan"))),
                    "informed_tilt_scale": float(informed_meta.get("informed_tilt_scale", float("nan"))),
                    "informed_tilt_reason": str(informed_meta.get("informed_tilt_reason", "")),
                    "applied_market_overlay_scale": float(
                        getattr(market_overlay_decision, "scale", float("nan"))
                    ),
                    "applied_market_overlay_active": bool(
                        getattr(market_overlay_decision, "active", False)
                    ),
                    "applied_market_overlay_reason": str(getattr(market_overlay_decision, "reason", "")),
                    "final_turnover": float(turnover),
                }
                self.rebal_diagnostics.append(
                    {
                        "date": date,
                        "risk_regime_state": getattr(
                            getattr(risk_snapshot, "regime", None), "name", "UNKNOWN"
                        ),
                        "risk_regime_score": float(getattr(risk_snapshot, "regime_score", float("nan"))),
                        "risk_scaling": float(getattr(risk_snapshot, "risk_scaling", float("nan"))),
                        "rebal_threshold": float(signal_diagnostics.get("rebal_threshold", float("nan"))),
                        "rebal_threshold_context": str(
                            signal_diagnostics.get("rebal_threshold_context", "")
                        ),
                        "old_regime_filter_scale": float(old_regime_meta.get("applied_scale", float("nan"))),
                        "market_regime_state": feature_market_state,
                        "market_regime_effective": eff_regime,
                        "market_regime_score": float(
                            getattr(market_regime_snapshot, "composite_score", float("nan"))
                        ),
                        "market_regime_confidence": float(
                            getattr(market_regime_snapshot, "confidence", float("nan"))
                        ),
                        "market_overlay_scale": float(getattr(market_overlay_decision, "scale", float("nan"))),
                        "market_overlay_active": bool(getattr(market_overlay_decision, "active", False)),
                        "market_overlay_reason": str(getattr(market_overlay_decision, "reason", "")),
                        "signal_generation_reason": str(signal_diagnostics.get("signal_reason", "")),
                        "gross_signal_raw": float(signal_diagnostics.get("gross_signal_raw", float("nan"))),
                        "gross_after_constraints": float(
                            signal_diagnostics.get("gross_after_constraints", float("nan"))
                        ),
                        "gross_after_risk_manager": float(
                            signal_diagnostics.get("gross_after_risk_manager", float("nan"))
                        ),
                        "gross_after_rebal_threshold": float(
                            signal_diagnostics.get("gross_after_rebal_threshold", float("nan"))
                        ),
                        "gross_after_old_regime_filter": float(gross_after_old_regime),
                        "gross_after_informed_tilt": float(gross_after_informed),
                        "gross_after_market_overlay": float(gross_after_overlay),
                        "market_regime_effective": eff_regime,
                        "market_regime_align": str(self._day_regime_ctx.get("market_regime_align_reason", "")),
                        "informed_tilt_scale": float(informed_meta.get("informed_tilt_scale", float("nan"))),
                        "n_orders": int(len(orders)),
                        "turnover": float(turnover),
                    }
                )
                logger.info(
                    f"  Rebal diag {date.date()} | "
                    f"Raw: {daily_rebal_diagnostics['gross_signal_raw']:.2f}x | "
                    f"Risk: {daily_rebal_diagnostics['gross_after_risk_manager']:.2f}x | "
                    f"Thr: {daily_rebal_diagnostics['rebal_threshold']:.3f} | "
                    f"Old regime: {daily_rebal_diagnostics['gross_after_old_regime_filter']:.2f}x | "
                    f"Informed: {daily_rebal_diagnostics['gross_after_informed_tilt']:.2f}x | "
                    f"Market: {daily_rebal_diagnostics['gross_after_market_overlay']:.2f}x | "
                    f"Orders: {len(orders)} | TO: {turnover:.1%}"
                )

                for order in orders:
                    self.broker.submit_order(order)

                if REBALANCE_FILL_SAME_BAR and orders:
                    for fill in self.broker.execute_pending(prices):
                        self.portfolio.fill_order(fill)
                        self.n_trades += 1
                    rs_rebal = self.risk_manager.apply_post_rebalance_recut_check(
                        date,
                        self.portfolio.portfolio_value,
                        self.portfolio.positions,
                    )
                    if rs_rebal is not None:
                        risk_snapshot = rs_rebal

            self.prev_prices = prices

            if risk_snapshot.trading_suspended:
                for symbol, qty in list(self.portfolio.positions.items()):
                    if qty != 0:
                        p = self.portfolio.prices.get(symbol, 0)
                        if p > 0:
                            direction = 1 if -qty > 0 else -1
                            fill_price = p * (1 + direction * self.broker.slippage_rate * 2)
                            commission = abs(qty) * fill_price * self.broker.commission_rate
                            fill = FillEvent(
                                date=date,
                                ticker=symbol,
                                quantity=-qty,
                                fill_price=fill_price,
                                commission=commission,
                            )
                            self.portfolio.fill_order(fill)
                            self.n_trades += 1

            merged_diag = {**daily_rebal_diagnostics, **self._day_regime_ctx}
            stats = self.portfolio.compute_stats(
                date=date,
                regime_score=regime_score,
                turnover=turnover,
                regime_state=getattr(getattr(risk_snapshot, "regime", None), "name", "UNKNOWN"),
                regime_confidence=getattr(risk_snapshot, "confidence", 0.0),
                trading_suspended=getattr(risk_snapshot, "trading_suspended", False),
                dd_max_stop=getattr(risk_snapshot, "dd_max_stop", False),
                suspension_reason=getattr(risk_snapshot, "suspension_reason", ""),
                suspended_days=getattr(risk_snapshot, "suspended_days", 0),
                diagnostics=merged_diag,
                market_regime_feature=self._day_regime_ctx.get("market_regime_feature", ""),
                market_regime_effective=self._day_regime_ctx.get("market_regime_effective", ""),
                market_regime_align_reason=self._day_regime_ctx.get("market_regime_align_reason", ""),
                risk_regime_name=self._day_regime_ctx.get("risk_regime_name", ""),
            )

            if self.per_day_callback is not None:
                self.per_day_callback(stats, self)

            if self.day_sleep_sec > 0:
                time.sleep(self.day_sleep_sec)

            if self.live_viz:
                self.visualizer.update(self.portfolio.history)

            day_count += 1

            if day_count % 252 == 0:
                pv = stats.portfolio_value
                cagr = (pv / self.portfolio.initial_capital) ** (252 / day_count) - 1
                logger.info(
                    f"  {date.date()} | PV: ${pv:,.0f} | CAGR: {cagr:.1%} | "
                    f"Vol: {stats.realized_vol:.1%} | DD: {stats.drawdown:.1%} | "
                    f"Régime: {stats.regime_score:.2f}"
                )

        logger.info(f"\n  Backtest terminé — {day_count} jours | {self.n_trades} trades")
        return self._compute_final_metrics()

    def _compute_final_metrics(self) -> dict:
        history = self.portfolio.history
        if not history:
            return {}

        returns = np.array([s.daily_return for s in history])
        pv = np.array([s.portfolio_value for s in history])
        n_years = len(returns) / 252
        rf_daily = (1 + RISK_FREE_RATE) ** (1 / 252) - 1

        cagr = (pv[-1] / self.portfolio.initial_capital) ** (1 / n_years) - 1
        sharpe = (returns.mean() - rf_daily) / (returns.std() + 1e-8) * np.sqrt(252)
        max_dd = float(np.array([s.drawdown for s in history]).min())
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        avg_to = np.mean([s.turnover for s in history]) * 252

        logger.info("\n" + "=" * 60)
        logger.info("  RÉSULTATS FINAUX — EVENT-DRIVEN")
        logger.info("=" * 60)
        logger.info(f"  CAGR         : {cagr:.2%}")
        logger.info(f"  Sharpe       : {sharpe:.3f}")
        logger.info(f"  Max DD       : {max_dd:.2%}")
        logger.info(f"  Calmar       : {calmar:.3f}")
        logger.info(f"  Valeur finale: ${pv[-1]:,.0f}")
        logger.info(f"  Nb trades    : {self.n_trades}")
        logger.info(f"  Turnover/an  : {avg_to:.1%}")

        self.final_metrics = {
            "cagr": cagr,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "calmar": calmar,
            "final_value": pv[-1],
            "n_trades": self.n_trades,
            "avg_turnover": avg_to,
        }
        return self.final_metrics

    def save_results(self) -> dict:
        history = self.portfolio.history
        if not history:
            return {}

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_df = pd.DataFrame(
            [
                {
                    "date": s.date,
                    "portfolio_value": s.portfolio_value,
                    "daily_return": s.daily_return,
                    "realized_vol": s.realized_vol,
                    "expected_vol": s.expected_vol,
                    "drawdown": s.drawdown,
                    "regime_score": s.regime_score,
                    "regime_state": s.regime_state,
                    "regime_confidence": s.regime_confidence,
                    "trading_suspended": s.trading_suspended,
                    "dd_max_stop": s.dd_max_stop,
                    "suspension_reason": s.suspension_reason,
                    "suspended_days": s.suspended_days,
                    "turnover": s.turnover,
                    "rebalancing_day": s.rebalancing_day,
                    "n_orders": s.n_orders,
                    "risk_scaling": s.risk_scaling,
                    "rebal_threshold": s.rebal_threshold,
                    "rebal_threshold_context": s.rebal_threshold_context,
                    "signal_generation_reason": s.signal_generation_reason,
                    "gross_signal_raw": s.gross_signal_raw,
                    "gross_after_constraints": s.gross_after_constraints,
                    "gross_after_risk_manager": s.gross_after_risk_manager,
                    "gross_after_rebal_threshold": s.gross_after_rebal_threshold,
                    "gross_after_old_regime_filter": s.gross_after_old_regime_filter,
                    "gross_after_market_overlay": s.gross_after_market_overlay,
                    "old_regime_filter_scale": s.old_regime_filter_scale,
                    "applied_market_overlay_scale": s.applied_market_overlay_scale,
                    "applied_market_overlay_active": s.applied_market_overlay_active,
                    "applied_market_overlay_reason": s.applied_market_overlay_reason,
                    "final_turnover": s.final_turnover,
                    "market_regime_feature": getattr(s, "market_regime_feature", ""),
                    "market_regime_effective": getattr(s, "market_regime_effective", ""),
                    "market_regime_align_reason": getattr(s, "market_regime_align_reason", ""),
                    "risk_regime_name": getattr(s, "risk_regime_name", ""),
                    "informed_tilt_scale": getattr(s, "informed_tilt_scale", float("nan")),
                    "informed_tilt_reason": getattr(s, "informed_tilt_reason", ""),
                    "trend_tilt_mult": getattr(s, "trend_tilt_mult", float("nan")),
                    "defensive_flat_phase": getattr(s, "defensive_flat_phase", ""),
                    "defensive_flat_reason": getattr(s, "defensive_flat_reason", ""),
                }
                for s in history
            ]
        )
        stats_df["date"] = pd.to_datetime(stats_df["date"])

        market_regime_df = self.market_regime_engine.history_frame()
        if not market_regime_df.empty:
            market_regime_df["date"] = pd.to_datetime(market_regime_df["date"])
            rename_map = {
                c: f"{c}_model" for c in market_regime_df.columns if c != "date"
            }
            stats_df = stats_df.merge(
                market_regime_df.rename(columns=rename_map),
                on="date",
                how="left",
            )
            mod = "market_regime_state_model"
            if mod in stats_df.columns:
                eff = stats_df["market_regime_effective"].fillna("").astype(str)
                stats_df["market_regime_state"] = eff.where(eff.str.len() > 0, stats_df[mod])
            elif "market_regime_effective" in stats_df.columns:
                stats_df["market_regime_state"] = stats_df["market_regime_effective"]
            for src, dst in (
                ("market_regime_score_model", "market_regime_score"),
                ("market_regime_confidence_model", "market_regime_confidence"),
            ):
                if src in stats_df.columns:
                    stats_df[dst] = stats_df[src]
        else:
            if "market_regime_effective" in stats_df.columns:
                stats_df["market_regime_state"] = stats_df["market_regime_effective"]

        stats_path = self.output_dir / f"stats_{ts}.csv"
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"  Stats sauvegardées : {stats_path}")

        self.visualizer.update(history, force=True)
        html_path = self.visualizer.save(self.output_dir, suffix="_final")

        self._compare_with_vectorized(stats_df)

        files = {"stats": str(stats_path), "dashboard": html_path}

        if self.rebal_diagnostics:
            rebal_df = pd.DataFrame(self.rebal_diagnostics)
            rebal_path = self.output_dir / f"rebal_diagnostics_{ts}.csv"
            rebal_df.to_csv(rebal_path, index=False)
            logger.info(f"  Rebal diagnostics sauvegardes : {rebal_path}")
            files["rebal_diagnostics"] = str(rebal_path)

        regime_log_df = build_regime_log_frame(stats_df)
        if not regime_log_df.empty:
            regime_log_path = self.output_dir / f"regimes_{ts}.csv"
            regime_log_df.to_csv(regime_log_path, index=False)
            logger.info(f"  Regimes sauvegardes : {regime_log_path}")
            files["regimes"] = str(regime_log_path)

        model_regime_col = None
        if "market_regime_state_model" in stats_df.columns:
            model_regime_col = "market_regime_state_model"
        elif "market_regime_feature" in stats_df.columns:
            model_regime_col = "market_regime_feature"

        effective_regime_col = (
            "market_regime_effective"
            if "market_regime_effective" in stats_df.columns
            else "market_regime_state"
        )

        def _log_regime_perf_block(perf_df: pd.DataFrame, title: str) -> None:
            logger.info("\n" + "=" * 60)
            logger.info(f"  {title}")
            logger.info("=" * 60)
            for _, row in perf_df.iterrows():
                sharpe_text = f"{row['sharpe']:.2f}" if pd.notna(row["sharpe"]) else "n/a"
                return_text = (
                    f"{row['annualized_return']:.2%}"
                    if pd.notna(row["annualized_return"])
                    else "n/a"
                )
                dd_text = f"{row['max_drawdown']:.2%}" if pd.notna(row["max_drawdown"]) else "n/a"
                mdd = row.get("mean_portfolio_drawdown")
                wdd = row.get("worst_portfolio_drawdown")
                mdd_text = f"{float(mdd):.2%}" if mdd is not None and pd.notna(mdd) else "n/a"
                wdd_text = f"{float(wdd):.2%}" if wdd is not None and pd.notna(wdd) else "n/a"
                n_dd = int(row.get("n_dd_episodes_gt_10d", 0) or 0)
                n_gr = int(row.get("n_growth_episodes_gt_5d", 0) or 0)
                logger.info(
                    f"  {row['regime_state']:10s} | {int(row['days']):4d}j | "
                    f"Sharpe: {sharpe_text:>6s} | "
                    f"AnnRet: {return_text:>8s} | "
                    f"MaxDD(subret): {dd_text:>8s} | "
                    f"Turnover/an: {row['avg_turnover'] * 252:>7.1%}"
                )
                logger.info(
                    f"    -> DD jour (moy/min): {mdd_text} / {wdd_text} | "
                    f"ep. sous le pic >10j: {n_dd} | ep. jours verts >5j: {n_gr}"
                )

        if model_regime_col:
            perf_model_df = summarize_regime_performance(
                stats_df,
                regime_col=model_regime_col,
                risk_free_rate=RISK_FREE_RATE,
            )
            if not perf_model_df.empty:
                path_m = self.output_dir / f"regime_performance_model_{ts}.csv"
                perf_model_df.to_csv(path_m, index=False)
                logger.info(f"  Performance par regime (MODELE features) : {path_m}")
                _log_regime_perf_block(perf_model_df, "PERFORMANCE PAR REGIME — MODELE (features, avant align risk)")
                files["regime_performance_model"] = str(path_m)

        perf_effective_df = summarize_regime_performance(
            stats_df,
            regime_col=effective_regime_col,
            risk_free_rate=RISK_FREE_RATE,
        )
        if not perf_effective_df.empty:
            path_e = self.output_dir / f"regime_performance_effective_{ts}.csv"
            perf_effective_df.to_csv(path_e, index=False)
            logger.info(f"  Performance par regime (EFFECTIF aligne) : {path_e}")
            _log_regime_perf_block(
                perf_effective_df,
                "PERFORMANCE PAR REGIME — EFFECTIF (aligne risk / nouveau systeme)",
            )
            files["regime_performance_effective"] = str(path_e)

        if not perf_effective_df.empty:
            path_legacy = self.output_dir / f"regime_performance_{ts}.csv"
            perf_effective_df.to_csv(path_legacy, index=False)
            files["regime_performance"] = str(path_legacy)

        baseline_comparison = None
        if (
            not self._skip_baseline_comparison
            and self._baseline_reference_path is not None
            and self._baseline_reference_path.exists()
        ):
            baseline_comparison = compare_with_baseline_reference(
                self.final_metrics,
                baseline_path=self._baseline_reference_path,
            )
        if baseline_comparison is not None:
            baseline_path = self.output_dir / f"baseline_comparison_{ts}.json"
            with baseline_path.open("w", encoding="utf-8") as handle:
                json.dump(baseline_comparison, handle, indent=2)
            logger.info(f"  Comparaison baseline : {baseline_path}")
            files["baseline_comparison"] = str(baseline_path)

        if not self._skip_strategy_benchmark_report:
            try:
                from .strategy_benchmark_compare import build_report_html

                bh_html = self.output_dir / f"strategy_vs_benchmark_{ts}.html"
                build_report_html(
                    stats_path,
                    self._data_path,
                    bh_html,
                    initial_capital=float(self.portfolio.initial_capital),
                )
                files["strategy_vs_benchmark"] = str(bh_html)
                logger.info(f"  Rapport strategie vs benchmark EW : {bh_html}")
            except Exception as exc:
                logger.warning("  Rapport strategie vs benchmark EW omis : %s", exc)

        return files

    def _compare_with_vectorized(self, stats_df: pd.DataFrame):
        val_dir = Path("./results/validation")
        if not val_dir.exists():
            return
        ret_files = sorted(val_dir.glob("returns_*.csv"))
        if not ret_files:
            return

        vec_ret = pd.read_csv(ret_files[-1], index_col=0, parse_dates=True).iloc[:, 0].dropna()
        ed_ret = pd.Series(
            stats_df["daily_return"].values,
            index=pd.to_datetime(stats_df["date"]),
        ).dropna()
        rf_d = (1 + RISK_FREE_RATE) ** (1 / 252) - 1

        vec_sharpe = (vec_ret.mean() - rf_d) / vec_ret.std() * np.sqrt(252)
        ed_sharpe = (ed_ret.mean() - rf_d) / ed_ret.std() * np.sqrt(252)

        vec_total_return = float((1.0 + vec_ret).prod() - 1.0)
        ed_total_return = float((1.0 + ed_ret).prod() - 1.0)
        vec_cagr = (1.0 + vec_total_return) ** (252 / len(vec_ret)) - 1.0 if len(vec_ret) else 0.0
        ed_cagr = (1.0 + ed_total_return) ** (252 / len(ed_ret)) - 1.0 if len(ed_ret) else 0.0

        logger.info("\n" + "=" * 60)
        logger.info("  COMPARAISON VECTORISÉ vs EVENT-DRIVEN")
        logger.info("=" * 60)
        logger.info(f"  {'Métrique':15s} {'Vectorisé':>12s} {'Event-Driven':>14s} {'Ratio':>8s}")
        logger.info(f"  {'-'*55}")
        logger.info(f"  {'CAGR':15s} {vec_cagr:>11.2%} {ed_cagr:>13.2%} {ed_cagr/vec_cagr:>7.2f}x")
        logger.info(f"  {'Sharpe':15s} {vec_sharpe:>12.3f} {ed_sharpe:>14.3f} {ed_sharpe/vec_sharpe:>7.2f}x")
        logger.info("  → Ratio proche de 1.0 = pas de biais look-ahead ✅")

    @staticmethod
    def _evaluate_baseline_verdict(baseline_metrics: dict, current_metrics: dict, delta: dict) -> dict:
        return evaluate_baseline_verdict(baseline_metrics, current_metrics, delta)
