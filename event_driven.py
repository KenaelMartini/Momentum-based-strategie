# ============================================================
# event_driven.py — Backtest Event-Driven Phase 3
# ============================================================
import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
from enum import Enum

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    INITIAL_CAPITAL, RISK_FREE_RATE,
    MOMENTUM_WEIGHTS, SKIP_DAYS,
    LONG_QUANTILE, SHORT_QUANTILE,
    TRANSACTION_COST_BPS, SLIPPAGE_BPS,
)

# Conversion bps → taux décimal (1 bps = 0.0001)
COMMISSION_RATE = TRANSACTION_COST_BPS / 10_000   # 0.0010 = 0.10%
SLIPPAGE_RATE   = SLIPPAGE_BPS / 10_000           # 0.0005 = 0.05%


# ============================================================
# ÉNUMÉRATIONS & DATACLASSES
# ============================================================

class Signal(Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    FLAT  = "FLAT"
    HOLD  = "HOLD"

class OrderType(Enum):
    MARKET = "MARKET"

class EventType(Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER  = "ORDER"
    FILL   = "FILL"
    STATS  = "STATS"

@dataclass
class MarketEvent:
    date      : pd.Timestamp
    prices    : pd.Series
    event_type: EventType = EventType.MARKET

@dataclass
class SignalEvent:
    date      : pd.Timestamp
    weights   : dict
    regime    : float
    signal    : Signal
    event_type: EventType = EventType.SIGNAL

@dataclass
class OrderEvent:
    date      : pd.Timestamp
    ticker    : str
    quantity  : float
    price     : float
    order_type: OrderType = OrderType.MARKET
    event_type: EventType = EventType.ORDER

@dataclass
class FillEvent:
    date      : pd.Timestamp
    ticker    : str
    quantity  : float
    fill_price: float
    commission: float
    event_type: EventType = EventType.FILL

@dataclass
class PortfolioStats:
    date           : pd.Timestamp
    portfolio_value: float
    cash           : float
    positions_value: float
    daily_return   : float
    realized_vol   : float
    expected_vol   : float
    drawdown       : float
    regime_score   : float
    positions      : dict
    turnover       : float


# ============================================================
# DATA HANDLER
# ============================================================

class DataHandler:
    """
    Distribue les données UNE DATE À LA FOIS.
    get_history(date, window) filtre strictement <= date
    → impossible de voir le futur par construction.
    """

    def __init__(self, data_path, start_date, end_date, min_history=252):
        self.data_path   = Path(data_path)
        self.start_date  = pd.Timestamp(start_date)
        self.end_date    = pd.Timestamp(end_date)
        self.min_history = min_history
        self.prices_full = None
        self.returns_full= None
        self.current_idx = 0
        self.dates       = None
        self._load_data()

    def _load_data(self):
        logger.info(f"Chargement des données : {self.data_path}")
        df = pd.read_csv(self.data_path, index_col=0, parse_dates=True).sort_index()
        mask = (df.index >= self.start_date - timedelta(days=365)) & \
               (df.index <= self.end_date)
        self.prices_full  = df[mask].copy()
        self.returns_full = np.log(self.prices_full / self.prices_full.shift(1))
        self.dates        = self.prices_full.index[self.prices_full.index >= self.start_date]
        self.current_idx  = 0
        logger.info(f"  {len(self.prices_full)} jours chargés | {len(self.prices_full.columns)} actifs")
        logger.info(f"  Période : {self.dates[0].date()} → {self.dates[-1].date()}")

    def get_next_bar(self) -> Optional[MarketEvent]:
        if self.current_idx >= len(self.dates):
            return None
        date   = self.dates[self.current_idx]
        prices = self.prices_full.loc[date]
        self.current_idx += 1
        return MarketEvent(date=date, prices=prices)

    def get_history(self, date: pd.Timestamp, window: int) -> pd.DataFrame:
        """Retourne les prix jusqu'à date — jamais au-delà."""
        history = self.prices_full[self.prices_full.index <= date]
        return history.iloc[-window:] if len(history) >= window else history

    @property
    def has_data(self) -> bool:
        return self.current_idx < len(self.dates)


# ============================================================
# SIGNAL GENERATOR
# ============================================================

class MomentumSignalGenerator:
    """
    Calcule le signal momentum sur les données disponibles à T.
    NE VOIT JAMAIS les données au-delà de T.

    Score_i = Σ w_k * log(P_{T-skip} / P_{T-skip-window_k})
    """

    def __init__(self, data_handler, momentum_weights=None,
                 skip_days=10, long_quantile=0.70, short_quantile=0.30):
        self.data       = data_handler
        self.weights    = momentum_weights or MOMENTUM_WEIGHTS
        self.skip_days  = skip_days
        self.long_q     = long_quantile
        self.short_q    = short_quantile
        self.max_window = max(self.weights.keys())

    def compute_signal(self, date):
        required = self.max_window + self.skip_days + 10
        prices   = self.data.get_history(date, required)
        if len(prices) < required:
            return None

        # Fenêtre séparée pour le régime — 300j minimum
        prices_regime = self.data.get_history(date, 300)
        regime_score, multiplier = self._compute_regime(prices_regime)

        # Score momentum : somme pondérée des rendements multi-fenêtres
        scores = pd.Series(0.0, index=prices.columns)
        for window, weight in self.weights.items():
            idx_end   = -self.skip_days if self.skip_days > 0 else len(prices)
            idx_start = -(window + self.skip_days)
            p_end     = prices.iloc[idx_end]
            p_start   = prices.iloc[idx_start]
            ret       = np.log(p_end / p_start).replace([np.inf, -np.inf], np.nan).fillna(0)
            scores   += weight * ret

        regime_score, multiplier = self._compute_regime(prices)

        if multiplier < 0.05:
            return SignalEvent(date=date, weights={}, regime=regime_score, signal=Signal.FLAT)

        valid   = scores.dropna()
        q_high  = valid.quantile(self.long_q)
        q_low   = valid.quantile(self.short_q)
        longs   = valid[valid >= q_high].index.tolist()
        shorts  = valid[valid <= q_low].index.tolist()

        weights_dict = {}

        # Poids borné à 10% max par position (diversification)
        base_weight = min(multiplier / max(len(longs), 1), 0.10)

        for t in longs:
            weights_dict[t] = base_weight
        for t in shorts:
            weights_dict[t] = -base_weight * 0.5

        return SignalEvent(date=date, weights=weights_dict,
                          regime=regime_score, signal=Signal.LONG)

    def _compute_regime(self, prices: pd.DataFrame):
        scores = []
        if len(prices) >= 200:
            ma200 = prices.iloc[-200:].mean()
            scores.append((prices.iloc[-1] > ma200).mean())
        if len(prices) >= 63:
            rets = np.log(prices / prices.shift(1)).fillna(0)
            vol_10j = rets.iloc[-10:].std().mean()
            vol_63j = rets.iloc[-63:].std().mean()
            vol_ratio = vol_10j / (vol_63j + 1e-8)
            if   vol_ratio >= 2.0:  vol_score = 0.0
            elif vol_ratio >= 1.5:  vol_score = 0.25
            elif vol_ratio >= 1.25: vol_score = float(np.interp(vol_ratio, [1.25, 1.5], [1.0, 0.25]))
            else:                   vol_score = 1.0
            scores.append(vol_score)
            corr = rets.iloc[-42:].corr()
            n = len(corr)
            avg_corr = (corr.sum().sum() - n) / (n * (n-1) + 1e-8)
            if   avg_corr > 0.45: corr_score = 0.10
            elif avg_corr > 0.35: corr_score = float(np.interp(avg_corr, [0.35, 0.45], [1.0, 0.10]))
            else:                 corr_score = 1.0
            scores.append(corr_score)
        if not scores:
            return 0.5, 0.5
        s = float(min(scores))

        # LOG TEMPORAIRE
        logger.info(f"  REGIME: trend={scores[0]:.2f} vol={scores[1]:.2f} corr={scores[2]:.2f} avg_corr={avg_corr:.3f} s={s:.2f}")

        if   s >= 0.70: m = 1.0
        elif s >= 0.50: m = 0.5 + (s - 0.50) / 0.20 * 0.5
        elif s >= 0.30: m = 0.25 + (s - 0.30) / 0.20 * 0.25
        elif s >= 0.10: m = 0.25
        else:           m = 0.0
        return s, m


# ============================================================
# PORTFOLIO
# ============================================================

class Portfolio:
    """
    Gère positions + cash.
    Ordres soumis à T, exécutés à T+1 (réaliste).
    """

    def __init__(self, initial_capital=INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash            = initial_capital
        self.positions       = {}
        self.prices          = {}
        self.history         = []
        self.returns_history = []
        self.peak_value      = initial_capital
        self.prev_value      = initial_capital

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

    def generate_orders(self, signal: SignalEvent, prices: pd.Series) -> list:
        orders   = []
        pv       = self.portfolio_value
        target_w = signal.weights

        # Fermer les positions qui ne sont plus dans le signal
        for t in list(self.positions.keys()):
            if t not in target_w and self.positions[t] != 0:
                p = self.prices.get(t, 0.0)
                if p > 0:
                    orders.append(OrderEvent(date=signal.date, ticker=t,
                                             quantity=-self.positions[t], price=p))

        # Ajuster les positions cibles
        REBAL_THRESHOLD = 0.015
        for t, w in target_w.items():
            p = self.prices.get(t, 0.0)
            if p <= 0:
                continue
            current_w = self.positions.get(t, 0) * p / pv if pv > 0 else 0
            delta_w   = w - current_w
            if abs(delta_w) < REBAL_THRESHOLD:
                continue
            delta_val = delta_w * pv
            orders.append(OrderEvent(date=signal.date, ticker=t,
                                     quantity=delta_val / p, price=p))
        return orders


    def fill_order(self, fill: FillEvent):
        t    = fill.ticker
        qty  = fill.quantity
        cost = qty * fill.fill_price + fill.commission
        self.positions[t] = self.positions.get(t, 0) + qty
        if abs(self.positions[t]) < 0.01:
            del self.positions[t]
        self.cash -= cost

    def compute_stats(self, date, regime_score, turnover) -> PortfolioStats:
        pv        = self.portfolio_value
        daily_ret = (pv / self.prev_value - 1) if self.prev_value > 0 else 0.0
        self.returns_history.append(daily_ret)
        self.prev_value = pv
        self.peak_value = max(self.peak_value, pv)
        drawdown        = (pv - self.peak_value) / self.peak_value

        # Volatilité réalisée rolling 21j (annualisée)
        recent = self.returns_history[-21:]
        rv = float(np.std(recent) * np.sqrt(252)) if len(recent) >= 5 else 0.0

        # Proxy GARCH : EV = 0.94 * vol_long + 0.06 * vol_court
        long_r = self.returns_history[-63:]
        lv = float(np.std(long_r) * np.sqrt(252)) if len(long_r) >= 21 else rv
        ev = 0.94 * lv + 0.06 * rv

        stats = PortfolioStats(
            date=date, portfolio_value=pv, cash=self.cash,
            positions_value=self.positions_value, daily_return=daily_ret,
            realized_vol=rv, expected_vol=ev, drawdown=drawdown,
            regime_score=regime_score, positions=dict(self.positions),
            turnover=turnover,
        )
        self.history.append(stats)
        return stats


# ============================================================
# BROKER SIMULÉ
# ============================================================

class SimulatedBroker:
    """
    Exécution réaliste :
    - Slippage : prix défavorable selon direction de l'ordre
    - Commission : % de la valeur du trade
    - Exécution à T+1 (ordres soumis à T, exécutés au bar suivant)
    """

    def __init__(self, commission_rate=COMMISSION_RATE, slippage_rate=SLIPPAGE_RATE):
        self.commission_rate = commission_rate
        self.slippage_rate   = slippage_rate
        self.pending_orders  = []

    def submit_order(self, order: OrderEvent):
        self.pending_orders.append(order)

    def execute_pending(self, current_prices: pd.Series) -> list:
        fills = []
        for order in self.pending_orders:
            t   = order.ticker
            qty = order.quantity
            if t not in current_prices.index:
                continue
            p = float(current_prices[t])
            if np.isnan(p) or p <= 0:
                continue
            direction  = 1 if qty > 0 else -1
            fill_price = p * (1 + direction * self.slippage_rate)
            commission = abs(qty) * fill_price * self.commission_rate
            fills.append(FillEvent(date=order.date, ticker=t, quantity=qty,
                                   fill_price=fill_price, commission=commission))
        self.pending_orders = []
        return fills


# ============================================================
# VISUALISEUR 3D
# ============================================================

class LiveVisualizer3D:
    """
    Dashboard 3D : surface P&L × Volatilité × Temps
    + 4 panels 2D (Vol réalisée/attendue, Drawdown, Régime, Equity)

    NOTE : add_hline/add_vline ne fonctionnent pas avec les subplots
    mixtes 3D+2D. On utilise add_shape + add_annotation à la place.
    Mapping des axes dans make_subplots(rows=3, cols=2) avec 3D en row=1 :
      row=2 col=1 → xref="x2", yref="y2"
      row=2 col=2 → xref="x3", yref="y3"
      row=3 col=1 → xref="x4", yref="y4"
      row=3 col=2 → xref="x5", yref="y5"
    """

    def __init__(self, update_every=20):
        self.update_every = update_every
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            self.go            = go
            self.make_subplots = make_subplots
            self.available     = True
        except ImportError:
            logger.warning("Plotly non disponible.")
            self.available = False
        self.fig = None

    def update(self, stats_list: list, force: bool = False):
        if not self.available:
            return
        n = len(stats_list)
        if not force and n % self.update_every != 0:
            return
        if n < 5:
            return
        self._render(stats_list)

    def _render(self, stats_list: list):
        go            = self.go
        make_subplots = self.make_subplots

        dates   = [s.date for s in stats_list]
        pv      = [s.portfolio_value for s in stats_list]
        rv      = [s.realized_vol for s in stats_list]
        ev      = [s.expected_vol for s in stats_list]
        dd      = [s.drawdown * 100 for s in stats_list]
        regime  = [s.regime_score for s in stats_list]
        pv_norm = [(v / stats_list[0].portfolio_value - 1) * 100 for v in pv]

        rv_arr = np.array(rv) * 100
        ev_arr = np.array(ev) * 100

        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{"type": "scene", "colspan": 2}, None],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
            ],
            subplot_titles=[
                "3D — P&L × Volatilité × Temps", "",
                "Realized Vol vs Expected Vol", "Drawdown",
                "Score de Régime", "Equity Curve",
            ],
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25],
        )

        # ── 3D scatter ─────────────────────────────────────
        fig.add_trace(go.Scatter3d(
            x=list(range(len(dates))), y=rv, z=pv_norm,
            mode="lines+markers",
            line=dict(color=pv_norm, colorscale="RdYlGn", width=4),
            marker=dict(
                size=3, opacity=0.8,
                color=regime,
                colorscale=[[0.0,"#f85149"],[0.3,"#e3b341"],
                            [0.7,"#58a6ff"],[1.0,"#3fb950"]],
            ),
            name="P&L",
            hovertemplate="Jour: %{x}<br>Vol: %{y:.1%}<br>P&L: %{z:.1f}%<extra></extra>",
        ), row=1, col=1)

        # ── Vol réalisée vs attendue ────────────────────────
        fig.add_trace(go.Scatter(
            x=dates, y=list(rv_arr), name="Vol réalisée",
            line=dict(color="#58a6ff", width=1.5),
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=list(ev_arr), name="Vol attendue",
            line=dict(color="#e3b341", width=1.5, dash="dot"),
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=list(np.maximum(rv_arr, ev_arr)) + list(np.minimum(rv_arr, ev_arr))[::-1],
            fill="toself", fillcolor="rgba(248,81,73,0.1)",
            line=dict(width=0), name="Zone stress", showlegend=False,
        ), row=2, col=1)

        # ── Drawdown ────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=dates, y=dd, name="Drawdown", fill="tozeroy",
            line=dict(color="#f85149", width=1),
            fillcolor="rgba(248,81,73,0.3)",
        ), row=2, col=2)

        # ── Score de régime ─────────────────────────────────
        fig.add_trace(go.Scatter(
            x=dates, y=regime, name="Régime",
            line=dict(width=0), fill="tozeroy",
            fillcolor="rgba(88,166,255,0.3)",
        ), row=3, col=1)

        # Seuils de régime via add_shape (add_hline incompatible 3D+2D)
        for y_val, color, label in [
            (0.7, "#3fb950", "BULL"),
            (0.5, "#58a6ff", "NORMAL"),
            (0.3, "#e3b341", "REDUIT"),
        ]:
            fig.add_shape(
                type="line", line=dict(dash="dot", color=color, width=1),
                x0=0, x1=1, xref="x4 domain",
                y0=y_val, y1=y_val, yref="y4",
            )
            fig.add_annotation(
                x=1, xref="x4 domain", xanchor="right",
                y=y_val, yref="y4",
                text=label, font=dict(color=color, size=9),
                showarrow=False, bgcolor="rgba(13,17,23,0.7)",
            )

        # ── Equity curve ────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=dates, y=pv_norm, name="P&L %",
            line=dict(color="#3fb950", width=2), fill="tozeroy",
            fillcolor="rgba(63,185,80,0.15)",
        ), row=3, col=2)

        # Ligne zéro via add_shape (add_hline incompatible 3D+2D)
        fig.add_shape(
            type="line", line=dict(color="#8b949e", width=0.8),
            x0=0, x1=1, xref="x5 domain",
            y0=0, y1=0, yref="y5",
        )

        # ── Layout ──────────────────────────────────────────
        n_days    = len(dates)
        last_pv   = pv_norm[-1] if pv_norm else 0
        last_rv   = rv[-1] * 100 if rv else 0
        last_reg  = regime[-1] if regime else 0
        reg_label = "BULL"   if last_reg > 0.7 else \
                    "NORMAL" if last_reg > 0.5 else \
                    "REDUIT" if last_reg > 0.3 else "CRISE"

        fig.update_layout(
            title=dict(
                text=(f"EVENT-DRIVEN BACKTEST — {n_days} jours | "
                      f"P&L: {last_pv:+.1f}% | Vol: {last_rv:.1f}% | Régime: {reg_label}"),
                font=dict(size=16, color="#58a6ff", family="monospace"),
                x=0.5, xanchor="center",
            ),
            paper_bgcolor="#0d1117",
            plot_bgcolor ="#161b22",
            font=dict(color="#c9d1d9", family="monospace"),
            height=1000,
            showlegend=True,
            legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor="#21262d"),
            scene=dict(
                xaxis=dict(title="Jours", backgroundcolor="#161b22", gridcolor="#21262d"),
                yaxis=dict(title="Vol réalisée (%)", backgroundcolor="#161b22", gridcolor="#21262d"),
                zaxis=dict(title="P&L (%)", backgroundcolor="#161b22", gridcolor="#21262d"),
                bgcolor="#0d1117",
                camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
            ),
        )

        for r in range(2, 4):
            for c in range(1, 3):
                try:
                    fig.update_xaxes(gridcolor="#21262d", row=r, col=c)
                    fig.update_yaxes(gridcolor="#21262d", row=r, col=c)
                except Exception:
                    pass

        self.fig = fig

    def save(self, output_dir: Path, suffix: str = "") -> str:
        if self.fig is None:
            return ""
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"event_driven_3d{suffix}_{ts}.html"
        self.fig.write_html(str(path), include_plotlyjs="cdn",
                            config={"displayModeBar": True, "scrollZoom": True})
        logger.info(f"  Dashboard 3D sauvegardé : {path}")
        return str(path)


# ============================================================
# MOTEUR PRINCIPAL
# ============================================================

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
    Cela évite le turnover excessif et les coûts de transaction destructeurs.
    """

    def __init__(self, data_path, start_date="2016-01-01", end_date="2024-12-31",
                 initial_capital=INITIAL_CAPITAL, live_viz=False,
                 output_dir="./results/event_driven"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.live_viz   = live_viz

        from event_driven_risk import EventDrivenRiskManager, MomentumSignalGeneratorV2

        self.data_handler = DataHandler(data_path, start_date, end_date)
        self.risk_manager = EventDrivenRiskManager(initial_capital)
        self.signal_gen   = MomentumSignalGeneratorV2(self.data_handler, self.risk_manager)
        self.portfolio    = Portfolio(initial_capital)
        self.broker       = SimulatedBroker()
        self.visualizer   = LiveVisualizer3D(update_every=21)
        self.n_trades     = 0
        self.last_rebal_date = None
        self.prev_prices     = None   # nécessaire pour les rendements cross-actifs

    def run(self) -> dict:
        logger.info("\n" + "="*60)
        logger.info("  BACKTEST EVENT-DRIVEN — PHASE 3")
        logger.info("="*60)

        day_count = 0

        while self.data_handler.has_data:

            # ── 1. Nouveau bar de marché ──────────────────────
            market_event = self.data_handler.get_next_bar()
            if market_event is None:
                break

            date   = market_event.date
            prices = market_event.prices

            self.portfolio.update_prices(prices)

            # ── 2. Exécution des ordres en attente (T-1 → T) ─
            fills = self.broker.execute_pending(prices)
            for fill in fills:
                self.portfolio.fill_order(fill)
                self.n_trades += 1

            # ── 3 + 4. Risk check + Signal mensuel ───────────
            signal_event = None
            turnover     = 0.0

            risk_snapshot = self.risk_manager.update(
                date              = date,
                prices            = prices,
                portfolio_value   = self.portfolio.portfolio_value,
                current_positions = self.portfolio.positions,
                entry_prices      = {t: self.portfolio.prices.get(t, 0)
                                     for t in self.portfolio.positions},
                prev_prices       = self.prev_prices,
            )
            self.signal_gen.update_ewma_vol(prices, self.prev_prices)
            regime_score = risk_snapshot.regime_score

            # Fermeture immédiate des stop-loss individuels
            for symbol in risk_snapshot.positions_to_close:
                if symbol in self.portfolio.positions:
                    p = self.portfolio.prices.get(symbol, 0)
                    if p > 0:
                        self.broker.submit_order(OrderEvent(
                            date=date, ticker=symbol,
                            quantity=-self.portfolio.positions[symbol], price=p
                        ))

            # Circuit breaker : liquider TOUTES les positions immédiatement
            if risk_snapshot.trading_suspended:
                for symbol, qty in list(self.portfolio.positions.items()):
                    if qty != 0:
                        p = self.portfolio.prices.get(symbol, 0)
                        if p > 0:
                            # Slippage majoré x2 pour simuler l'urgence de liquidation
                            direction  = 1 if -qty > 0 else -1
                            fill_price = p * (1 + direction * SLIPPAGE_RATE * 2)
                            commission = abs(qty) * fill_price * COMMISSION_RATE
                            fill = FillEvent(date=date, ticker=symbol, quantity=-qty,
                                            fill_price=fill_price, commission=commission)
                            self.portfolio.fill_order(fill)
                            self.n_trades += 1

            # Rebalancement mensuel (seulement si pas suspendu)
            rebal_today = (
                self.last_rebal_date is None or
                (date.year * 12 + date.month) >
                (self.last_rebal_date.year * 12 + self.last_rebal_date.month)
            )

            if rebal_today and not risk_snapshot.trading_suspended:
                self.last_rebal_date = date
                weights = self.signal_gen.compute_weights(date, risk_snapshot)
                signal_event = SignalEvent(
                    date=date, weights=weights, regime=regime_score,
                    signal=Signal.FLAT if not weights else Signal.HOLD
                )
                orders = self.portfolio.generate_orders(signal_event, prices)
                pv = self.portfolio.portfolio_value
                if pv > 0 and orders:
                    turnover = sum(abs(o.quantity * o.price) for o in orders) / pv
                for order in orders:
                    self.broker.submit_order(order)

            self.prev_prices = prices

            # ── 5. Statistiques du portefeuille ──────────────
            stats = self.portfolio.compute_stats(
                date=date, regime_score=regime_score, turnover=turnover)

            # ── 6. Visualisation live ─────────────────────────
            if self.live_viz:
                self.visualizer.update(self.portfolio.history)

            day_count += 1

            if day_count % 252 == 0:
                pv   = stats.portfolio_value
                cagr = (pv / self.portfolio.initial_capital) ** (252/day_count) - 1
                logger.info(
                    f"  {date.date()} | PV: ${pv:,.0f} | CAGR: {cagr:.1%} | "
                    f"Vol: {stats.realized_vol:.1%} | DD: {stats.drawdown:.1%} | "
                    f"Régime: {stats.regime_score:.2f}"
                )

        logger.info(f"\n  Backtest terminé — {day_count} jours | {self.n_trades} trades")
        return self._compute_final_metrics()

    def _compute_final_metrics(self) -> dict:
        history  = self.portfolio.history
        if not history:
            return {}

        returns  = np.array([s.daily_return for s in history])
        pv       = np.array([s.portfolio_value for s in history])
        n_years  = len(returns) / 252
        rf_daily = (1 + RISK_FREE_RATE) ** (1/252) - 1

        cagr   = (pv[-1] / self.portfolio.initial_capital) ** (1/n_years) - 1
        sharpe = (returns.mean() - rf_daily) / (returns.std() + 1e-8) * np.sqrt(252)
        max_dd = float(np.array([s.drawdown for s in history]).min())
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        avg_to = np.mean([s.turnover for s in history]) * 252

        logger.info("\n" + "="*60)
        logger.info("  RÉSULTATS FINAUX — EVENT-DRIVEN")
        logger.info("="*60)
        logger.info(f"  CAGR         : {cagr:.2%}")
        logger.info(f"  Sharpe       : {sharpe:.3f}")
        logger.info(f"  Max DD       : {max_dd:.2%}")
        logger.info(f"  Calmar       : {calmar:.3f}")
        logger.info(f"  Valeur finale: ${pv[-1]:,.0f}")
        logger.info(f"  Nb trades    : {self.n_trades}")
        logger.info(f"  Turnover/an  : {avg_to:.1%}")

        return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd,
                "calmar": calmar, "final_value": pv[-1],
                "n_trades": self.n_trades, "avg_turnover": avg_to}

    def save_results(self) -> dict:
        history = self.portfolio.history
        if not history:
            return {}

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        stats_df = pd.DataFrame([{
            "date": s.date, "portfolio_value": s.portfolio_value,
            "daily_return": s.daily_return, "realized_vol": s.realized_vol,
            "expected_vol": s.expected_vol, "drawdown": s.drawdown,
            "regime_score": s.regime_score, "turnover": s.turnover,
        } for s in history])

        stats_path = self.output_dir / f"stats_{ts}.csv"
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"  Stats sauvegardées : {stats_path}")

        self.visualizer.update(history, force=True)
        html_path = self.visualizer.save(self.output_dir, suffix="_final")

        self._compare_with_vectorized(stats_df)

        return {"stats": str(stats_path), "dashboard": html_path}

    def _compare_with_vectorized(self, stats_df: pd.DataFrame):
        """
        Compare event-driven vs vectorisé.
        Ratio proche de 1.0 = pas de biais look-ahead dans le vectorisé.
        """
        val_dir = Path("./results/validation")
        if not val_dir.exists():
            return
        ret_files = sorted(val_dir.glob("returns_*.csv"))
        if not ret_files:
            return

        vec_ret  = pd.read_csv(ret_files[-1], index_col=0, parse_dates=True).iloc[:, 0].dropna()
        ed_ret   = pd.Series(stats_df["daily_return"].values,
                             index=pd.to_datetime(stats_df["date"]))
        rf_d     = (1 + RISK_FREE_RATE) ** (1/252) - 1

        vec_sharpe = (vec_ret.mean() - rf_d) / vec_ret.std() * np.sqrt(252)
        ed_sharpe  = (ed_ret.mean() - rf_d) / ed_ret.std() * np.sqrt(252)
        vec_cagr   = (1 + vec_ret.mean()) ** 252 - 1
        ed_cagr    = (1 + ed_ret.mean()) ** 252 - 1

        logger.info("\n" + "="*60)
        logger.info("  COMPARAISON VECTORISÉ vs EVENT-DRIVEN")
        logger.info("="*60)
        logger.info(f"  {'Métrique':15s} {'Vectorisé':>12s} {'Event-Driven':>14s} {'Ratio':>8s}")
        logger.info(f"  {'-'*55}")
        logger.info(f"  {'CAGR':15s} {vec_cagr:>11.2%} {ed_cagr:>13.2%} {ed_cagr/vec_cagr:>7.2f}x")
        logger.info(f"  {'Sharpe':15s} {vec_sharpe:>12.3f} {ed_sharpe:>14.3f} {ed_sharpe/vec_sharpe:>7.2f}x")
        logger.info("  → Ratio proche de 1.0 = pas de biais look-ahead ✅")



# ============================================================
# SCRIPT PRINCIPAL
# ============================================================

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Backtest event-driven Phase 3")
    parser.add_argument("--start",  default="2016-01-01")
    parser.add_argument("--end",    default="2024-12-31")
    parser.add_argument("--live",   action="store_true")
    parser.add_argument("--data",   default="./data/processed/price_matrix.csv")
    parser.add_argument("--output", default="./results/event_driven")
    args = parser.parse_args()

    print("=" * 60)
    print("  PHASE 3 — BACKTEST EVENT-DRIVEN")
    print("=" * 60)
    print(f"  Période   : {args.start} → {args.end}")
    print(f"  Capital   : ${INITIAL_CAPITAL:,.0f}")
    print(f"  Visu live : {'OUI' if args.live else 'NON'}")
    print("=" * 60)

    engine = EventDrivenEngine(
        data_path=args.data, start_date=args.start, end_date=args.end,
        initial_capital=INITIAL_CAPITAL, live_viz=args.live, output_dir=args.output,
    )

    metrics = engine.run()

    print("\n  Génération de la visualisation 3D...")
    files = engine.save_results()

    print("\n" + "=" * 60)
    print("  FICHIERS GÉNÉRÉS")
    print("=" * 60)
    for key, path in files.items():
        if path:
            print(f"  {key:12s} : {path}")
    if files.get("dashboard"):
        print(f"\n  Ouvre dans ton navigateur :")
        print(f"  {files['dashboard']}")