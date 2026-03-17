# ============================================================
# event_driven_live.py — Backtest Event-Driven + Dashboard Live
# ============================================================
# USAGE :
#   python event_driven_live.py --speed 0.05
#   → ouvre http://localhost:8050 dans ton navigateur
# ============================================================

import sys
import os
import time
import threading
import logging
import warnings
from collections import deque

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
from event_driven import (
    DataHandler, MomentumSignalGenerator,
    Portfolio, SimulatedBroker, PortfolioStats,
    DailyRiskManager, SignalEvent, Signal,
)
from config import INITIAL_CAPITAL, RISK_FREE_RATE, MAX_PORTFOLIO_DRAWDOWN


# ============================================================
# ÉTAT PARTAGÉ
# ============================================================

class SharedState:
    def __init__(self):
        self.lock       = threading.Lock()
        self.stats      = []
        self.is_running = True
        self.is_done    = False
        self.n_trades   = 0

    def add_stats(self, stats):
        with self.lock:
            self.stats.append(stats)
            self.n_trades = len(self.stats)

    def get_stats_copy(self):
        with self.lock:
            return list(self.stats)

    def set_done(self):
        with self.lock:
            self.is_running = False
            self.is_done    = True


# ============================================================
# THREAD BACKTEST
# ============================================================

class BacktestThread(threading.Thread):
    def __init__(self, shared_state, data_path,
                 start_date="2016-01-01", end_date="2024-12-31",
                 initial_capital=INITIAL_CAPITAL, speed_factor=0.05):
        super().__init__(daemon=True)
        self.shared       = shared_state
        self.data_path    = data_path
        self.start_date   = start_date
        self.end_date     = end_date
        self.initial_cap  = initial_capital
        self.speed_factor = speed_factor

    def run(self):
        logger.info("  Thread backtest démarré...")
        data_handler    = DataHandler(self.data_path, self.start_date, self.end_date)
        signal_gen      = MomentumSignalGenerator(data_handler)
        portfolio       = Portfolio(self.initial_cap)
        broker          = SimulatedBroker()
        day_count       = 0
        n_trades        = 0
        last_rebal_date = None
        base_signal_weights = {}
        regime_buffer   = deque(maxlen=3)
        dd_max_triggered = False
        risk_manager    = DailyRiskManager()

        while data_handler.has_data:
            market_event = data_handler.get_next_bar()
            if market_event is None:
                break

            date   = market_event.date
            prices = market_event.prices

            portfolio.update_prices(prices)

            fills = broker.execute_pending(prices)
            for fill in fills:
                portfolio.fill_order(fill)
                n_trades += 1

            prices_hist = data_handler.get_history(date, 252)
            portfolio_values = portfolio.get_portfolio_value_series(date)
            risk_snapshot = risk_manager.compute_risk_snapshot(
                date=date,
                prices=prices_hist,
                portfolio_values=portfolio_values,
            )

            if dd_max_triggered:
                risk_snapshot.dd_score = 0.0
                risk_snapshot.regime_score_raw = 0.0
                risk_snapshot.regime_score = 0.0
                risk_snapshot.dd_max_stop = True
            else:
                regime_buffer.append(risk_snapshot.regime_score_raw)
                risk_snapshot.regime_score = float(np.mean(regime_buffer))
                if risk_snapshot.dd_max_stop:
                    risk_snapshot.regime_score_raw = 0.0
                    risk_snapshot.regime_score = 0.0

            regime_score = risk_snapshot.regime_score
            turnover     = 0.0
            emergency_exit = False

            if risk_snapshot.dd_max_stop and not dd_max_triggered:
                dd_max_triggered = True
                base_signal_weights = {}
                emergency_exit = True
                logger.warning(
                    f"  DD.MAX TRIGGER - {date.date()} | "
                    f"DD courant: {risk_snapshot.drawdown:.2%} | "
                    f"Seuil: -{MAX_PORTFOLIO_DRAWDOWN:.0%}"
                )

            rebal_today = (
                not dd_max_triggered and (
                    last_rebal_date is None or
                    (date.year * 12 + date.month) >
                    (last_rebal_date.year * 12 + last_rebal_date.month)
                )
            )

            if rebal_today:
                base_weights = signal_gen.compute_base_weights(date)
                if base_weights is not None:
                    base_signal_weights = base_weights
                    last_rebal_date = date

            target_weights = signal_gen.scale_weights(base_signal_weights, regime_score)
            signal_event = SignalEvent(
                date=date,
                weights=target_weights,
                regime=regime_score,
                signal=Signal.FLAT if not target_weights else Signal.HOLD,
            )
            orders = portfolio.generate_orders(signal_event, prices)
            if orders:
                pv = portfolio.portfolio_value
                if pv > 0:
                    turnover = sum(abs(o.quantity * o.price) for o in orders) / pv
                for order in orders:
                    broker.submit_order(order)

            stats = portfolio.compute_stats(
                date=date,
                regime_score=regime_score,
                turnover=turnover,
                emergency_exit=emergency_exit,
                dd_score=risk_snapshot.dd_score,
                dd_max_stop=risk_snapshot.dd_max_stop,
            )

            self.shared.add_stats(stats)
            day_count += 1

            if self.speed_factor > 0:
                time.sleep(self.speed_factor)

        self.shared.set_done()
        logger.info(f"  Thread backtest terminé — {day_count} jours | {n_trades} trades")


# ============================================================
# APPLICATION DASH
# ============================================================

def create_dash_app(shared_state):
    try:
        import dash
        from dash import dcc, html, Input, Output
        import plotly.graph_objects as go
    except ImportError:
        logger.error("Dash non installé. pip install dash")
        sys.exit(1)

    C = {
        "bg"    : "#0d1117",
        "paper" : "#161b22",
        "grid"  : "#21262d",
        "text"  : "#c9d1d9",
        "blue"  : "#58a6ff",
        "green" : "#3fb950",
        "red"   : "#f85149",
        "gray"  : "#8b949e",
        "orange": "#e3b341",
    }

    app = dash.Dash(__name__, title="Event-Driven Live")

    app.layout = html.Div(
        style={"backgroundColor": C["bg"], "minHeight": "100vh",
               "fontFamily": "monospace", "padding": "20px"},
        children=[
            html.H1("EVENT-DRIVEN BACKTEST — LIVE",
                    style={"color": C["blue"], "textAlign": "center",
                           "fontSize": "22px", "marginBottom": "5px"}),

            html.Div(id="status-bar",
                     style={"color": C["gray"], "textAlign": "center",
                            "fontSize": "13px", "marginBottom": "15px"}),

            html.Div(id="metrics-row", style={
                "display": "flex", "justifyContent": "space-around",
                "marginBottom": "15px",
            }),

            dcc.Graph(id="chart-3d", style={"height": "520px"},
                      config={"displayModeBar": True, "scrollZoom": True}),

            html.Div([
                html.Div([dcc.Graph(id="chart-vol",
                                   style={"height": "230px"},
                                   config={"displayModeBar": False})],
                         style={"width": "50%"}),
                html.Div([dcc.Graph(id="chart-dd",
                                   style={"height": "230px"},
                                   config={"displayModeBar": False})],
                         style={"width": "50%"}),
            ], style={"display": "flex"}),

            html.Div([
                html.Div([dcc.Graph(id="chart-regime",
                                   style={"height": "200px"},
                                   config={"displayModeBar": False})],
                         style={"width": "50%"}),
                html.Div([dcc.Graph(id="chart-equity",
                                   style={"height": "200px"},
                                   config={"displayModeBar": False})],
                         style={"width": "50%"}),
            ], style={"display": "flex"}),

            dcc.Interval(id="interval", interval=500, n_intervals=0),
        ]
    )

    def empty_fig():
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=C["bg"], plot_bgcolor=C["paper"],
            font=dict(color=C["text"]),
            margin=dict(l=40, r=20, t=30, b=30),
        )
        return fig

    @app.callback(
        Output("chart-3d",     "figure"),
        Output("chart-vol",    "figure"),
        Output("chart-dd",     "figure"),
        Output("chart-regime", "figure"),
        Output("chart-equity", "figure"),
        Output("metrics-row",  "children"),
        Output("status-bar",   "children"),
        Input("interval",      "n_intervals"),
    )
    def update_all(n_intervals):
        stats_list = shared_state.get_stats_copy()

        if len(stats_list) < 3:
            e = empty_fig()
            return e, e, e, e, e, [], "⏳ Initialisation..."

        dates   = [s.date for s in stats_list]
        pv      = [s.portfolio_value for s in stats_list]
        rv      = [s.realized_vol for s in stats_list]
        ev      = [s.expected_vol for s in stats_list]
        dd      = [s.drawdown * 100 for s in stats_list]
        regime  = [s.regime_score for s in stats_list]
        pv_norm = [(v / stats_list[0].portfolio_value - 1) * 100 for v in pv]

        rv_arr = np.array(rv) * 100
        ev_arr = np.array(ev) * 100
        n_days = len(stats_list)

        last_pv  = pv_norm[-1]
        last_rv  = rv[-1] * 100
        last_reg = regime[-1]

        reg_label = "BULL"   if last_reg > 0.7 else \
                    "NORMAL" if last_reg > 0.5 else \
                    "REDUIT" if last_reg > 0.3 else "CRISE"
        reg_color = C["green"]  if last_reg > 0.7 else \
                    C["blue"]   if last_reg > 0.5 else \
                    C["orange"] if last_reg > 0.3 else C["red"]

        # Status
        icon   = "🏃" if shared_state.is_running else "✅ TERMINÉ"
        status = (f"{icon} | Jour {n_days}/2217 | "
                  f"Date: {dates[-1].strftime('%Y-%m-%d')} | "
                  f"Régime: {reg_label}")

        # Métriques
        rets   = np.array([s.daily_return for s in stats_list])
        rf_d   = (1 + RISK_FREE_RATE) ** (1/252) - 1
        sharpe = float((rets.mean() - rf_d) / (rets.std() + 1e-8) * np.sqrt(252))
        max_dd = float(min(dd) / 100)
        cagr   = float((pv[-1] / INITIAL_CAPITAL) ** (252/n_days) - 1) \
                 if n_days > 252 else 0.0

        def card(label, value, fmt, good):
            color = C["green"] if good else C["red"]
            return html.Div([
                html.Div(format(value, fmt),
                         style={"color": color, "fontSize": "22px",
                                "fontWeight": "bold"}),
                html.Div(label,
                         style={"color": C["gray"], "fontSize": "10px"}),
            ], style={
                "background": C["paper"],
                "border": f"1px solid {C['grid']}",
                "borderRadius": "6px",
                "padding": "10px 16px",
                "textAlign": "center",
                "minWidth": "100px",
            })

        metrics = [
            card("CAGR",   cagr,        ".1%",  cagr > 0),
            card("Sharpe", sharpe,       ".3f",  sharpe > 0.5),
            card("Max DD", max_dd,       ".1%",  max_dd > -0.20),
            card("P&L",    last_pv/100,  ".1%",  last_pv > 0),
            card("Vol",    last_rv/100,  ".1%",  True),
            card("Régime", last_reg,     ".2f",  last_reg > 0.5),
        ]

        # ── FIGURE 3D ─────────────────────────────────────
        fig_3d = go.Figure()

        fig_3d.add_trace(go.Scatter3d(
            x=list(range(n_days)),
            y=list(rv_arr),
            z=pv_norm,
            mode="lines",
            line=dict(color=pv_norm, colorscale="RdYlGn", width=5,
                      cmin=min(pv_norm), cmax=max(pv_norm)),
            name="P&L",
            hovertemplate="Jour %{x}<br>Vol: %{y:.1f}%<br>P&L: %{z:.1f}%<extra></extra>",
        ))

        step = max(1, n_days // 80)
        fig_3d.add_trace(go.Scatter3d(
            x=list(range(0, n_days, step)),
            y=[rv_arr[i] for i in range(0, n_days, step)],
            z=[pv_norm[i] for i in range(0, n_days, step)],
            mode="markers",
            marker=dict(
                size=4, opacity=0.7,
                color=[regime[i] for i in range(0, n_days, step)],
                colorscale=[[0.0,"#f85149"],[0.3,"#e3b341"],
                            [0.7,"#58a6ff"],[1.0,"#3fb950"]],
                cmin=0, cmax=1,
            ),
            name="Régime", showlegend=False,
        ))

        fig_3d.update_layout(
            paper_bgcolor=C["bg"],
            font=dict(color=C["text"], family="monospace"),
            margin=dict(l=0, r=0, t=45, b=0),
            title=dict(
                text=f"P&L: {last_pv:+.1f}% | Vol: {last_rv:.1f}% | Régime: {reg_label}",
                font=dict(size=13, color=reg_color),
                x=0.5, xanchor="center",
            ),
            showlegend=True,
            legend=dict(bgcolor="rgba(22,27,34,0.8)", font=dict(size=9)),
            scene=dict(
                xaxis=dict(title="Jours",
                           backgroundcolor=C["paper"],
                           gridcolor=C["grid"], showgrid=True),
                yaxis=dict(title="Vol réalisée (%)",
                           backgroundcolor=C["paper"],
                           gridcolor=C["grid"], showgrid=True),
                zaxis=dict(title="P&L (%)",
                           backgroundcolor=C["paper"],
                           gridcolor=C["grid"], showgrid=True),
                bgcolor=C["bg"],
                camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
            ),
        )

        # ── VOL RÉALISÉE vs ATTENDUE ──────────────────────
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=dates, y=list(rv_arr), name="Vol réalisée",
            line=dict(color=C["blue"], width=1.5)))
        fig_vol.add_trace(go.Scatter(
            x=dates, y=list(ev_arr), name="Vol attendue",
            line=dict(color=C["orange"], width=1.5, dash="dot")))
        fig_vol.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=list(np.maximum(rv_arr, ev_arr)) + \
              list(np.minimum(rv_arr, ev_arr))[::-1],
            fill="toself", fillcolor="rgba(248,81,73,0.08)",
            line=dict(width=0), showlegend=False))
        fig_vol.update_layout(
            paper_bgcolor=C["bg"],
            plot_bgcolor =C["paper"],
            font=dict(color=C["text"], family="monospace"),
            margin=dict(l=50, r=20, t=35, b=35),
            showlegend=True,
            legend=dict(bgcolor="rgba(22,27,34,0.8)", font=dict(size=9)),
            title=dict(text="Realized Vol vs Expected Vol",
                       font=dict(size=11, color=C["blue"])),
            xaxis=dict(gridcolor=C["grid"]),
            yaxis=dict(gridcolor=C["grid"], ticksuffix="%"),
        )

        # ── DRAWDOWN ──────────────────────────────────────
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dates, y=dd, name="Drawdown",
            fill="tozeroy",
            line=dict(color=C["red"], width=1),
            fillcolor="rgba(248,81,73,0.3)"))
        fig_dd.update_layout(
            paper_bgcolor=C["bg"],
            plot_bgcolor =C["paper"],
            font=dict(color=C["text"], family="monospace"),
            margin=dict(l=50, r=20, t=35, b=35),
            showlegend=True,
            legend=dict(bgcolor="rgba(22,27,34,0.8)", font=dict(size=9)),
            title=dict(text=f"Drawdown (Max: {min(dd):.1f}%)",
                       font=dict(size=11, color=C["red"])),
            xaxis=dict(gridcolor=C["grid"]),
            yaxis=dict(gridcolor=C["grid"], ticksuffix="%"),
        )

        # ── SCORE DE RÉGIME ───────────────────────────────
        fig_regime = go.Figure()
        fig_regime.add_trace(go.Scatter(
            x=dates, y=regime, name="Score régime",
            fill="tozeroy",
            line=dict(color=C["blue"], width=1.5),
            fillcolor="rgba(88,166,255,0.2)"))
        for y_val, color, label in [
            (0.7, C["green"],  "BULL"),
            (0.5, C["blue"],   "NORMAL"),
            (0.3, C["orange"], "REDUIT"),
        ]:
            fig_regime.add_shape(
                type="line",
                line=dict(dash="dot", color=color, width=1),
                x0=dates[0], x1=dates[-1], xref="x",
                y0=y_val, y1=y_val, yref="y")
            fig_regime.add_annotation(
                x=dates[-1], y=y_val, text=label,
                font=dict(color=color, size=9),
                showarrow=False, xanchor="right")
        fig_regime.update_layout(
            paper_bgcolor=C["bg"],
            plot_bgcolor =C["paper"],
            font=dict(color=C["text"], family="monospace"),
            margin=dict(l=50, r=20, t=35, b=35),
            showlegend=True,
            legend=dict(bgcolor="rgba(22,27,34,0.8)", font=dict(size=9)),
            title=dict(text=f"Régime: {reg_label} ({last_reg:.2f})",
                       font=dict(size=11, color=reg_color)),
            xaxis=dict(gridcolor=C["grid"]),
            yaxis=dict(gridcolor=C["grid"], range=[0, 1.05]),
        )

        # ── EQUITY CURVE ─────────────────────────────────
        fig_equity = go.Figure()
        pv_dollar = [INITIAL_CAPITAL * (1 + p/100) for p in pv_norm]
        fig_equity.add_trace(go.Scatter(
            x=dates, y=pv_dollar, name="Portefeuille",
            line=dict(color=C["green"], width=2),
            fill="tozeroy",
            fillcolor="rgba(63,185,80,0.1)",
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>"))
        fig_equity.update_layout(
            paper_bgcolor=C["bg"],
            plot_bgcolor =C["paper"],
            font=dict(color=C["text"], family="monospace"),
            margin=dict(l=50, r=20, t=35, b=35),
            showlegend=True,
            legend=dict(bgcolor="rgba(22,27,34,0.8)", font=dict(size=9)),
            title=dict(text=f"Equity — ${pv_dollar[-1]:,.0f}",
                       font=dict(size=11, color=C["green"])),
            xaxis=dict(gridcolor=C["grid"]),
            yaxis=dict(gridcolor=C["grid"],
                       tickprefix="$", tickformat=",.0f"),
        )

        return (fig_3d, fig_vol, fig_dd, fig_regime, fig_equity,
                metrics, status)

    return app


# ============================================================
# MAIN
# ============================================================

def main():
    sys.stdout.reconfigure(encoding="utf-8")

    import argparse
    parser = argparse.ArgumentParser(description="Event-Driven Live Dashboard")
    parser.add_argument("--start",  default="2016-01-01")
    parser.add_argument("--end",    default="2024-12-31")
    parser.add_argument("--data",   default="./data/processed/price_matrix.csv")
    parser.add_argument("--port",   default=8050, type=int)
    parser.add_argument("--speed",  default=0.05, type=float,
                        help="Délai entre jours (0=max, 0.05=20j/s, 0.1=10j/s)")
    args = parser.parse_args()

    speed_str = "MAX" if args.speed == 0 else f"{1/args.speed:.0f} jours/sec"

    print("=" * 60)
    print("  EVENT-DRIVEN LIVE DASHBOARD")
    print("=" * 60)
    print(f"  Période : {args.start} → {args.end}")
    print(f"  Capital : ${INITIAL_CAPITAL:,.0f}")
    print(f"  Speed   : {speed_str}")
    print(f"  URL     : http://localhost:{args.port}")
    print("=" * 60)

    shared = SharedState()

    backtest = BacktestThread(
        shared_state   = shared,
        data_path      = args.data,
        start_date     = args.start,
        end_date       = args.end,
        initial_capital= INITIAL_CAPITAL,
        speed_factor   = args.speed,
    )
    backtest.start()
    logger.info("  Thread backtest lancé")

    app = create_dash_app(shared)

    print(f"\n  → Ouvre : http://localhost:{args.port}\n")

    app.run(
        host         = "0.0.0.0",
        port         = args.port,
        debug        = False,
        use_reloader = False,
    )


if __name__ == "__main__":
    main()
