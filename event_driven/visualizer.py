# ============================================================
# LiveVisualizer3D — export HTML Plotly
# ============================================================
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class LiveVisualizer3D:
    """
    Dashboard 3D : surface P&L × Volatilité × Temps
    + 4 panels 2D (Vol réalisée/attendue, Drawdown, Régime, Equity)

    NOTE : add_hline/add_vline ne fonctionnent pas avec les subplots
    mixtes 3D+2D. On utilise add_shape + add_annotation à la place.
    """

    def __init__(self, update_every=20):
        self.update_every = update_every
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            self.go = go
            self.make_subplots = make_subplots
            self.available = True
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
        go = self.go
        make_subplots = self.make_subplots

        dates = [s.date for s in stats_list]
        pv = [s.portfolio_value for s in stats_list]
        rv = [s.realized_vol for s in stats_list]
        ev = [s.expected_vol for s in stats_list]
        dd = [s.drawdown * 100 for s in stats_list]
        regime = [s.regime_score for s in stats_list]
        pv_norm = [(v / stats_list[0].portfolio_value - 1) * 100 for v in pv]

        rv_arr = np.array(rv) * 100
        ev_arr = np.array(ev) * 100

        fig = make_subplots(
            rows=3,
            cols=2,
            specs=[
                [{"type": "scene", "colspan": 2}, None],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
            ],
            subplot_titles=[
                "3D — P&L × Volatilité × Temps",
                "",
                "Realized Vol vs Expected Vol",
                "Drawdown",
                "Score de Régime",
                "Equity Curve",
            ],
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25],
        )

        fig.add_trace(
            go.Scatter3d(
                x=list(range(len(dates))),
                y=rv,
                z=pv_norm,
                mode="lines+markers",
                line=dict(color=pv_norm, colorscale="RdYlGn", width=4),
                marker=dict(
                    size=3,
                    opacity=0.8,
                    color=regime,
                    colorscale=[
                        [0.0, "#f85149"],
                        [0.3, "#e3b341"],
                        [0.7, "#58a6ff"],
                        [1.0, "#3fb950"],
                    ],
                ),
                name="P&L",
                hovertemplate="Jour: %{x}<br>Vol: %{y:.1%}<br>P&L: %{z:.1f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=list(rv_arr),
                name="Vol réalisée",
                line=dict(color="#58a6ff", width=1.5),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=list(ev_arr),
                name="Vol attendue",
                line=dict(color="#e3b341", width=1.5, dash="dot"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates + dates[::-1],
                y=list(np.maximum(rv_arr, ev_arr)) + list(np.minimum(rv_arr, ev_arr))[::-1],
                fill="toself",
                fillcolor="rgba(248,81,73,0.1)",
                line=dict(width=0),
                name="Zone stress",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=dd,
                name="Drawdown",
                fill="tozeroy",
                line=dict(color="#f85149", width=1),
                fillcolor="rgba(248,81,73,0.3)",
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=regime,
                name="Régime",
                line=dict(width=0),
                fill="tozeroy",
                fillcolor="rgba(88,166,255,0.3)",
            ),
            row=3,
            col=1,
        )

        for y_val, color, label in [
            (0.7, "#3fb950", "BULL"),
            (0.5, "#58a6ff", "NORMAL"),
            (0.3, "#e3b341", "REDUIT"),
        ]:
            fig.add_shape(
                type="line",
                line=dict(dash="dot", color=color, width=1),
                x0=0,
                x1=1,
                xref="x4 domain",
                y0=y_val,
                y1=y_val,
                yref="y4",
            )
            fig.add_annotation(
                x=1,
                xref="x4 domain",
                xanchor="right",
                y=y_val,
                yref="y4",
                text=label,
                font=dict(color=color, size=9),
                showarrow=False,
                bgcolor="rgba(13,17,23,0.7)",
            )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=pv_norm,
                name="P&L %",
                line=dict(color="#3fb950", width=2),
                fill="tozeroy",
                fillcolor="rgba(63,185,80,0.15)",
            ),
            row=3,
            col=2,
        )

        fig.add_shape(
            type="line",
            line=dict(color="#8b949e", width=0.8),
            x0=0,
            x1=1,
            xref="x5 domain",
            y0=0,
            y1=0,
            yref="y5",
        )

        n_days = len(dates)
        last_pv = pv_norm[-1] if pv_norm else 0
        last_rv = rv[-1] * 100 if rv else 0
        last_reg = regime[-1] if regime else 0
        reg_label = (
            "BULL"
            if last_reg > 0.7
            else "NORMAL"
            if last_reg > 0.5
            else "REDUIT"
            if last_reg > 0.3
            else "CRISE"
        )

        fig.update_layout(
            title=dict(
                text=(
                    f"EVENT-DRIVEN BACKTEST — {n_days} jours | "
                    f"P&L: {last_pv:+.1f}% | Vol: {last_rv:.1f}% | Régime: {reg_label}"
                ),
                font=dict(size=16, color="#58a6ff", family="monospace"),
                x=0.5,
                xanchor="center",
            ),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
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
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"event_driven_3d{suffix}_{ts}.html"
        self.fig.write_html(
            str(path),
            include_plotlyjs="cdn",
            config={"displayModeBar": True, "scrollZoom": True},
        )
        logger.info(f"  Dashboard 3D sauvegardé : {path}")
        return str(path)
