# CLI: python -m event_driven.strategy_benchmark_compare
# HTML autonome : courbe event-driven vs buy-and-hold équipondéré (même univers / dates).
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import INITIAL_CAPITAL, PROCESSED_DATA_PATH, RISK_FREE_RATE

from .regime_phase_report import build_equal_weight_benchmark, find_run_files

TRADING_DAYS = 252


def _ann_return_from_daily(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    return float((1.0 + r).prod() ** (TRADING_DAYS / len(r)) - 1.0)


def _ann_vol(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    return float(r.std(ddof=0) * np.sqrt(TRADING_DAYS))


def _sharpe(r: pd.Series, rf_annual: float) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    rf_d = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0
    ex = r - rf_d
    vol = ex.std(ddof=0)
    if vol <= 1e-12:
        return float("nan")
    return float(ex.mean() / vol * np.sqrt(TRADING_DAYS))


def _max_dd_from_equity(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _metrics_block(name: str, daily_ret: pd.Series, equity: pd.Series, dd_series: pd.Series | None) -> dict:
    mdd = float(dd_series.min()) if dd_series is not None and not dd_series.empty else _max_dd_from_equity(equity)
    return {
        "name": name,
        "cagr": _ann_return_from_daily(daily_ret),
        "ann_vol": _ann_vol(daily_ret),
        "max_dd": mdd,
        "sharpe": _sharpe(daily_ret, RISK_FREE_RATE),
        "final_value": float(equity.iloc[-1]) if not equity.empty else float("nan"),
    }


def _html_table(rows: list[dict]) -> str:
    headers = ["", "CAGR", "Vol. ann.", "Max DD", "Sharpe*", "Valeur finale"]
    cells = []
    for row in rows:
        cells.append(
            "<tr>"
            f"<td><b>{row['name']}</b></td>"
            f"<td>{row['cagr']:.2%}</td>"
            f"<td>{row['ann_vol']:.2%}</td>"
            f"<td>{row['max_dd']:.2%}</td>"
            f"<td>{row['sharpe']:.3f}</td>"
            f"<td>${row['final_value']:,.0f}</td>"
            "</tr>"
        )
    return (
        "<table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;font-family:system-ui'>"
        "<thead><tr>"
        + "".join(f"<th>{h}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(cells)
        + "</tbody></table>"
        f"<p style='font-size:0.9em;color:#555'>* Sharpe annualisé, excès vs taux sans risque config "
        f"(RISK_FREE_RATE = {RISK_FREE_RATE:.1%} annualisé).</p>"
    )


def build_report_html(
    stats_path: Path,
    price_matrix_path: Path,
    output_path: Path,
    initial_capital: float = INITIAL_CAPITAL,
) -> Path:
    stats = pd.read_csv(stats_path, parse_dates=["date"])
    stats = stats.sort_values("date").reset_index(drop=True)
    if "portfolio_value" not in stats.columns:
        raise ValueError("stats CSV doit contenir portfolio_value")

    strat_equity = stats["portfolio_value"].astype(float)
    strat_dd = stats["drawdown"].astype(float) if "drawdown" in stats.columns else None
    if "daily_return" in stats.columns:
        strat_r = stats["daily_return"].astype(float)
    else:
        strat_r = strat_equity.pct_change().fillna(0.0)

    bm = build_equal_weight_benchmark(stats["date"], price_matrix_path, initial_capital)
    if bm is None or bm.empty:
        raise RuntimeError("Impossible de construire le benchmark EW (price_matrix manquant ou vide ?)")

    bm = bm.sort_values("date").reset_index(drop=True)
    bm_equity = pd.Series(bm["benchmark_value"].values, index=bm.index)
    bm_dd = pd.Series(bm["benchmark_drawdown"].values, index=bm.index)
    bm_r = bm_equity.pct_change().fillna(0.0)

    m_strat = _metrics_block("Event-driven", strat_r, strat_equity, strat_dd)
    m_bm = _metrics_block("Buy & hold équipondéré (univers)", bm_r, bm_equity, bm_dd)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        row_heights=[0.58, 0.42],
        subplot_titles=("Valeur du portefeuille ($)", "Drawdown"),
    )
    fig.add_trace(
        go.Scatter(
            x=stats["date"],
            y=strat_equity,
            name="Event-driven",
            line=dict(color="#2b8a3e", width=2.2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=bm["date"],
            y=bm["benchmark_value"],
            name="Buy & hold EW",
            line=dict(color="#1864ab", width=2, dash="dot"),
        ),
        row=1,
        col=1,
    )
    if strat_dd is not None:
        fig.add_trace(
            go.Scatter(
                x=stats["date"],
                y=strat_dd,
                name="DD stratégie",
                line=dict(color="#c92a2a", width=1.4),
                fill="tozeroy",
                fillcolor="rgba(201,42,42,0.12)",
            ),
            row=2,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=bm["date"],
            y=bm["benchmark_drawdown"],
            name="DD benchmark EW",
            line=dict(color="#0b7285", width=1.4, dash="dot"),
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title="Event-driven vs buy-and-hold équipondéré (même calendrier)",
        template="plotly_white",
        height=720,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=100, b=60),
    )
    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)

    chart_div = fig.to_html(full_html=False, include_plotlyjs="cdn")

    intro = (
        "<p>Comparaison sur <b>les mêmes dates</b> que le run event-driven. "
        "Le benchmark est un <b>buy-and-hold équipondéré</b> rééquilibré implicitement "
        "chaque jour via la moyenne des rendements journaliers des titres de "
        f"<code>{price_matrix_path.name}</code> (pas de coûts de transaction sur le benchmark).</p>"
        "<p><i>En pratique, un EW long-only sur ce type d’univers peut montrer une perf brute "
        "plus élevée en phase haussière, souvent avec un drawdown plus profond — "
        "ce que tu peux lire sur le graphique du bas.</i></p>"
    )

    meta = (
        f"<p style='color:#666'>Fichier stats : <code>{stats_path}</code><br>"
        f"Capital initial : ${initial_capital:,.0f}</p>"
    )

    full = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Stratégie vs benchmark EW</title>
  <style>body{{font-family:system-ui,Segoe UI,sans-serif;margin:24px;max-width:1100px;}}</style>
</head>
<body>
  <h1>Event-driven vs buy-and-hold équipondéré</h1>
  {intro}
  {meta}
  <h2>Métriques (sur la fenêtre du run)</h2>
  {_html_table([m_strat, m_bm])}
  <h2 style="margin-top:2rem">Graphiques</h2>
  {chart_div}
</body>
</html>"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère un HTML : courbes + tableau event-driven vs EW buy-and-hold."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./results/event_driven"),
        help="Dossier contenant stats_*.csv (utilise le plus récent si --stats absent)",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=None,
        help="Chemin explicite vers stats_*.csv (prioritaire sur --results-dir)",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Suffixe YYYYMMDD_HHMMSS du run si plusieurs stats dans results-dir",
    )
    parser.add_argument(
        "--price-matrix",
        type=Path,
        default=None,
        help=f"CSV prix (défaut: {Path(PROCESSED_DATA_PATH) / 'price_matrix.csv'})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Fichier HTML de sortie",
    )
    args = parser.parse_args()

    pm = (
        Path(args.price_matrix)
        if args.price_matrix
        else Path(PROCESSED_DATA_PATH) / "price_matrix.csv"
    )
    if args.stats is not None:
        stats_path = Path(args.stats)
        if not stats_path.exists():
            raise SystemExit(f"Fichier introuvable : {stats_path}")
        ts = stats_path.stem.replace("stats_", "", 1)
    else:
        rf = find_run_files(Path(args.results_dir), args.timestamp)
        stats_path = rf.stats
        ts = rf.timestamp

    out = args.output
    if out is None:
        out = Path(args.results_dir) / f"strategy_vs_benchmark_{ts}.html"

    path = build_report_html(stats_path, pm, out)
    print(f"Rapport écrit : {path.resolve()}")


if __name__ == "__main__":
    main()
