from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import to_html

from config import INITIAL_CAPITAL, PROCESSED_DATA_PATH


TRADING_DAYS = 252
RISK_OFF_STRICT_LABELS = {"RISK_OFF", "CRISIS", "STRESS"}
RISK_OFF_EXTENDED_LABELS = {"TRANSITION"}


@dataclass
class RunFiles:
    stats: Path
    rebal: Path | None
    regimes: Path | None
    timestamp: str


def _infer_timestamp(path: Path, prefix: str) -> str:
    name = path.stem
    return name.replace(prefix, "", 1)


def find_run_files(results_dir: Path, timestamp: str | None) -> RunFiles:
    if timestamp:
        stats = results_dir / f"stats_{timestamp}.csv"
        if not stats.exists():
            raise FileNotFoundError(f"Missing file: {stats}")
    else:
        stats_files = sorted(results_dir.glob("stats_*.csv"))
        if not stats_files:
            raise FileNotFoundError(f"No stats_*.csv found in {results_dir}")
        stats = stats_files[-1]
        timestamp = _infer_timestamp(stats, "stats_")

    rebal = results_dir / f"rebal_diagnostics_{timestamp}.csv"
    regimes = results_dir / f"regimes_{timestamp}.csv"
    return RunFiles(
        stats=stats,
        rebal=rebal if rebal.exists() else None,
        regimes=regimes if regimes.exists() else None,
        timestamp=timestamp,
    )


def build_equal_weight_benchmark(
    stats_dates: pd.Series,
    price_matrix_path: Path,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame | None:
    """
    Buy-and-hold équipondéré sur la matrice de prix (même univers que le backtest en pratique).
    Aligné sur les dates du run ; valeurs en $ comme la courbe stratégie.
    """
    path = Path(price_matrix_path)
    if not path.exists():
        return None
    try:
        pm = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    except Exception:
        return None
    if pm.empty or pm.shape[1] < 1:
        return None
    pm.index = pd.to_datetime(pm.index).normalize()
    dates = pd.to_datetime(stats_dates).dt.normalize()
    prices = pm.reindex(dates).ffill().bfill()
    if prices.isna().all().all():
        return None
    rets = prices.pct_change().mean(axis=1, skipna=True).fillna(0.0)
    equity = float(initial_capital) * (1.0 + rets).cumprod()
    dd = equity / equity.cummax() - 1.0
    return pd.DataFrame(
        {
            "date": dates.values,
            "benchmark_value": equity.values.astype(float),
            "benchmark_drawdown": dd.values.astype(float),
        }
    )


def _safe_ann_return(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return np.nan
    return float((1.0 + r).prod() ** (TRADING_DAYS / len(r)) - 1.0)


def _safe_ann_vol(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return np.nan
    return float(r.std(ddof=0) * np.sqrt(TRADING_DAYS))


def _safe_sharpe(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return np.nan
    vol = r.std(ddof=0)
    if vol <= 1e-12:
        return np.nan
    return float((r.mean() / vol) * np.sqrt(TRADING_DAYS))


def _safe_sortino(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return np.nan
    downside = r[r < 0]
    dd = downside.std(ddof=0)
    if dd <= 1e-12:
        return np.nan
    return float((r.mean() / dd) * np.sqrt(TRADING_DAYS))


def _max_dd_from_returns(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return np.nan
    curve = (1.0 + r).cumprod()
    peak = curve.cummax()
    dd = curve / peak - 1.0
    return float(dd.min())


def _episode_table(df: pd.DataFrame, value_col: str, predicate, group_col: str) -> pd.DataFrame:
    working = df.copy()
    working["in_episode"] = predicate(working[value_col]).astype(int)
    phase_change = working[group_col].ne(working[group_col].shift())
    episode_change = working["in_episode"].ne(working["in_episode"].shift())
    working["segment_id"] = (phase_change | episode_change).cumsum()

    rows = []
    for _, seg in working.groupby("segment_id", sort=False):
        if int(seg["in_episode"].iloc[0]) == 0:
            continue
        phase = str(seg[group_col].iloc[0])
        start_date = pd.to_datetime(seg["date"].iloc[0])
        end_date = pd.to_datetime(seg["date"].iloc[-1])
        length = int(len(seg))
        min_dd = float(seg["drawdown"].min()) if "drawdown" in seg.columns else np.nan
        cum_return = float((1.0 + seg["daily_return"]).prod() - 1.0) if "daily_return" in seg.columns else np.nan
        rows.append(
            {
                "phase": phase,
                "start_date": start_date,
                "end_date": end_date,
                "length_days": length,
                "min_drawdown": min_dd,
                "cum_return": cum_return,
            }
        )
    return pd.DataFrame(rows)


def _is_risk_off_label(label: str) -> bool:
    upper = (label or "").upper().strip()
    if not upper:
        return False
    if upper in RISK_OFF_STRICT_LABELS or upper in RISK_OFF_EXTENDED_LABELS:
        return True
    return ("RISK_OFF" in upper) or ("CRISIS" in upper) or ("STRESS" in upper)


def build_metrics_tables(stats_df: pd.DataFrame, rebal_df: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats = stats_df.copy()
    stats["phase"] = stats["market_regime_effective"].fillna("").astype(str)
    if stats["phase"].str.len().eq(0).all():
        stats["phase"] = stats["market_regime_state"].fillna("").astype(str)
    if stats["phase"].str.len().eq(0).all():
        stats["phase"] = stats["regime_state"].fillna("").astype(str)
    stats["phase"] = stats["phase"].replace("", "UNKNOWN")
    stats["risk_off_like"] = stats["phase"].map(_is_risk_off_label)

    phase_rows = []
    n_total = len(stats)
    for phase, grp in stats.groupby("phase", sort=False):
        ret = grp["daily_return"]
        phase_rows.append(
            {
                "phase": phase,
                "days": int(len(grp)),
                "pct_days": float(len(grp) / n_total),
                "avg_daily_return": float(ret.mean()),
                "annualized_return": _safe_ann_return(ret),
                "annualized_vol": _safe_ann_vol(ret),
                "sharpe": _safe_sharpe(ret),
                "sortino": _safe_sortino(ret),
                "max_dd_subcurve": _max_dd_from_returns(ret),
                "mean_portfolio_drawdown": float(grp["drawdown"].mean()),
                "worst_portfolio_drawdown": float(grp["drawdown"].min()),
                "q05_portfolio_drawdown": float(grp["drawdown"].quantile(0.05)),
                "avg_turnover_daily": float(grp["turnover"].fillna(0.0).mean()),
                "avg_turnover_annualized": float(grp["turnover"].fillna(0.0).mean() * TRADING_DAYS),
                "avg_realized_vol": float(grp["realized_vol"].mean()),
                "avg_risk_scaling": float(grp["risk_scaling"].mean()),
                "suspended_days": int(grp["trading_suspended"].fillna(False).astype(bool).sum()),
                "dd_max_stop_days": int(grp["dd_max_stop"].fillna(False).astype(bool).sum()),
            }
        )
    phase_table = pd.DataFrame(phase_rows).sort_values("days", ascending=False).reset_index(drop=True)

    risk_band_rows = []
    for band, grp in stats.groupby("risk_off_like", sort=False):
        name = "RISK_OFF_LIKE" if band else "RISK_ON_LIKE"
        ret = grp["daily_return"]
        risk_band_rows.append(
            {
                "band": name,
                "days": int(len(grp)),
                "pct_days": float(len(grp) / n_total),
                "annualized_return": _safe_ann_return(ret),
                "annualized_vol": _safe_ann_vol(ret),
                "sharpe": _safe_sharpe(ret),
                "sortino": _safe_sortino(ret),
                "max_dd_subcurve": _max_dd_from_returns(ret),
                "mean_portfolio_drawdown": float(grp["drawdown"].mean()),
                "worst_portfolio_drawdown": float(grp["drawdown"].min()),
                "avg_turnover_annualized": float(grp["turnover"].fillna(0.0).mean() * TRADING_DAYS),
                "avg_risk_scaling": float(grp["risk_scaling"].mean()),
            }
        )
    risk_band_table = pd.DataFrame(risk_band_rows)

    if rebal_df is not None and not rebal_df.empty:
        rb = rebal_df.copy()
        rb["phase"] = rb["market_regime_effective"].fillna("").replace("", "UNKNOWN")
        exposure_table = (
            rb.groupby("phase", as_index=False)
            .agg(
                rebal_count=("date", "count"),
                avg_gross_signal_raw=("gross_signal_raw", "mean"),
                avg_gross_after_risk=("gross_after_risk_manager", "mean"),
                avg_gross_after_old_filter=("gross_after_old_regime_filter", "mean"),
                avg_gross_after_informed=("gross_after_informed_tilt", "mean"),
                avg_gross_after_overlay=("gross_after_market_overlay", "mean"),
                avg_turnover_rebal=("turnover", "mean"),
                total_turnover_rebal=("turnover", "sum"),
                avg_n_orders=("n_orders", "mean"),
            )
            .sort_values("rebal_count", ascending=False)
            .reset_index(drop=True)
        )
    else:
        exposure_table = pd.DataFrame(
            columns=[
                "phase",
                "rebal_count",
                "avg_gross_signal_raw",
                "avg_gross_after_risk",
                "avg_gross_after_old_filter",
                "avg_gross_after_informed",
                "avg_gross_after_overlay",
                "avg_turnover_rebal",
                "total_turnover_rebal",
                "avg_n_orders",
            ]
        )

    transitions = stats[["date", "phase"]].copy()
    transitions["phase_next"] = transitions["phase"].shift(-1)
    transitions = transitions.dropna()
    transition_table = (
        transitions.groupby(["phase", "phase_next"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    return phase_table, risk_band_table, exposure_table, transition_table


def build_figures(
    stats_df: pd.DataFrame,
    rebal_df: pd.DataFrame | None,
    phase_table: pd.DataFrame,
    transition_table: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
) -> list[str]:
    stats = stats_df.copy()
    stats["phase"] = stats["market_regime_effective"].fillna("").astype(str).replace("", "UNKNOWN")
    stats["risk_off_like"] = stats["phase"].map(_is_risk_off_label)

    fig_equity = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.62, 0.38])
    fig_equity.add_trace(
        go.Scatter(x=stats["date"], y=stats["portfolio_value"], name="Portfolio value", line=dict(color="#2b8a3e", width=2)),
        row=1,
        col=1,
    )
    if benchmark_df is not None and not benchmark_df.empty:
        bm = benchmark_df.sort_values("date")
        fig_equity.add_trace(
            go.Scatter(
                x=bm["date"],
                y=bm["benchmark_value"],
                name="Benchmark EW buy-and-hold",
                line=dict(color="#1f77b4", width=1.8, dash="dot"),
            ),
            row=1,
            col=1,
        )
    fig_equity.add_trace(
        go.Scatter(
            x=stats["date"],
            y=stats["drawdown"],
            name="Drawdown",
            line=dict(color="#c92a2a", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(201,42,42,0.18)",
        ),
        row=2,
        col=1,
    )
    if benchmark_df is not None and not benchmark_df.empty:
        bm = benchmark_df.sort_values("date")
        fig_equity.add_trace(
            go.Scatter(
                x=bm["date"],
                y=bm["benchmark_drawdown"],
                name="Drawdown benchmark EW",
                line=dict(color="#0b7285", width=1.3, dash="dot"),
            ),
            row=2,
            col=1,
        )
    risk_off_mask = stats["risk_off_like"].astype(bool).values
    starts = np.where((risk_off_mask == 1) & np.r_[True, risk_off_mask[:-1] == 0])[0]
    ends = np.where((risk_off_mask == 1) & np.r_[risk_off_mask[1:] == 0, True])[0]
    for s, e in zip(starts, ends):
        fig_equity.add_vrect(
            x0=stats["date"].iloc[int(s)],
            x1=stats["date"].iloc[int(e)],
            fillcolor="rgba(255,193,7,0.16)",
            line_width=0,
            row="all",
            col=1,
        )
    fig_equity.update_layout(
        title="Equity et drawdown (strategie vs benchmark EW + zones risk-off-like)",
        template="plotly_white",
        height=700,
        legend=dict(orientation="h"),
    )

    phase_order = phase_table["phase"].tolist()
    fig_returns = go.Figure()
    for phase in phase_order:
        grp = stats.loc[stats["phase"] == phase, "daily_return"]
        fig_returns.add_trace(
            go.Box(
                y=grp,
                name=phase,
                boxmean=True,
                marker_color="#1f77b4" if not _is_risk_off_label(phase) else "#d62728",
            )
        )
    fig_returns.update_layout(
        title="Distribution des rendements journaliers par phase",
        yaxis_title="Daily return",
        template="plotly_white",
        height=520,
    )

    fig_expo = go.Figure()
    if rebal_df is not None and not rebal_df.empty:
        rb = rebal_df.copy()
        rb["phase"] = rb["market_regime_effective"].fillna("").astype(str).replace("", "UNKNOWN")
        expo = rb.groupby("phase", as_index=False)["gross_after_market_overlay"].mean()
        expo = expo.set_index("phase").reindex(phase_order).reset_index()
        fig_expo.add_trace(go.Bar(x=expo["phase"], y=expo["gross_after_market_overlay"], name="Avg gross after overlay"))
    fig_expo.add_trace(
        go.Scatter(
            x=phase_table["phase"],
            y=phase_table["avg_turnover_annualized"],
            mode="lines+markers",
            yaxis="y2",
            name="Turnover annualisé",
            line=dict(color="#c92a2a", width=2),
        )
    )
    fig_expo.update_layout(
        title="Exposition engagée et turnover par phase",
        template="plotly_white",
        height=520,
        yaxis=dict(title="Gross exposure (x)"),
        yaxis2=dict(title="Turnover annualisé", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )

    if transition_table.empty:
        fig_trans = go.Figure()
        fig_trans.update_layout(title="Transitions de phase (aucune transition détectée)", template="plotly_white", height=400)
    else:
        matrix = transition_table.pivot(index="phase", columns="phase_next", values="count").fillna(0.0)
        fig_trans = go.Figure(
            data=go.Heatmap(
                z=matrix.values,
                x=matrix.columns,
                y=matrix.index,
                colorscale="Blues",
                colorbar=dict(title="Count"),
            )
        )
        fig_trans.update_layout(
            title="Matrice de transition des phases",
            template="plotly_white",
            height=520,
            xaxis_title="Phase suivante",
            yaxis_title="Phase courante",
        )

    return [
        to_html(fig_equity, full_html=False, include_plotlyjs="cdn"),
        to_html(fig_returns, full_html=False, include_plotlyjs=False),
        to_html(fig_expo, full_html=False, include_plotlyjs=False),
        to_html(fig_trans, full_html=False, include_plotlyjs=False),
    ]


def _format_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    if df.empty:
        return "<p><i>Aucune donnée disponible.</i></p>"
    return df.to_html(
        index=False,
        classes="metric-table",
        border=0,
        justify="left",
        float_format=lambda x: float_fmt.format(x),
    )


def generate_report(
    run_files: RunFiles,
    output_path: Path,
    top_n_worst_days: int = 25,
    price_matrix_path: Path | None = None,
    include_benchmark: bool = True,
) -> Path:
    stats = pd.read_csv(run_files.stats)
    stats["date"] = pd.to_datetime(stats["date"])
    stats = stats.sort_values("date").reset_index(drop=True)

    rebal = pd.read_csv(run_files.rebal) if run_files.rebal is not None else None
    if rebal is not None and not rebal.empty:
        rebal["date"] = pd.to_datetime(rebal["date"])
        rebal = rebal.sort_values("date").reset_index(drop=True)

    benchmark_df = None
    if include_benchmark:
        pm_path = price_matrix_path or (Path(PROCESSED_DATA_PATH) / "price_matrix.csv")
        benchmark_df = build_equal_weight_benchmark(stats["date"], pm_path, INITIAL_CAPITAL)

    phase_table, risk_band_table, exposure_table, transition_table = build_metrics_tables(stats, rebal)
    figures = build_figures(
        stats,
        rebal,
        phase_table,
        transition_table,
        benchmark_df=benchmark_df,
    )

    dd_episodes = _episode_table(
        df=stats.assign(phase=stats["market_regime_effective"].fillna("").replace("", "UNKNOWN")),
        value_col="drawdown",
        predicate=lambda s: s < 0,
        group_col="phase",
    )
    dd_episodes = dd_episodes.sort_values(["length_days", "min_drawdown"], ascending=[False, True]).reset_index(drop=True)
    dd_episodes_gt10 = dd_episodes[dd_episodes["length_days"] > 10].copy()

    worst_days = (
        stats.assign(phase=stats["market_regime_effective"].fillna("").replace("", "UNKNOWN"))
        .sort_values("drawdown")
        .head(top_n_worst_days)[
            [
                "date",
                "phase",
                "risk_regime_name",
                "drawdown",
                "daily_return",
                "turnover",
                "risk_scaling",
                "trading_suspended",
                "defensive_flat_phase",
            ]
        ]
        .reset_index(drop=True)
    )

    suspension_events = (
        stats.assign(phase=stats["market_regime_effective"].fillna("").replace("", "UNKNOWN"))
        .loc[
            lambda d: d["trading_suspended"].fillna(False).astype(bool),
            [
                "date",
                "phase",
                "risk_regime_name",
                "suspension_reason",
                "suspended_days",
                "drawdown",
                "daily_return",
                "risk_scaling",
                "defensive_flat_phase",
            ],
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )
    fast_cut_events = suspension_events.loc[
        suspension_events["suspension_reason"].fillna("").eq("FAST_DRAWDOWN_BREACH")
    ].copy()

    run_start = stats["date"].min().date()
    run_end = stats["date"].max().date()
    total_days = len(stats)
    overall_dd = float(stats["drawdown"].min())
    overall_ret = float((1.0 + stats["daily_return"].fillna(0.0)).prod() - 1.0)
    overall_cagr = _safe_ann_return(stats["daily_return"])
    overall_sharpe = _safe_sharpe(stats["daily_return"])

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8" />
  <title>Analyse phases de marche - {run_files.timestamp}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      background: #f7f8fa;
      color: #1f2937;
      margin: 0;
      padding: 24px 28px 48px;
    }}
    h1, h2, h3 {{ margin: 0 0 10px; }}
    h1 {{ font-size: 26px; }}
    h2 {{ margin-top: 28px; font-size: 20px; }}
    h3 {{ margin-top: 18px; font-size: 16px; color: #374151; }}
    .subtle {{ color: #6b7280; margin-bottom: 12px; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(180px, 1fr));
      gap: 10px;
      margin: 14px 0 20px;
    }}
    .kpi {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 12px 14px;
    }}
    .kpi .label {{ font-size: 12px; color: #6b7280; }}
    .kpi .value {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
    .card {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 14px 16px;
      margin-top: 14px;
    }}
    .metric-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: white;
    }}
    .metric-table th, .metric-table td {{
      border: 1px solid #e5e7eb;
      padding: 6px 8px;
      text-align: left;
      white-space: nowrap;
    }}
    .metric-table th {{
      background: #f3f4f6;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    .table-wrap {{
      overflow-x: auto;
      max-height: 520px;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      background: white;
    }}
    .sep {{ height: 8px; }}
  </style>
</head>
<body>
  <h1>Analyse approfondie des phases de marche</h1>
  <div class="subtle">
    Run <b>{run_files.timestamp}</b> | Période: <b>{run_start}</b> → <b>{run_end}</b> | Fichier source: <code>{run_files.stats.name}</code>
  </div>

  <div class="kpi-grid">
    <div class="kpi"><div class="label">Jours analysés</div><div class="value">{total_days}</div></div>
    <div class="kpi"><div class="label">Total return</div><div class="value">{overall_ret:.2%}</div></div>
    <div class="kpi"><div class="label">CAGR</div><div class="value">{overall_cagr:.2%}</div></div>
    <div class="kpi"><div class="label">Sharpe</div><div class="value">{overall_sharpe:.2f}</div></div>
    <div class="kpi"><div class="label">Worst drawdown</div><div class="value">{overall_dd:.2%}</div></div>
  </div>

  <h2>1) Vue d'ensemble temporelle</h2>
  <div class="card">{figures[0]}</div>

  <h2>2) Rendements conditionnels par phase</h2>
  <div class="card">{figures[1]}</div>

  <h2>3) Exposition engagee et turnover par phase</h2>
  <div class="card">{figures[2]}</div>

  <h2>4) Transitions entre phases</h2>
  <div class="card">{figures[3]}</div>

  <h2>5) Metriques detaillees par phase</h2>
  <div class="table-wrap">{_format_table(phase_table)}</div>

  <h2>6) Focus risk-off vs risk-on</h2>
  <div class="table-wrap">{_format_table(risk_band_table)}</div>

  <h2>7) Volumes engages et execution (rebalancing)</h2>
  <div class="table-wrap">{_format_table(exposure_table)}</div>

  <h2>8) Episodes de drawdown</h2>
  <h3>Tous les episodes (phase-consecutifs)</h3>
  <div class="table-wrap">{_format_table(dd_episodes, float_fmt="{:.5f}")}</div>
  <h3>Episodes de drawdown > 10 jours</h3>
  <div class="table-wrap">{_format_table(dd_episodes_gt10, float_fmt="{:.5f}")}</div>

  <h2>9) Pires jours (drawdown minimum)</h2>
  <div class="table-wrap">{_format_table(worst_days, float_fmt="{:.5f}")}</div>

  <h2>10) Journal des suspensions / breakers</h2>
  <h3>FAST_DRAWDOWN_BREACH (focus)</h3>
  <div class="table-wrap">{_format_table(fast_cut_events, float_fmt="{:.5f}")}</div>
  <h3>Toutes les suspensions</h3>
  <div class="table-wrap">{_format_table(suspension_events, float_fmt="{:.5f}")}</div>

</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genere un rapport HTML detaille des phases de marche (risk-on/risk-off)."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./results/event_driven"),
        help="Dossier contenant stats_*.csv et rebal_diagnostics_*.csv",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Timestamp du run a analyser (ex: 20260323_165542). Si absent: dernier run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Chemin du rapport HTML. Par defaut: results/event_driven/regime_phase_deep_dive_<ts>.html",
    )
    parser.add_argument(
        "--top-worst-days",
        type=int,
        default=25,
        help="Nombre de pires jours a inclure.",
    )
    parser.add_argument(
        "--price-matrix",
        type=Path,
        default=None,
        help="Matrice de prix pour le benchmark EW (defaut: PROCESSED_DATA_PATH/price_matrix.csv).",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Ne pas tracer le benchmark buy-and-hold equi-poids.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_files = find_run_files(args.results_dir, args.timestamp)
    output = args.output or (args.results_dir / f"regime_phase_deep_dive_{run_files.timestamp}.html")
    path = generate_report(
        run_files,
        output,
        top_n_worst_days=max(5, int(args.top_worst_days)),
        price_matrix_path=args.price_matrix,
        include_benchmark=not args.no_benchmark,
    )
    print(path)


if __name__ == "__main__":
    main()
