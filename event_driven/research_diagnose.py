# CLI : python -m event_driven.research_diagnose path/to/stats_*.csv
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _col_bool(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        return s.astype(str).str.lower().isin(("true", "1", "yes"))
    return s.astype(bool)


def diagnose_stats_csv(path: Path) -> dict:
    df = pd.read_csv(path, parse_dates=["date"])
    out: dict = {}

    n = len(df)
    out["rows"] = n
    if n == 0:
        return out

    if "portfolio_value" in df.columns:
        pv = df["portfolio_value"].astype(float)
        out["pv_first"] = float(pv.iloc[0])
        out["pv_last"] = float(pv.iloc[-1])
        out["total_return"] = float(pv.iloc[-1] / max(pv.iloc[0], 1e-12) - 1.0)

    if "drawdown" in df.columns:
        dd = df["drawdown"].astype(float)
        out["max_drawdown"] = float(dd.min())
        out["median_drawdown"] = float(dd.median())

    if "turnover" in df.columns:
        to = df["turnover"].astype(float)
        out["mean_daily_turnover"] = float(to.mean())
        out["annualized_turnover_approx"] = float(to.mean() * 252)

    if "trading_suspended" in df.columns:
        ts = _col_bool(df["trading_suspended"])
        out["days_trading_suspended"] = int(ts.sum())

    if "rebalancing_day" in df.columns:
        rb = _col_bool(df["rebalancing_day"])
        out["rebalance_days"] = int(rb.sum())

    if "n_orders" in df.columns:
        no = df["n_orders"].fillna(0).astype(float)
        rb = _col_bool(df["rebalancing_day"]) if "rebalancing_day" in df.columns else pd.Series(True, index=df.index)
        sub = df.loc[rb & (no == 0)]
        out["rebalance_days_zero_orders"] = int(len(sub))

    if "gross_after_old_regime_filter" in df.columns:
        g = df["gross_after_old_regime_filter"].astype(float)
        rb = _col_bool(df["rebalancing_day"]) if "rebalancing_day" in df.columns else pd.Series(True, index=df.index)
        zero_gross_rb = int((rb & (g.abs() < 1e-9)).sum())
        out["rebalance_days_gross_old_regime_zero"] = zero_gross_rb

    if "risk_regime_name" in df.columns:
        out["risk_regime_counts"] = df["risk_regime_name"].value_counts().head(8).to_dict()

    if "signal_generation_reason" in df.columns:
        out["signal_reason_counts"] = df["signal_generation_reason"].value_counts().head(6).to_dict()

    return out


def format_report(path: Path, d: dict) -> str:
    lines = [f"Diagnostic : {path}", "=" * 50]
    for k, v in d.items():
        if k.endswith("_counts") and isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                lines.append(f"  {kk}: {vv}")
        else:
            lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("Leviers prioritaires (à valider sur train 1) :")
    levers = []
    if d.get("annualized_turnover_approx", 0) > 6.0:
        levers.append(
            "1) Turnover annualisé élevé -> seuil rebalance (REBALANCE_THRESHOLD_DEFAULT) ou coûts stress."
        )
    if d.get("rebalance_days_gross_old_regime_zero", 0) > 5 or d.get("rebalance_days_zero_orders", 0) > 5:
        levers.append(
            "2) Rebalances sans ordres / gross filtre régime nul -> risk overlay (CRISIS), alignement régime."
        )
    if d.get("days_trading_suspended", 0) > 10:
        levers.append("3) Nombreux jours trading_suspended → circuit breaker / réentrée.")
    if d.get("max_drawdown", 0) < -0.18:
        levers.append("4) Drawdown profond → vol cible, levier, ou composition univers.")
    if not levers:
        levers.append("(Aucun levier automatique fort détecté ; affiner à la main.)")
    lines.extend(levers)
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(
        description="Résumé diagnostic CSV stats event_driven (train 1 / research)."
    )
    parser.add_argument("stats_csv", type=Path, help="Chemin vers stats_*.csv")
    args = parser.parse_args()
    p = args.stats_csv
    if not p.exists():
        print(f"Fichier introuvable : {p}", file=sys.stderr)
        sys.exit(1)
    d = diagnose_stats_csv(p)
    print(format_report(p, d))


if __name__ == "__main__":
    main()
