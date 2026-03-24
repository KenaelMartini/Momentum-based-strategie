# CLI: python -m event_driven.research_risk_train1
# Sweeps bloc risque sur train 1 uniquement (patch config en mémoire, sans éditer config.py).
from __future__ import annotations

import argparse
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import config as cfg


@contextmanager
def _patch_config(updates: dict[str, Any]):
    old = {k: getattr(cfg, k) for k in updates}
    try:
        for k, v in updates.items():
            setattr(cfg, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(cfg, k, v)


def _run_train1(name: str, out_root: Path, patches: dict[str, Any]) -> dict[str, float]:
    logging.disable(logging.CRITICAL)
    with _patch_config(patches):
        from config import INITIAL_CAPITAL, RESEARCH_TRAIN_1_END, RESEARCH_TRAIN_1_START
        from event_driven.engine import EventDrivenEngine

        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        eng = EventDrivenEngine(
            data_path="./data/processed/price_matrix.csv",
            start_date=RESEARCH_TRAIN_1_START,
            end_date=RESEARCH_TRAIN_1_END,
            initial_capital=INITIAL_CAPITAL,
            live_viz=False,
            output_dir=str(out),
            skip_baseline_comparison=True,
            baseline_reference_path=None,
        )
        eng.run()
        m = eng.final_metrics
        return {
            "cagr": float(m["cagr"]),
            "sharpe": float(m["sharpe"]),
            "max_dd": float(m["max_dd"]),
            "turnover_ann": float(m["avg_turnover"]),
            "n_trades": int(m["n_trades"]),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweeps risque train 1 (patch config RAM).")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./results/risk_train1_plan"),
        help="Dossier racine des sous-dossiers par scénario",
    )
    args = parser.parse_args()
    root: Path = args.output_root
    root.mkdir(parents=True, exist_ok=True)

    scenarios: list[tuple[str, dict[str, Any]]] = [
        ("ref_baseline_ram", {}),
        (
            "p1_crisis_hardflat_true",
            {"REGIME_WEIGHT_FILTER_CRISIS_HARD_FLAT": True},
        ),
        ("p2_maxdd_018", {"MAX_PORTFOLIO_DRAWDOWN": 0.18}),
        ("p2_maxdd_022", {"MAX_PORTFOLIO_DRAWDOWN": 0.22}),
        (
            "p2_susp_cooldown_28",
            {"SUSPENSION_COOLDOWN_CALENDAR_DAYS": 28},
        ),
        (
            "p2_reentry_dd_from_exit_004",
            {"SUSPENSION_REENTRY_DD_FROM_EXIT": -0.04},
        ),
        (
            "p3_risk_informed_on",
            {"RISK_INFORMED_EXPOSURE_TILT_ENABLED": True},
        ),
        (
            "p3_market_overlay_risk_off",
            {
                "ENABLE_MARKET_OVERLAY": True,
                "MARKET_OVERLAY_MODE": "risk_off_only",
            },
        ),
        ("p4_fast_dd_on", {"FAST_DRAWDOWN_CUT_ENABLED": True}),
        ("p4_defensive_flat_on", {"DEFENSIVE_FLAT_ENABLED": True}),
        ("p4_prolonged_underwater_on", {"PROLONGED_UNDERWATER_ENABLED": True}),
    ]

    print("scenario,cagr,sharpe,max_dd,turnover_ann,n_trades")
    for name, patches in scenarios:
        m = _run_train1(name, root, patches)
        print(
            f"{name},{m['cagr']:.6f},{m['sharpe']:.6f},{m['max_dd']:.6f},"
            f"{m['turnover_ann']:.6f},{m['n_trades']}"
        )


if __name__ == "__main__":
    main()
