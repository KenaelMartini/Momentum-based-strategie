# CLI : python -m event_driven
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

from config import (
    BACKTEST_END,
    BACKTEST_START,
    EVENT_DRIVEN_BASELINE_JSON,
    EVENT_DRIVEN_BASELINE_JSON_TRAIN1,
    INITIAL_CAPITAL,
    RESEARCH_OOS_AFTER_TRAIN_1_END,
    RESEARCH_OOS_AFTER_TRAIN_1_START,
    RESEARCH_TRAIN_1_END,
    RESEARCH_TRAIN_1_START,
)

from .engine import EventDrivenEngine


def _configure_logging() -> None:
    warnings.filterwarnings("ignore")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    _configure_logging()
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Backtest event-driven Phase 3")
    parser.add_argument(
        "--start",
        default=BACKTEST_START,
        help=f"Début (défaut: BACKTEST_START = {BACKTEST_START})",
    )
    parser.add_argument(
        "--end",
        default=BACKTEST_END,
        help=f"Fin (défaut: BACKTEST_END = {BACKTEST_END})",
    )
    parser.add_argument(
        "--train1",
        action="store_true",
        help=(
            "Période train 1 recherche : RESEARCH_TRAIN_1_START → RESEARCH_TRAIN_1_END "
            f"({RESEARCH_TRAIN_1_START} → {RESEARCH_TRAIN_1_END}). "
            "Écrase --start / --end."
        ),
    )
    parser.add_argument(
        "--oos1",
        action="store_true",
        help=(
            "Hors échantillon après train 1 : RESEARCH_OOS_AFTER_TRAIN_1_* "
            f"({RESEARCH_OOS_AFTER_TRAIN_1_START} → {RESEARCH_OOS_AFTER_TRAIN_1_END}). "
            "Écrase --start / --end."
        ),
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--data", default="./data/processed/price_matrix.csv")
    parser.add_argument("--output", default="./results/event_driven")
    parser.add_argument(
        "--baseline-json",
        default=None,
        help="Fichier JSON baseline pour save_results (défaut : train1 → "
        f"{EVENT_DRIVEN_BASELINE_JSON_TRAIN1}, sinon {EVENT_DRIVEN_BASELINE_JSON})",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Ne pas comparer aux métriques baseline (ex. run OOS de validation)",
    )
    parser.add_argument(
        "--stress-cost-mult",
        type=float,
        default=None,
        help="Multiplicateur commission + slippage (ex. 1.2 = stress +20 %).",
    )
    parser.add_argument(
        "--rebalance-threshold",
        type=float,
        default=None,
        help=(
            "Surcharge du seuil de rebalance (sinon REBALANCE_THRESHOLD_DEFAULT dans config). "
            "Utile pour sweeps train 1 sans modifier config.py."
        ),
    )
    parser.add_argument(
        "--skip-strategy-benchmark-report",
        action="store_true",
        help="Ne pas générer strategy_vs_benchmark_*.html (comparaison EW) à la fin du run.",
    )
    args = parser.parse_args()

    if args.train1 and args.oos1:
        parser.error("Utilise --train1 ou --oos1, pas les deux.")
    if args.train1:
        args.start = RESEARCH_TRAIN_1_START
        args.end = RESEARCH_TRAIN_1_END
    elif args.oos1:
        args.start = RESEARCH_OOS_AFTER_TRAIN_1_START
        args.end = RESEARCH_OOS_AFTER_TRAIN_1_END

    print("=" * 60)
    print("  PHASE 3 — BACKTEST EVENT-DRIVEN")
    print("=" * 60)
    print(f"  Période   : {args.start} → {args.end}")
    print(f"  Capital   : ${INITIAL_CAPITAL:,.0f}")
    print(f"  Visu live : {'OUI' if args.live else 'NON'}")
    if args.skip_baseline or (args.oos1 and args.baseline_json is None):
        print(
            "  Baseline  : ignorée"
            + (" (--skip-baseline)" if args.skip_baseline else " (OOS par défaut)")
        )
    elif args.baseline_json:
        print(f"  Baseline  : {args.baseline_json}")
    elif args.train1:
        print(f"  Baseline  : {EVENT_DRIVEN_BASELINE_JSON_TRAIN1}")
    else:
        print(f"  Baseline  : {EVENT_DRIVEN_BASELINE_JSON}")
    if args.stress_cost_mult is not None:
        print(f"  Stress coûts: ×{args.stress_cost_mult:g}")
    if args.rebalance_threshold is not None:
        print(f"  Seuil rebal. : {args.rebalance_threshold:g} (--rebalance-threshold)")
    print("=" * 60)

    skip_bl = bool(args.skip_baseline)
    baseline_path: Path | None = None
    if not skip_bl:
        if args.oos1 and args.baseline_json is None:
            skip_bl = True
        elif args.baseline_json is not None:
            baseline_path = Path(args.baseline_json)
        elif args.train1:
            baseline_path = Path(EVENT_DRIVEN_BASELINE_JSON_TRAIN1)
        else:
            baseline_path = Path(EVENT_DRIVEN_BASELINE_JSON)

    engine = EventDrivenEngine(
        data_path=args.data,
        start_date=args.start,
        end_date=args.end,
        initial_capital=INITIAL_CAPITAL,
        live_viz=args.live,
        output_dir=args.output,
        baseline_reference_path=baseline_path,
        skip_baseline_comparison=skip_bl,
        transaction_cost_stress_multiplier=args.stress_cost_mult,
        rebalance_threshold=args.rebalance_threshold,
        skip_strategy_benchmark_report=args.skip_strategy_benchmark_report,
    )

    engine.run()

    print("\n  Génération de la visualisation 3D...")
    files = engine.save_results()

    print("\n" + "=" * 60)
    print("  FICHIERS GÉNÉRÉS")
    print("=" * 60)
    for key, path in files.items():
        if path:
            print(f"  {key:12s} : {path}")
    if files.get("dashboard"):
        print("\n  Ouvre dans ton navigateur :")
        print(f"  {files['dashboard']}")


if __name__ == "__main__":
    main()
