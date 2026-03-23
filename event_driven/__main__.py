# CLI : python -m event_driven
from __future__ import annotations

import argparse
import logging
import sys
import warnings

from config import INITIAL_CAPITAL

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
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--data", default="./data/processed/price_matrix.csv")
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
        data_path=args.data,
        start_date=args.start,
        end_date=args.end,
        initial_capital=INITIAL_CAPITAL,
        live_viz=args.live,
        output_dir=args.output,
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
