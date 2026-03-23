# ============================================================
# Comparaison métriques vs baseline figée (JSON)
# ============================================================
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from config import (
    BASELINE_MAX_CAGR_DEGRADATION,
    BASELINE_MAX_MAX_DD_DEGRADATION,
    BASELINE_MAX_TURNOVER_INCREASE,
    BASELINE_MIN_ACCEPTED_SHARPE,
)

logger = logging.getLogger(__name__)


def evaluate_baseline_verdict(
    baseline_metrics: dict,
    current_metrics: dict,
    delta: dict,
) -> dict:
    baseline_cagr = float(baseline_metrics.get("cagr", 0.0))
    baseline_max_dd = float(baseline_metrics.get("max_drawdown", 0.0))
    baseline_turnover = float(baseline_metrics.get("annualized_turnover", 0.0))

    guardrails = {
        "sharpe_floor": {
            "passed": current_metrics["sharpe"] >= BASELINE_MIN_ACCEPTED_SHARPE,
            "threshold": BASELINE_MIN_ACCEPTED_SHARPE,
            "actual": current_metrics["sharpe"],
        },
        "cagr_drift": {
            "passed": current_metrics["cagr"] >= baseline_cagr - BASELINE_MAX_CAGR_DEGRADATION,
            "threshold": baseline_cagr - BASELINE_MAX_CAGR_DEGRADATION,
            "actual": current_metrics["cagr"],
        },
        "max_dd_drift": {
            "passed": current_metrics["max_drawdown"]
            >= baseline_max_dd - BASELINE_MAX_MAX_DD_DEGRADATION,
            "threshold": baseline_max_dd - BASELINE_MAX_MAX_DD_DEGRADATION,
            "actual": current_metrics["max_drawdown"],
        },
        "turnover_drift": {
            "passed": current_metrics["annualized_turnover"]
            <= baseline_turnover + BASELINE_MAX_TURNOVER_INCREASE,
            "threshold": baseline_turnover + BASELINE_MAX_TURNOVER_INCREASE,
            "actual": current_metrics["annualized_turnover"],
        },
    }

    severe_failures = []
    if delta["sharpe"] < -0.02:
        severe_failures.append("SHARPE_REGRESSION")
    if current_metrics["cagr"] < baseline_cagr - BASELINE_MAX_CAGR_DEGRADATION:
        severe_failures.append("CAGR_REGRESSION")
    if current_metrics["max_drawdown"] < baseline_max_dd - BASELINE_MAX_MAX_DD_DEGRADATION:
        severe_failures.append("MAX_DD_WORSE")
    if current_metrics["annualized_turnover"] > baseline_turnover + BASELINE_MAX_TURNOVER_INCREASE:
        severe_failures.append("TURNOVER_HIGHER")

    if all(item["passed"] for item in guardrails.values()):
        status = "PASS"
        summary = "Toutes les guardrails baseline sont respectees."
    elif severe_failures:
        status = "FAIL"
        summary = ", ".join(severe_failures)
    else:
        status = "WATCH"
        summary = "Iteration proche de la baseline mais hors zone d'acceptation."

    return {
        "status": status,
        "summary": summary,
        "guardrails": guardrails,
        "severe_failures": severe_failures,
    }


def compare_with_baseline_reference(
    final_metrics: dict,
    baseline_path: Path | None = None,
) -> Optional[dict]:
    path = baseline_path or Path("./baseline_event_driven_reference.json")
    if not path.exists() or not final_metrics:
        return None

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger.warning(f"  Impossible de lire la baseline de reference : {exc}")
        return None

    baseline_metrics = payload.get("metrics", {})
    current_metrics = {
        "cagr": float(final_metrics.get("cagr", 0.0)),
        "sharpe": float(final_metrics.get("sharpe", 0.0)),
        "max_drawdown": float(final_metrics.get("max_dd", 0.0)),
        "calmar": float(final_metrics.get("calmar", 0.0)),
        "trade_count": int(final_metrics.get("n_trades", 0)),
        "annualized_turnover": float(final_metrics.get("avg_turnover", 0.0)),
    }
    delta = {
        "cagr": current_metrics["cagr"] - float(baseline_metrics.get("cagr", 0.0)),
        "sharpe": current_metrics["sharpe"] - float(baseline_metrics.get("sharpe", 0.0)),
        "max_drawdown": current_metrics["max_drawdown"]
        - float(baseline_metrics.get("max_drawdown", 0.0)),
        "calmar": current_metrics["calmar"] - float(baseline_metrics.get("calmar", 0.0)),
        "trade_count": current_metrics["trade_count"] - int(baseline_metrics.get("trade_count", 0)),
        "annualized_turnover": current_metrics["annualized_turnover"]
        - float(baseline_metrics.get("annualized_turnover", 0.0)),
    }
    verdict = evaluate_baseline_verdict(
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
        delta=delta,
    )

    logger.info("\n" + "=" * 60)
    logger.info("  COMPARAISON BASELINE vs ITERATION")
    logger.info("=" * 60)
    logger.info(f"  {'Metrique':18s} {'Baseline':>12s} {'Iteration':>12s} {'Delta':>12s}")
    logger.info(f"  {'-' * 58}")
    logger.info(
        f"  {'CAGR':18s} "
        f"{float(baseline_metrics.get('cagr', 0.0)):>11.2%} "
        f"{current_metrics['cagr']:>11.2%} "
        f"{delta['cagr']:>+11.2%}"
    )
    logger.info(
        f"  {'Sharpe':18s} "
        f"{float(baseline_metrics.get('sharpe', 0.0)):>12.3f} "
        f"{current_metrics['sharpe']:>12.3f} "
        f"{delta['sharpe']:>+12.3f}"
    )
    logger.info(
        f"  {'Max DD':18s} "
        f"{float(baseline_metrics.get('max_drawdown', 0.0)):>11.2%} "
        f"{current_metrics['max_drawdown']:>11.2%} "
        f"{delta['max_drawdown']:>+11.2%}"
    )
    logger.info(
        f"  {'Calmar':18s} "
        f"{float(baseline_metrics.get('calmar', 0.0)):>12.3f} "
        f"{current_metrics['calmar']:>12.3f} "
        f"{delta['calmar']:>+12.3f}"
    )
    logger.info(
        f"  {'Nb trades':18s} "
        f"{int(baseline_metrics.get('trade_count', 0)):>12d} "
        f"{current_metrics['trade_count']:>12d} "
        f"{delta['trade_count']:>+12d}"
    )
    logger.info(
        f"  {'Turnover/an':18s} "
        f"{float(baseline_metrics.get('annualized_turnover', 0.0)):>11.1%} "
        f"{current_metrics['annualized_turnover']:>11.1%} "
        f"{delta['annualized_turnover']:>+11.1%}"
    )
    logger.info(f"  {'Verdict':18s} {verdict['status']:>12s} {verdict['summary']}")

    return {
        "baseline_name": payload.get("name", "event_driven_baseline"),
        "baseline_captured_on": payload.get("captured_on"),
        "baseline_metrics": baseline_metrics,
        "current_metrics": current_metrics,
        "delta": delta,
        "verdict": verdict,
    }
