import json
from pathlib import Path

from event_driven import EventDrivenEngine


def _load_frozen_baseline() -> dict:
    baseline_path = Path(__file__).resolve().parents[1] / "baseline_event_driven_reference.json"
    with baseline_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["metrics"]


def test_frozen_baseline_verdict_passes_on_identical_metrics():
    engine = EventDrivenEngine.__new__(EventDrivenEngine)
    baseline = _load_frozen_baseline()
    current = {
        "cagr": float(baseline["cagr"]),
        "sharpe": float(baseline["sharpe"]),
        "max_drawdown": float(baseline["max_drawdown"]),
        "calmar": float(baseline["calmar"]),
        "trade_count": int(baseline["trade_count"]),
        "annualized_turnover": float(baseline["annualized_turnover"]),
    }
    delta = {
        "cagr": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "calmar": 0.0,
        "trade_count": 0,
        "annualized_turnover": 0.0,
    }

    verdict = engine._evaluate_baseline_verdict(
        baseline_metrics=baseline,
        current_metrics=current,
        delta=delta,
    )

    assert verdict["status"] == "PASS"
    assert all(item["passed"] for item in verdict["guardrails"].values())


def test_frozen_baseline_verdict_fails_on_clear_regression():
    engine = EventDrivenEngine.__new__(EventDrivenEngine)
    baseline = _load_frozen_baseline()
    current = {
        "cagr": float(baseline["cagr"]) - 0.01,
        "sharpe": float(baseline["sharpe"]) - 0.05,
        "max_drawdown": float(baseline["max_drawdown"]) - 0.03,
        "calmar": float(baseline["calmar"]) - 0.10,
        "trade_count": int(baseline["trade_count"]) + 40,
        "annualized_turnover": float(baseline["annualized_turnover"]) + 0.40,
    }
    delta = {
        "cagr": current["cagr"] - float(baseline["cagr"]),
        "sharpe": current["sharpe"] - float(baseline["sharpe"]),
        "max_drawdown": current["max_drawdown"] - float(baseline["max_drawdown"]),
        "calmar": current["calmar"] - float(baseline["calmar"]),
        "trade_count": current["trade_count"] - int(baseline["trade_count"]),
        "annualized_turnover": current["annualized_turnover"] - float(baseline["annualized_turnover"]),
    }

    verdict = engine._evaluate_baseline_verdict(
        baseline_metrics=baseline,
        current_metrics=current,
        delta=delta,
    )

    assert verdict["status"] == "FAIL"
    assert "SHARPE_REGRESSION" in verdict["severe_failures"]
    assert "CAGR_REGRESSION" in verdict["severe_failures"]
    assert "MAX_DD_WORSE" in verdict["severe_failures"]
    assert "TURNOVER_HIGHER" in verdict["severe_failures"]
