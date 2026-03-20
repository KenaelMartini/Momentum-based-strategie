from __future__ import annotations

from typing import Optional

try:
    from config import (
        REBALANCE_THRESHOLD_BY_MARKET_REGIME,
        REBALANCE_THRESHOLD_DEFAULT,
    )
except ImportError:
    REBALANCE_THRESHOLD_DEFAULT = 0.030
    REBALANCE_THRESHOLD_BY_MARKET_REGIME = {
        "TREND": REBALANCE_THRESHOLD_DEFAULT,
        "RISK_ON": 0.040,
        "TRANSITION": REBALANCE_THRESHOLD_DEFAULT,
        "RISK_OFF": REBALANCE_THRESHOLD_DEFAULT,
    }


def resolve_market_rebalance_threshold(market_regime_state: Optional[str]) -> tuple[float, str]:
    """
    Return the rebalancing threshold associated with the market regime.
    Falls back to the default threshold when the regime is unknown.
    """
    state = str(market_regime_state or "").strip().upper()
    if not state:
        return float(REBALANCE_THRESHOLD_DEFAULT), "DEFAULT"

    threshold = REBALANCE_THRESHOLD_BY_MARKET_REGIME.get(state, REBALANCE_THRESHOLD_DEFAULT)
    return float(threshold), state
