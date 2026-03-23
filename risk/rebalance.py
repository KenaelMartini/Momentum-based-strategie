from __future__ import annotations

from typing import Optional

from config import REBALANCE_THRESHOLD_DEFAULT


def resolve_market_rebalance_threshold(market_regime_state: Optional[str]) -> tuple[float, str]:
    """
    Seuil de rebalancement (unique, depuis config).
    La signature est conservée pour compatibilité avec l'event-driven.
    """
    _ = market_regime_state
    return float(REBALANCE_THRESHOLD_DEFAULT), "DEFAULT"
