# ============================================================
# Types d'événements et stats journalières — backtest event-driven
# ============================================================
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class Signal(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"
    HOLD = "HOLD"


class OrderType(Enum):
    MARKET = "MARKET"


class EventType(Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    STATS = "STATS"


@dataclass
class MarketEvent:
    date: pd.Timestamp
    prices: pd.Series
    event_type: EventType = EventType.MARKET


@dataclass
class SignalEvent:
    date: pd.Timestamp
    weights: dict
    regime: float = 0.0
    regime_state: str = "UNKNOWN"
    regime_confidence: float = 0.0
    signal: Signal = Signal.FLAT
    event_type: EventType = EventType.SIGNAL


@dataclass
class OrderEvent:
    date: pd.Timestamp
    ticker: str
    quantity: float
    price: float
    order_type: OrderType = OrderType.MARKET
    event_type: EventType = EventType.ORDER


@dataclass
class FillEvent:
    date: pd.Timestamp
    ticker: str
    quantity: float
    fill_price: float
    commission: float
    event_type: EventType = EventType.FILL


@dataclass
class PortfolioStats:
    date: pd.Timestamp
    portfolio_value: float
    cash: float
    positions_value: float
    daily_return: float
    realized_vol: float
    expected_vol: float
    drawdown: float
    regime_score: float
    regime_state: str = "UNKNOWN"
    regime_confidence: float = 0.0
    trading_suspended: bool = False
    dd_max_stop: bool = False
    suspension_reason: str = ""
    suspended_days: int = 0
    positions: Optional[dict] = None
    turnover: float = 0.0
    rebalancing_day: bool = False
    n_orders: int = 0
    risk_scaling: float = float("nan")
    rebal_threshold: float = float("nan")
    rebal_threshold_context: str = ""
    signal_generation_reason: str = ""
    gross_signal_raw: float = float("nan")
    gross_after_constraints: float = float("nan")
    gross_after_risk_manager: float = float("nan")
    gross_after_rebal_threshold: float = float("nan")
    gross_after_old_regime_filter: float = float("nan")
    gross_after_market_overlay: float = float("nan")
    old_regime_filter_scale: float = float("nan")
    applied_market_overlay_scale: float = float("nan")
    applied_market_overlay_active: bool = False
    applied_market_overlay_reason: str = ""
    final_turnover: float = float("nan")
    market_regime_feature: str = ""
    market_regime_effective: str = ""
    market_regime_align_reason: str = ""
    risk_regime_name: str = ""
    informed_tilt_scale: float = float("nan")
    informed_tilt_reason: str = ""
    trend_tilt_mult: float = float("nan")
    defensive_flat_phase: str = ""
    defensive_flat_reason: str = ""
