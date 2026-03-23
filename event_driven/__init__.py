# ============================================================
# Package event_driven — backtest jour par jour (Phase 3)
# ============================================================
from __future__ import annotations

from .baseline import compare_with_baseline_reference, evaluate_baseline_verdict
from .broker import COMMISSION_RATE, SLIPPAGE_RATE, SimulatedBroker
from .data_handler import DataHandler
from .engine import EventDrivenEngine
from .events import (
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderType,
    PortfolioStats,
    Signal,
    SignalEvent,
)
from .portfolio import Portfolio
from .signal_generator import MomentumSignalGenerator
from .visualizer import LiveVisualizer3D

__all__ = [
    "COMMISSION_RATE",
    "SLIPPAGE_RATE",
    "DataHandler",
    "EventDrivenEngine",
    "EventType",
    "FillEvent",
    "LiveVisualizer3D",
    "MarketEvent",
    "MomentumSignalGenerator",
    "OrderEvent",
    "OrderType",
    "Portfolio",
    "PortfolioStats",
    "Signal",
    "SignalEvent",
    "SimulatedBroker",
    "compare_with_baseline_reference",
    "evaluate_baseline_verdict",
]
