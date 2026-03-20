from types import SimpleNamespace

import pandas as pd

from event_driven_risk import EventDrivenRiskManager, MarketRegime
from risk.overlay import apply_market_regime_overlay, apply_regime_weight_filter


def test_legacy_overlay_flats_suspended_regime():
    snapshot = SimpleNamespace(
        regime=SimpleNamespace(name="SUSPENDED"),
        regime_score=0.0,
    )

    filtered, meta = apply_regime_weight_filter(
        {"AAPL": 0.20, "MSFT": -0.10},
        snapshot,
        return_meta=True,
    )

    assert filtered == {}
    assert meta["hard_flat"] is True
    assert meta["regime_name"] == "SUSPENDED"


def test_market_overlay_scales_weights():
    scaled = apply_market_regime_overlay(
        {"AAPL": 0.20, "MSFT": -0.10},
        SimpleNamespace(scale=0.5),
    )

    assert scaled == {"AAPL": 0.10, "MSFT": -0.05}


def test_risk_manager_reports_suspended_state_and_cooldown():
    manager = EventDrivenRiskManager(initial_capital=100_000)
    prices = pd.Series({"AAPL": 100.0, "MSFT": 200.0})

    manager.update(
        date=pd.Timestamp("2024-01-02"),
        prices=prices,
        portfolio_value=100_000.0,
        current_positions={},
        entry_prices={},
        prev_prices=None,
    )
    breach = manager.update(
        date=pd.Timestamp("2024-01-03"),
        prices=prices,
        portfolio_value=80_000.0,
        current_positions={},
        entry_prices={},
        prev_prices=prices,
    )

    assert breach.regime == MarketRegime.SUSPENDED
    assert breach.trading_suspended is True
    assert breach.suspension_reason == "MAX_DRAWDOWN_BREACH"
    assert breach.risk_scaling == 0.0

    cooldown = manager.update(
        date=pd.Timestamp("2024-01-10"),
        prices=prices,
        portfolio_value=80_000.0,
        current_positions={},
        entry_prices={},
        prev_prices=prices,
    )

    assert cooldown.regime == MarketRegime.SUSPENDED
    assert cooldown.trading_suspended is True
    assert cooldown.suspension_reason == "COOLDOWN_ACTIVE"
    assert cooldown.suspended_days == 7
