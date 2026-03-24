from types import SimpleNamespace

import pandas as pd

import event_driven_risk as edr
from event_driven_risk import EventDrivenRiskManager, MarketRegime
from risk.overlay import apply_market_regime_overlay, apply_regime_weight_filter


def test_regime_weight_filter_crisis_respects_hard_flat_flag(monkeypatch):
    import config as cfg

    monkeypatch.setattr(cfg, "REGIME_WEIGHT_FILTER_CRISIS_HARD_FLAT", True, raising=False)
    snap = SimpleNamespace(regime=SimpleNamespace(name="CRISIS"), regime_score=0.25)
    flat, meta = apply_regime_weight_filter({"AAPL": 0.1}, snap, return_meta=True)
    assert flat == {}
    assert meta["hard_flat"] is True

    monkeypatch.setattr(cfg, "REGIME_WEIGHT_FILTER_CRISIS_HARD_FLAT", False, raising=False)
    scaled, meta2 = apply_regime_weight_filter({"AAPL": 0.1}, snap, return_meta=True)
    assert "AAPL" in scaled
    assert abs(scaled["AAPL"]) > 1e-9
    assert meta2["hard_flat"] is False


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


def test_risk_manager_reports_suspended_state_and_cooldown(monkeypatch):
    monkeypatch.setattr(edr, "MAX_PORTFOLIO_DRAWDOWN", 0.15)
    monkeypatch.setattr(edr, "SUSPENSION_REENTRY_REQUIRE_REGIME_CONFIRMATION", False)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_ENABLED", False)
    monkeypatch.setattr(edr, "SUSPENSION_REENTRY_FAST_CALENDAR_DAYS", 0)
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
        portfolio_value=90_000.0,
        current_positions={},
        entry_prices={},
        prev_prices=prices,
    )

    assert cooldown.regime == MarketRegime.SUSPENDED
    assert cooldown.trading_suspended is True
    assert cooldown.suspension_reason == "COOLDOWN_ACTIVE"
    assert cooldown.suspended_days == 7


def test_risk_manager_fast_drawdown_cut_triggers_suspension(monkeypatch):
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_ENABLED", True)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_ONLY_UNDER_STRESS", False)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_CONFIRM_DAYS", 1)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_THRESHOLD", 0.05)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_WINDOW_DAYS", 2)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_WINDOW_DAYS_RISK_OFF", 2)
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
    fast_cut = manager.update(
        date=pd.Timestamp("2024-01-03"),
        prices=prices,
        portfolio_value=94_500.0,  # -5.5% depuis le pic récent
        current_positions={},
        entry_prices={},
        prev_prices=prices,
    )

    assert fast_cut.regime == MarketRegime.SUSPENDED
    assert fast_cut.trading_suspended is True
    assert fast_cut.suspension_reason == "FAST_DRAWDOWN_BREACH"
    assert fast_cut.risk_scaling == 0.0


def test_risk_manager_fast_drawdown_cut_uses_risk_off_window(monkeypatch):
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_ENABLED", True)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_ONLY_UNDER_STRESS", False)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_CONFIRM_DAYS", 1)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_THRESHOLD", 0.05)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_WINDOW_DAYS", 7)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_WINDOW_DAYS_RISK_OFF", 5)
    manager = EventDrivenRiskManager(initial_capital=100_000)
    prices = pd.Series({"AAPL": 100.0, "MSFT": 200.0})

    manager._risk_regime_current = MarketRegime.STRESS
    values = [100_000.0, 99_500.0, 99_000.0, 98_500.0, 98_000.0, 94_900.0]
    dates = pd.bdate_range("2024-01-02", periods=len(values))

    last = None
    for d, v in zip(dates, values):
        last = manager.update(
            date=pd.Timestamp(d),
            prices=prices,
            portfolio_value=v,
            current_positions={},
            entry_prices={},
            prev_prices=prices,
        )

    assert last is not None
    assert last.trading_suspended is True
    assert last.suspension_reason == "FAST_DRAWDOWN_BREACH"


def test_reentry_requires_allowed_regime_confirmation(monkeypatch):
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_ENABLED", True)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_ONLY_UNDER_STRESS", False)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_CONFIRM_DAYS", 1)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_THRESHOLD", 0.05)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_WINDOW_DAYS", 2)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_WINDOW_DAYS_RISK_OFF", 2)
    manager = EventDrivenRiskManager(initial_capital=100_000)
    prices = pd.Series({"AAPL": 100.0, "MSFT": 200.0})

    monkeypatch.setattr(edr, "SUSPENSION_COOLDOWN_CALENDAR_DAYS", 1)
    monkeypatch.setattr(edr, "SUSPENSION_REENTRY_REQUIRE_REGIME_CONFIRMATION", True)
    monkeypatch.setattr(edr, "SUSPENSION_REENTRY_ALLOWED_RISK_REGIMES", ("NEVER",))
    monkeypatch.setattr(edr, "SUSPENSION_REENTRY_MIN_CONSECUTIVE_RISK_DAYS", 2)

    manager.update(
        date=pd.Timestamp("2024-01-02"),
        prices=prices,
        portfolio_value=100_000.0,
        current_positions={},
        entry_prices={},
        prev_prices=None,
    )
    manager.update(
        date=pd.Timestamp("2024-01-03"),
        prices=prices,
        portfolio_value=94_500.0,
        current_positions={},
        entry_prices={},
        prev_prices=prices,
    )
    blocked = manager.update(
        date=pd.Timestamp("2024-01-05"),
        prices=prices,
        portfolio_value=94_600.0,
        current_positions={},
        entry_prices={},
        prev_prices=prices,
    )

    assert blocked.trading_suspended is True
    assert blocked.suspension_reason == "REENTRY_REGIME_NOT_CONFIRMED"


def test_reentry_ramp_and_post_reentry_guardrail(monkeypatch):
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_ENABLED", True)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_ONLY_UNDER_STRESS", False)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_CONFIRM_DAYS", 1)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_THRESHOLD", 0.05)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_WINDOW_DAYS", 2)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_WINDOW_DAYS_RISK_OFF", 2)
    manager = EventDrivenRiskManager(initial_capital=100_000)
    prices = pd.Series({"AAPL": 100.0, "MSFT": 200.0})

    monkeypatch.setattr(edr, "SUSPENSION_COOLDOWN_CALENDAR_DAYS", 1)
    monkeypatch.setattr(edr, "SUSPENSION_REENTRY_REQUIRE_REGIME_CONFIRMATION", True)
    monkeypatch.setattr(edr, "SUSPENSION_REENTRY_ALLOWED_RISK_REGIMES", ("BULL", "NORMAL", "STRESS", "CRISIS"))
    monkeypatch.setattr(edr, "SUSPENSION_REENTRY_MIN_CONSECUTIVE_RISK_DAYS", 1)
    monkeypatch.setattr(edr, "SUSPENSION_REENTRY_RAMP_ENABLED", True)
    monkeypatch.setattr(edr, "SUSPENSION_REENTRY_RAMP_SCALES", (0.30, 0.60, 1.00))
    monkeypatch.setattr(edr, "SUSPENSION_POST_REENTRY_GUARD_ENABLED", True)
    monkeypatch.setattr(edr, "SUSPENSION_POST_REENTRY_GUARD_CALENDAR_DAYS", 5)
    monkeypatch.setattr(edr, "SUSPENSION_POST_REENTRY_GUARD_DD", -0.02)
    monkeypatch.setattr(edr, "SUSPENSION_POST_REENTRY_RECUT_ENABLED", False)

    manager.update(
        date=pd.Timestamp("2024-01-02"),
        prices=prices,
        portfolio_value=100_000.0,
        current_positions={},
        entry_prices={},
        prev_prices=None,
    )
    manager.update(
        date=pd.Timestamp("2024-01-03"),
        prices=prices,
        portfolio_value=94_500.0,
        current_positions={},
        entry_prices={},
        prev_prices=prices,
    )

    reentry_day = manager.update(
        date=pd.Timestamp("2024-01-05"),
        prices=prices,
        portfolio_value=96_000.0,
        current_positions={},
        entry_prices={},
        prev_prices=prices,
    )
    assert reentry_day.trading_suspended is False
    assert reentry_day.risk_scaling <= 0.45

    guard_breach = manager.update(
        date=pd.Timestamp("2024-01-08"),
        prices=prices,
        portfolio_value=93_000.0,
        current_positions={},
        entry_prices={},
        prev_prices=prices,
    )
    assert guard_breach.trading_suspended is True
    assert guard_breach.suspension_reason == "POST_REENTRY_GUARDRAIL_BREACH"


def test_post_reentry_recut_loss_resuspends(monkeypatch):
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_ENABLED", True)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_ONLY_UNDER_STRESS", False)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_CONFIRM_DAYS", 1)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_THRESHOLD", 0.05)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_THRESHOLD_LONG", 0.99)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_WINDOW_DAYS", 2)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_WINDOW_DAYS_RISK_OFF", 2)
    monkeypatch.setattr(edr, "SUSPENSION_COOLDOWN_CALENDAR_DAYS", 1)
    monkeypatch.setattr(edr, "SUSPENSION_REENTRY_REQUIRE_REGIME_CONFIRMATION", True)
    monkeypatch.setattr(edr, "SUSPENSION_REENTRY_ALLOWED_RISK_REGIMES", ("BULL", "NORMAL", "STRESS", "CRISIS"))
    monkeypatch.setattr(edr, "SUSPENSION_REENTRY_MIN_CONSECUTIVE_RISK_DAYS", 1)
    monkeypatch.setattr(edr, "SUSPENSION_POST_REENTRY_GUARD_ENABLED", False)
    monkeypatch.setattr(edr, "SUSPENSION_POST_REENTRY_RECUT_ENABLED", True)
    monkeypatch.setattr(edr, "SUSPENSION_POST_REENTRY_RECUT_SESSION_DAYS", 5)
    monkeypatch.setattr(edr, "SUSPENSION_POST_REENTRY_RECUT_LOSS", 0.02)

    manager = EventDrivenRiskManager(initial_capital=100_000)
    manager._risk_regime_current = MarketRegime.BULL
    prices = pd.Series({"AAPL": 100.0, "MSFT": 200.0})

    manager.update(
        date=pd.Timestamp("2024-01-02"),
        prices=prices,
        portfolio_value=100_000.0,
        current_positions={},
        entry_prices={},
        prev_prices=None,
    )
    manager.update(
        date=pd.Timestamp("2024-01-03"),
        prices=prices,
        portfolio_value=94_500.0,
        current_positions={},
        entry_prices={},
        prev_prices=prices,
    )
    inv = {"AAPL": 10.0}
    reentry = manager.update(
        date=pd.Timestamp("2024-01-05"),
        prices=prices,
        portfolio_value=96_000.0,
        current_positions=inv,
        entry_prices={},
        prev_prices=prices,
    )
    assert reentry.trading_suspended is False
    manager.update(
        date=pd.Timestamp("2024-01-06"),
        prices=prices,
        portfolio_value=96_000.0,
        current_positions=inv,
        entry_prices={},
        prev_prices=prices,
    )

    recut = manager.update(
        date=pd.Timestamp("2024-01-08"),
        prices=prices,
        portfolio_value=93_500.0,
        current_positions=inv,
        entry_prices={},
        prev_prices=prices,
    )
    assert recut.trading_suspended is True
    assert recut.suspension_reason == "POST_REENTRY_RECUT_LOSS"


def test_post_reentry_recut_after_weekend_uses_sessions_not_calendar_days(monkeypatch):
    """Jeudi réentrée → lundi : l’écart calendaire > 3 ne doit pas désactiver le recut (début janvier)."""
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_ENABLED", True)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_ONLY_UNDER_STRESS", False)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_CONFIRM_DAYS", 1)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_THRESHOLD", 0.05)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_THRESHOLD_LONG", 0.99)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_WINDOW_DAYS", 2)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_WINDOW_DAYS_RISK_OFF", 2)
    monkeypatch.setattr(edr, "SUSPENSION_COOLDOWN_CALENDAR_DAYS", 1)
    monkeypatch.setattr(edr, "SUSPENSION_REENTRY_REQUIRE_REGIME_CONFIRMATION", True)
    monkeypatch.setattr(edr, "SUSPENSION_REENTRY_ALLOWED_RISK_REGIMES", ("BULL", "NORMAL", "STRESS", "CRISIS"))
    monkeypatch.setattr(edr, "SUSPENSION_REENTRY_MIN_CONSECUTIVE_RISK_DAYS", 1)
    monkeypatch.setattr(edr, "SUSPENSION_POST_REENTRY_GUARD_ENABLED", False)
    monkeypatch.setattr(edr, "SUSPENSION_POST_REENTRY_RECUT_ENABLED", True)
    monkeypatch.setattr(edr, "SUSPENSION_POST_REENTRY_RECUT_SESSION_DAYS", 5)
    monkeypatch.setattr(edr, "SUSPENSION_POST_REENTRY_RECUT_LOSS", 0.02)

    manager = EventDrivenRiskManager(initial_capital=100_000)
    manager._risk_regime_current = MarketRegime.BULL
    prices = pd.Series({"AAPL": 100.0, "MSFT": 200.0})

    manager.update(
        date=pd.Timestamp("2022-01-03"),
        prices=prices,
        portfolio_value=100_000.0,
        current_positions={},
        entry_prices={},
        prev_prices=None,
    )
    manager.update(
        date=pd.Timestamp("2022-01-04"),
        prices=prices,
        portfolio_value=94_500.0,
        current_positions={},
        entry_prices={},
        prev_prices=prices,
    )
    inv = {"AAPL": 10.0}
    manager.update(
        date=pd.Timestamp("2022-01-06"),
        prices=prices,
        portfolio_value=96_000.0,
        current_positions=inv,
        entry_prices={},
        prev_prices=prices,
    )
    assert manager.trading_suspended is False
    manager.update(
        date=pd.Timestamp("2022-01-07"),
        prices=prices,
        portfolio_value=96_000.0,
        current_positions=inv,
        entry_prices={},
        prev_prices=prices,
    )

    recut = manager.update(
        date=pd.Timestamp("2022-01-10"),
        prices=prices,
        portfolio_value=93_500.0,
        current_positions=inv,
        entry_prices={},
        prev_prices=prices,
    )
    assert (pd.Timestamp("2022-01-10") - pd.Timestamp("2022-01-06")).days == 4
    assert recut.trading_suspended is True
    assert recut.suspension_reason == "POST_REENTRY_RECUT_LOSS"


def test_rebal_filter_forces_sign_flip_execution(monkeypatch):
    monkeypatch.setattr(edr, "REBALANCE_FORCE_SIGN_FLIP_EXECUTION", True)
    gen = edr.MomentumSignalGeneratorV2(data_handler=None, risk_manager=None)
    out = gen._filter_by_rebal_threshold(
        new_weights={"AAPL": -0.03},
        prev_weights={"AAPL": 0.02},
        threshold=0.10,
        market_regime_state="RISK_ON",
    )
    assert out["AAPL"] < 0.0


def test_risk_off_derisk_only_blocks_exposure_increase(monkeypatch):
    monkeypatch.setattr(edr, "RISK_OFF_ONLY_DERISK_ENABLED", True)
    gen = edr.MomentumSignalGeneratorV2(data_handler=None, risk_manager=None)
    out = gen._filter_by_rebal_threshold(
        new_weights={"AAPL": 0.25, "MSFT": -0.10},
        prev_weights={"AAPL": 0.10, "MSFT": -0.20},
        threshold=0.0,
        market_regime_state="RISK_OFF",
    )
    assert out["AAPL"] == 0.10
    assert out["MSFT"] == -0.10


def test_fast_drawdown_skipped_in_bull_when_only_under_stress(monkeypatch):
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_ENABLED", True)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_ONLY_UNDER_STRESS", True)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_CONFIRM_DAYS", 1)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_THRESHOLD", 0.05)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_THRESHOLD_LONG", 0.99)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_WINDOW_DAYS", 2)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_WINDOW_DAYS_RISK_OFF", 2)
    manager = EventDrivenRiskManager(initial_capital=100_000)
    manager._risk_regime_current = MarketRegime.BULL
    prices = pd.Series({"AAPL": 100.0, "MSFT": 200.0})
    manager.update(
        date=pd.Timestamp("2024-01-02"),
        prices=prices,
        portfolio_value=100_000.0,
        current_positions={},
        entry_prices={},
        prev_prices=None,
    )
    snap = manager.update(
        date=pd.Timestamp("2024-01-03"),
        prices=prices,
        portfolio_value=94_500.0,
        current_positions={},
        entry_prices={},
        prev_prices=prices,
    )
    assert snap.trading_suspended is False
    assert snap.suspension_reason != "FAST_DRAWDOWN_BREACH"


def test_fast_drawdown_long_triggers_in_bull_when_only_under_stress(monkeypatch):
    """Court bloqué en BULL ; long (seuil plus bas) peut encore suspendre."""
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_ENABLED", True)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_ONLY_UNDER_STRESS", True)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_CONFIRM_DAYS", 1)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_THRESHOLD", 0.08)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_THRESHOLD_LONG", 0.055)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_WINDOW_DAYS", 2)
    monkeypatch.setattr(edr, "FAST_DRAWDOWN_CUT_WINDOW_DAYS_RISK_OFF", 2)
    manager = EventDrivenRiskManager(initial_capital=100_000)
    manager._risk_regime_current = MarketRegime.BULL
    prices = pd.Series({"AAPL": 100.0, "MSFT": 200.0})
    manager.update(
        date=pd.Timestamp("2024-01-02"),
        prices=prices,
        portfolio_value=100_000.0,
        current_positions={},
        entry_prices={},
        prev_prices=None,
    )
    snap = manager.update(
        date=pd.Timestamp("2024-01-03"),
        prices=prices,
        portfolio_value=94_500.0,
        current_positions={},
        entry_prices={},
        prev_prices=prices,
    )
    assert snap.trading_suspended is True
    assert snap.suspension_reason == "FAST_DRAWDOWN_BREACH"


def test_regime_net_exposure_target_caps_risk_off_net(monkeypatch):
    monkeypatch.setattr(edr, "REGIME_NET_EXPOSURE_TARGET_ENABLED", True)
    monkeypatch.setattr(edr, "REGIME_NET_TARGET_RISK_OFF_MIN", 0.0)
    monkeypatch.setattr(edr, "REGIME_NET_TARGET_RISK_OFF_MAX", 0.15)
    gen = edr.MomentumSignalGeneratorV2(data_handler=None, risk_manager=None)
    capped = gen._apply_regime_net_exposure_target(
        {"AAPL": 0.30, "MSFT": 0.20, "XOM": -0.05},
        market_regime_state="RISK_OFF",
    )
    assert abs(sum(capped.values()) - 0.15) < 1e-6
