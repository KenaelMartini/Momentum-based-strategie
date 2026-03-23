"""Machine à états flat défensif + réentrée régime."""

import pandas as pd

from risk.defensive_flat import DefensiveFlatController, DefensiveFlatPhase


def test_defensive_flat_disabled_stays_invested(monkeypatch):
    import config as cfg

    monkeypatch.setattr(cfg, "DEFENSIVE_FLAT_ENABLED", False, raising=False)
    c = DefensiveFlatController()
    r = c.step(
        pd.Timestamp("2020-01-02"),
        "RISK_OFF",
        "STRESS",
        -0.10,
        False,
    )
    assert r.phase == DefensiveFlatPhase.INVESTED
    assert not r.should_hold_flat


def test_defensive_flat_enter_await_reentry(monkeypatch):
    import config as cfg

    monkeypatch.setattr(cfg, "DEFENSIVE_FLAT_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "DEFENSIVE_FLAT_ENTRY_MIN_CONSECUTIVE_DAYS", 2, raising=False)
    monkeypatch.setattr(cfg, "DEFENSIVE_FLAT_ENTRY_MIN_DD", -0.05, raising=False)
    monkeypatch.setattr(cfg, "DEFENSIVE_FLAT_MIN_CALENDAR_DAYS", 0, raising=False)
    monkeypatch.setattr(cfg, "DEFENSIVE_FLAT_REENTRY_EFFECTIVE_CONSECUTIVE", 99, raising=False)
    monkeypatch.setattr(cfg, "DEFENSIVE_FLAT_REENTRY_RISK_CONSECUTIVE", 1, raising=False)

    c = DefensiveFlatController()
    r1 = c.step(pd.Timestamp("2020-01-02"), "RISK_OFF", "STRESS", -0.06, False)
    assert r1.phase == DefensiveFlatPhase.INVESTED

    r2 = c.step(pd.Timestamp("2020-01-03"), "RISK_OFF", "STRESS", -0.06, False)
    assert r2.entered_today
    assert r2.phase == DefensiveFlatPhase.DEFENSIVE_FLAT
    assert r2.should_hold_flat

    r3 = c.step(pd.Timestamp("2020-01-04"), "RISK_OFF", "STRESS", -0.06, False)
    assert r3.phase == DefensiveFlatPhase.AWAIT_REGIME

    r4 = c.step(pd.Timestamp("2020-01-05"), "RISK_OFF", "BULL", -0.01, False)
    assert r4.exited_today
    assert r4.phase == DefensiveFlatPhase.INVESTED
    assert not r4.should_hold_flat


def test_trading_suspended_resets_defensive(monkeypatch):
    import config as cfg

    monkeypatch.setattr(cfg, "DEFENSIVE_FLAT_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "DEFENSIVE_FLAT_ENTRY_MIN_CONSECUTIVE_DAYS", 1, raising=False)
    monkeypatch.setattr(cfg, "DEFENSIVE_FLAT_ENTRY_MIN_DD", -0.01, raising=False)

    c = DefensiveFlatController()
    c.step(pd.Timestamp("2020-01-02"), "RISK_OFF", "STRESS", -0.02, False)
    assert c.is_flat()

    r = c.step(pd.Timestamp("2020-01-03"), "RISK_OFF", "STRESS", -0.02, True)
    assert r.phase == DefensiveFlatPhase.INVESTED
    assert r.reason == "override_trading_suspended"
