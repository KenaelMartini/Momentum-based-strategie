from risk.overlay import apply_risk_informed_exposure_tilt
from risk.regime_alignment import align_market_regime_with_risk


def test_align_crisis_maps_to_risk_off():
    eff, reason = align_market_regime_with_risk("TREND", "CRISIS")
    assert eff == "RISK_OFF"
    assert "risk" in reason


def test_align_suspended_maps_to_risk_off():
    eff, reason = align_market_regime_with_risk("TREND", "SUSPENDED")
    assert eff == "RISK_OFF"
    assert reason == "risk_suspended"


def test_align_stress_downgrades_risk_on():
    eff, _ = align_market_regime_with_risk("RISK_ON", "STRESS")
    assert eff == "TRANSITION"


def test_align_stress_keeps_trend_with_distinct_reason():
    eff, reason = align_market_regime_with_risk("TREND", "STRESS")
    assert eff == "TREND"
    assert reason == "stress_keep_trend"


def test_informed_tilt_trend_under_stress(monkeypatch):
    import config as cfg

    monkeypatch.setattr(cfg, "RISK_INFORMED_EXPOSURE_TILT_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "RISK_INFORMED_SCALE_TREND_UNDER_STRESS", 0.95, raising=False)
    out, meta = apply_risk_informed_exposure_tilt(
        {"AAPL": 1.0},
        "TREND",
        "STRESS",
        return_meta=True,
    )
    assert meta["informed_tilt_reason"] == "trend_under_stress"
    assert abs(out["AAPL"] - 0.95) < 1e-9
