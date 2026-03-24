from __future__ import annotations

import numpy as np


def apply_regime_weight_filter(
    weights: dict,
    risk_snapshot,
    return_meta: bool = False,
    aligned_market_regime: str = "",
):
    """
    Legacy risk overlay driven by the event-driven risk manager state.
    Returns filtered weights and optional metadata for diagnostics.
    """
    if not weights:
        empty_meta = {
            "applied_scale": 1.0,
            "regime_name": getattr(getattr(risk_snapshot, "regime", None), "name", "NORMAL"),
            "regime_score": float(getattr(risk_snapshot, "regime_score", 0.75) or 0.75),
            "hard_flat": False,
            "trend_tilt_mult": 1.0,
        }
        return ({}, empty_meta) if return_meta else {}

    regime_name = getattr(getattr(risk_snapshot, "regime", None), "name", "NORMAL")
    regime_score = float(getattr(risk_snapshot, "regime_score", 0.75) or 0.75)

    try:
        import config as _cfg

        crisis_hard_flat = bool(getattr(_cfg, "REGIME_WEIGHT_FILTER_CRISIS_HARD_FLAT", True))
    except ImportError:
        crisis_hard_flat = True

    # Suspension explicite : toujours flat.
    if regime_name == "SUSPENDED":
        crisis_meta = {
            "applied_scale": 0.0,
            "regime_name": regime_name,
            "regime_score": regime_score,
            "hard_flat": True,
            "trend_tilt_mult": 1.0,
        }
        return ({}, crisis_meta) if return_meta else {}

    # CRISIS : hard-flat optionnel (tail risk prod) vs scaling continu (recherche momentum).
    if regime_name == "CRISIS" and crisis_hard_flat:
        crisis_meta = {
            "applied_scale": 0.0,
            "regime_name": regime_name,
            "regime_score": regime_score,
            "hard_flat": True,
            "trend_tilt_mult": 1.0,
        }
        return ({}, crisis_meta) if return_meta else {}

    # Solution B — scale continu basé sur `regime_score`.
    # On vise à approximer l'ancien comportement discret (base scale par régime
    # + multipliers par paliers sur regime_score) mais sans les “jumps”,
    # ce qui rend les transitions plus lisses et réduit le churn autour des frontières.
    s = float(np.clip(regime_score, 0.0, 1.0))

    # Points de contrôle (issus du comportement précédent, à différentes valeurs de `regime_score`)
    # -> interpolation linéaire entre ces points.
    control_points = [
        (0.00, 0.00),
        (0.30, 0.30),
        (0.35, 0.45),
        (0.50, 0.54),
        (0.70, 0.92),
        (0.80, 1.05),
        (0.85, 1.1025),
        (1.00, 1.1025),
    ]

    # Interpolation piecewise linéaire.
    scale = float(control_points[0][1])
    for (x0, y0), (x1, y1) in zip(control_points[:-1], control_points[1:]):
        if x0 <= s <= x1:
            if x1 - x0 <= 1e-12:
                scale = float(y1)
            else:
                t = (s - x0) / (x1 - x0)
                scale = float(y0 + t * (y1 - y0))
            break

    trend_tilt_mult = 1.0
    try:
        from config import (
            TREND_DRAWDOWN_TILT_ENABLED,
            TREND_DD_MULT_LE12PCT,
            TREND_DD_MULT_LE8PCT,
            TREND_DD_MULT_LE5PCT,
            TREND_RECOVERY_MAX_DD_FOR_BOOST,
            TREND_RECOVERY_MIN_REGIME_SCORE,
            TREND_RECOVERY_MULT,
        )
    except ImportError:
        TREND_DRAWDOWN_TILT_ENABLED = False

    if TREND_DRAWDOWN_TILT_ENABLED and str(aligned_market_regime).strip().upper() == "TREND":
        cd = float(getattr(risk_snapshot, "current_drawdown", 0.0) or 0.0)
        rs = float(getattr(risk_snapshot, "regime_score", 1.0) or 1.0)
        if cd <= -0.12:
            trend_tilt_mult = float(TREND_DD_MULT_LE12PCT)
        elif cd <= -0.08:
            trend_tilt_mult = float(TREND_DD_MULT_LE8PCT)
        elif cd <= -0.05:
            trend_tilt_mult = float(TREND_DD_MULT_LE5PCT)
        elif cd > float(TREND_RECOVERY_MAX_DD_FOR_BOOST) and rs >= float(TREND_RECOVERY_MIN_REGIME_SCORE):
            trend_tilt_mult = float(TREND_RECOVERY_MULT)
        scale = float(np.clip(scale * trend_tilt_mult, 0.0, 1.30))

    filtered = {}
    for ticker, weight in weights.items():
        new_weight = weight * scale
        if abs(new_weight) > 1e-6:
            filtered[ticker] = new_weight

    meta = {
        "applied_scale": float(scale),
        "regime_name": regime_name,
        "regime_score": regime_score,
        "hard_flat": False,
        "trend_tilt_mult": float(trend_tilt_mult),
    }
    return (filtered, meta) if return_meta else filtered


def apply_risk_informed_exposure_tilt(
    weights: dict,
    aligned_market_regime: str,
    risk_regime_name: str,
    return_meta: bool = False,
):
    """
    Deuxième couche explicite : réduit l’exposition quand le régime aligné / risque
    le commande, sans dépendre de ENABLE_MARKET_OVERLAY.
    """
    meta = {"informed_tilt_scale": 1.0, "informed_tilt_reason": "none"}
    if not weights:
        return ({}, meta) if return_meta else {}

    try:
        import config as _cfg

        RISK_INFORMED_EXPOSURE_TILT_ENABLED = bool(
            getattr(_cfg, "RISK_INFORMED_EXPOSURE_TILT_ENABLED", False)
        )
        RISK_INFORMED_SCALE_RISK_OFF = float(getattr(_cfg, "RISK_INFORMED_SCALE_RISK_OFF", 1.0))
        RISK_INFORMED_SCALE_TRANSITION_UNDER_STRESS = float(
            getattr(_cfg, "RISK_INFORMED_SCALE_TRANSITION_UNDER_STRESS", 1.0)
        )
        RISK_INFORMED_SCALE_TREND_UNDER_STRESS = float(
            getattr(_cfg, "RISK_INFORMED_SCALE_TREND_UNDER_STRESS", 1.0)
        )
    except ImportError:
        return (dict(weights), meta) if return_meta else dict(weights)

    if not RISK_INFORMED_EXPOSURE_TILT_ENABLED:
        return (dict(weights), meta) if return_meta else dict(weights)

    scale = 1.0
    reason = "none"
    am = str(aligned_market_regime or "").strip().upper()
    rr = str(risk_regime_name or "").strip().upper()

    if am == "RISK_OFF":
        scale = float(RISK_INFORMED_SCALE_RISK_OFF)
        reason = "aligned_risk_off"
    elif am == "TRANSITION" and rr == "STRESS":
        scale = float(RISK_INFORMED_SCALE_TRANSITION_UNDER_STRESS)
        reason = "transition_under_stress"
    elif am == "TREND" and rr == "STRESS":
        scale = float(RISK_INFORMED_SCALE_TREND_UNDER_STRESS)
        reason = "trend_under_stress"

    meta = {"informed_tilt_scale": float(scale), "informed_tilt_reason": reason}
    if scale >= 0.999:
        return (dict(weights), meta) if return_meta else dict(weights)

    out = {t: w * scale for t, w in weights.items() if abs(w * scale) > 1e-6}
    return (out, meta) if return_meta else out


def apply_market_regime_overlay(weights: dict, overlay_decision) -> dict:
    """
    Market-state overlay from the new modular regime engine.
    In the current audit configuration this usually acts as pass-through.
    """
    if not weights:
        return {}

    scale = float(getattr(overlay_decision, "scale", 1.0) or 1.0)
    if scale >= 0.999:
        return dict(weights)

    filtered = {}
    for ticker, weight in weights.items():
        new_weight = weight * scale
        if abs(new_weight) > 1e-6:
            filtered[ticker] = new_weight

    return filtered
