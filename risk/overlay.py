from __future__ import annotations

import numpy as np


def apply_regime_weight_filter(weights: dict, risk_snapshot, return_meta: bool = False):
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
        }
        return ({}, empty_meta) if return_meta else {}

    regime_name = getattr(getattr(risk_snapshot, "regime", None), "name", "NORMAL")
    regime_score = float(getattr(risk_snapshot, "regime_score", 0.75) or 0.75)

    # Hard-flat en crise extrême (conservateur pour le tail risk).
    if regime_name in {"CRISIS", "SUSPENDED"}:
        crisis_meta = {
            "applied_scale": 0.0,
            "regime_name": regime_name,
            "regime_score": regime_score,
            "hard_flat": True,
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
    }
    return (filtered, meta) if return_meta else filtered


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
