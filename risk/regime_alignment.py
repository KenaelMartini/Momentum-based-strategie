"""
Aligne le libellé de régime « marché » (TREND / RISK_ON / …) sur le régime risque
discret (BULL / NORMAL / STRESS / CRISIS) pour éviter un RISK_OFF quasi absent
alors que le risk manager voit déjà du stress.
"""

from __future__ import annotations

try:
    from config import ALIGN_MARKET_REGIME_WITH_RISK
except ImportError:
    ALIGN_MARKET_REGIME_WITH_RISK = True


def align_market_regime_with_risk(
    feature_state: str,
    risk_regime_name: str,
) -> tuple[str, str]:
    """
    Retourne (état_effectif, raison courte pour logs / CSV).

    - CRISIS / SUSPENDED → RISK_OFF (hedge informationnel fort)
    - STRESS → RISK_OFF si le modèle était déjà défensif, sinon TRANSITION ;
      TREND reste TREND mais sera couvert par le tilt DD dédié
    - Sinon → état issu du classifieur features
    """
    if not ALIGN_MARKET_REGIME_WITH_RISK:
        return (str(feature_state or "TRANSITION").strip().upper() or "TRANSITION", "align_off")

    fr = str(feature_state or "").strip().upper() or "TRANSITION"
    rr = str(risk_regime_name or "").strip().upper() or "NORMAL"

    if rr in ("CRISIS", "SUSPENDED"):
        return ("RISK_OFF", f"risk_{rr.lower()}")

    if rr == "STRESS":
        if fr in ("RISK_OFF", "TRANSITION"):
            return ("RISK_OFF", "stress_to_risk_off")
        if fr == "RISK_ON":
            return ("TRANSITION", "stress_from_risk_on")
        return (fr, "stress_keep_trend")

    if rr == "NORMAL" and fr == "RISK_ON":
        return ("TRANSITION", "normal_trim_risk_on")

    return (fr, "feature_only")
