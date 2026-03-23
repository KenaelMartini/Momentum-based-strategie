"""
Machine à états : passage en cash défensif (hors circuit breaker) puis réentrée
souple selon le régime marché effectif et/ou le régime risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

try:
    import config as _cfg
except ImportError:
    _cfg = None


class DefensiveFlatPhase(str, Enum):
    INVESTED = "INVESTED"
    """Exposition normale selon le signal."""

    DEFENSIVE_FLAT = "DEFENSIVE_FLAT"
    """Cash forcé ; sous-période initiale (min jours) sans écouter la réentrée."""

    AWAIT_REGIME = "AWAIT_REGIME"
    """Cash ; min jours écoulés — on surveille les régimes pour réentrer."""


@dataclass(frozen=True)
class DefensiveFlatStepResult:
    phase: DefensiveFlatPhase
    reason: str
    entered_today: bool
    exited_today: bool
    should_hold_flat: bool


def _cfg_bool(name: str, default: bool) -> bool:
    if _cfg is None:
        return default
    return bool(getattr(_cfg, name, default))


def _cfg_int(name: str, default: int) -> int:
    if _cfg is None:
        return default
    return int(getattr(_cfg, name, default))


def _cfg_float(name: str, default: float) -> float:
    if _cfg is None:
        return default
    return float(getattr(_cfg, name, default))


def _cfg_set_str(name: str, default: frozenset[str]) -> frozenset[str]:
    if _cfg is None:
        return default
    raw = getattr(_cfg, name, None)
    if raw is None:
        return default
    if isinstance(raw, (frozenset, set)):
        return frozenset(str(x).strip().upper() for x in raw if str(x).strip())
    if isinstance(raw, (list, tuple)):
        return frozenset(str(x).strip().upper() for x in raw if str(x).strip())
    return default


class DefensiveFlatController:
    """
    États : INVESTED → (conditions entrée) → DEFENSIVE_FLAT → (min jours) → AWAIT_REGIME
    → (conditions réentrée) → INVESTED.

    Indépendant du circuit breaker MAX_PORTFOLIO_DRAWDOWN : si trading_suspended, on
    remet la machine à INVESTED pour éviter les doubles logiques.
    """

    def __init__(self) -> None:
        self.phase = DefensiveFlatPhase.INVESTED
        self._entry_streak = 0
        self._flat_start: Optional[pd.Timestamp] = None
        self._days_in_flat = 0
        self._reentry_eff_streak = 0
        self._reentry_risk_streak = 0
        self._last_reason = ""

    def is_flat(self) -> bool:
        return self.phase != DefensiveFlatPhase.INVESTED

    def _reset_streaks(self) -> None:
        self._entry_streak = 0
        self._reentry_eff_streak = 0
        self._reentry_risk_streak = 0

    def _to_invested(self, reason: str) -> DefensiveFlatStepResult:
        prev = self.phase
        self.phase = DefensiveFlatPhase.INVESTED
        self._flat_start = None
        self._days_in_flat = 0
        self._reset_streaks()
        self._last_reason = reason
        return DefensiveFlatStepResult(
            phase=self.phase,
            reason=reason,
            entered_today=False,
            exited_today=prev != DefensiveFlatPhase.INVESTED,
            should_hold_flat=False,
        )

    def step(
        self,
        date: pd.Timestamp,
        effective_regime: str,
        risk_regime: str,
        current_drawdown: float,
        trading_suspended: bool,
    ) -> DefensiveFlatStepResult:
        if not _cfg_bool("DEFENSIVE_FLAT_ENABLED", False):
            if self.phase != DefensiveFlatPhase.INVESTED:
                return self._to_invested("disabled")
            self._last_reason = "disabled"
            return DefensiveFlatStepResult(
                DefensiveFlatPhase.INVESTED, "disabled", False, False, False
            )

        eff = str(effective_regime or "").strip().upper()
        risk = str(risk_regime or "").strip().upper()
        dd = float(current_drawdown)

        if trading_suspended:
            if self.phase != DefensiveFlatPhase.INVESTED:
                return self._to_invested("override_trading_suspended")
            self._reset_streaks()
            self._last_reason = "trading_suspended"
            return DefensiveFlatStepResult(
                DefensiveFlatPhase.INVESTED, "trading_suspended", False, False, False
            )

        entry_regimes = _cfg_set_str("DEFENSIVE_FLAT_ENTRY_EFFECTIVE_REGIMES", frozenset({"RISK_OFF"}))
        entry_min_dd = _cfg_float("DEFENSIVE_FLAT_ENTRY_MIN_DD", -0.06)
        entry_days_need = _cfg_int("DEFENSIVE_FLAT_ENTRY_MIN_CONSECUTIVE_DAYS", 4)

        min_flat = _cfg_int("DEFENSIVE_FLAT_MIN_CALENDAR_DAYS", 3)
        re_eff = _cfg_set_str("DEFENSIVE_FLAT_REENTRY_EFFECTIVE_REGIMES", frozenset({"TREND", "RISK_ON"}))
        re_eff_n = _cfg_int("DEFENSIVE_FLAT_REENTRY_EFFECTIVE_CONSECUTIVE", 2)
        re_risk = _cfg_set_str("DEFENSIVE_FLAT_REENTRY_RISK_REGIMES", frozenset({"BULL", "NORMAL"}))
        re_risk_n = _cfg_int("DEFENSIVE_FLAT_REENTRY_RISK_CONSECUTIVE", 2)

        if self.phase == DefensiveFlatPhase.INVESTED:
            cond = eff in entry_regimes and dd <= entry_min_dd
            if cond:
                self._entry_streak += 1
            else:
                self._entry_streak = 0

            if self._entry_streak >= entry_days_need:
                self.phase = DefensiveFlatPhase.DEFENSIVE_FLAT
                self._flat_start = date
                self._days_in_flat = 0
                self._entry_streak = 0
                self._reentry_eff_streak = 0
                self._reentry_risk_streak = 0
                self._last_reason = "entered_defensive_flat"
                return DefensiveFlatStepResult(
                    self.phase,
                    self._last_reason,
                    True,
                    False,
                    True,
                )

            self._last_reason = "invested"
            return DefensiveFlatStepResult(
                self.phase, self._last_reason, False, False, False
            )

        # --- En flat ---
        if self._flat_start is not None:
            self._days_in_flat = int((date - self._flat_start).days)
        else:
            self._flat_start = date
            self._days_in_flat = 0

        if self.phase == DefensiveFlatPhase.DEFENSIVE_FLAT and self._days_in_flat >= min_flat:
            self.phase = DefensiveFlatPhase.AWAIT_REGIME
            self._reentry_eff_streak = 0
            self._reentry_risk_streak = 0
            self._last_reason = "await_regime"
            return DefensiveFlatStepResult(
                self.phase,
                self._last_reason,
                False,
                False,
                True,
            )

        if self.phase == DefensiveFlatPhase.AWAIT_REGIME:
            if eff in re_eff:
                self._reentry_eff_streak += 1
            else:
                self._reentry_eff_streak = 0
            if risk in re_risk:
                self._reentry_risk_streak += 1
            else:
                self._reentry_risk_streak = 0

            ok_eff = self._reentry_eff_streak >= re_eff_n
            ok_risk = self._reentry_risk_streak >= re_risk_n
            if ok_eff or ok_risk:
                reason = "reentry_effective" if ok_eff else "reentry_risk"
                return self._to_invested(reason)

            self._last_reason = "await_regime"
            return DefensiveFlatStepResult(
                self.phase, self._last_reason, False, False, True
            )

        # DEFENSIVE_FLAT mais encore < min_flat
        self._last_reason = "defensive_flat_hold"
        return DefensiveFlatStepResult(
            self.phase, self._last_reason, False, False, True
        )
