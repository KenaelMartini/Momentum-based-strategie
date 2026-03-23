# ============================================================
# event_driven_risk.py — Système de Risque Event-Driven Complet
# ============================================================
# RÔLE DE CE FICHIER :
# Intégré au package event_driven (engine) — remplace l’ancien risk simplifié.
# par le système complet équivalent à risk_manager.py +
# risk_enhanced.py, adapté à la boucle event-driven.
#
# DIFFÉRENCE FONDAMENTALE VECTORISÉ vs EVENT-DRIVEN :
#   Vectorisé  → calcule TOUT sur toute la période d'un coup (numpy)
#   Event-driven → calcule UN JOUR À LA FOIS, dans l'ordre chronologique
#
# C'est cette contrainte qui nécessite une réécriture :
# on ne peut pas utiliser .rolling() sur l'historique futur —
# on maintient les états manuellement via des deques (files circulaires).
#
# PIPELINE EVENT-DRIVEN :
#   MarketEvent(t) → EventDrivenRiskManager.update(t)
#                 → RiskSnapshot(t) [régime + scaling + stops]
#                 → MomentumSignalGenerator.compute_signal(t)
#                 → PortfolioConstructor.generate_orders(t)
#
# DÉPENDANCES :
#   pip install pandas numpy scipy
# ============================================================

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

# On importe tous les seuils depuis config.py
# → même source de vérité que le vectorisé
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config as _cfg
from config import (
    INITIAL_CAPITAL,
    MAX_PORTFOLIO_DRAWDOWN,  # 0.20 — circuit breaker global
    MAX_POSITION_LOSS,       # 0.15 — stop-loss par position
    MAX_LEVERAGE,            # 1.5
    MAX_POSITION_SIZE,       # 0.10 — cap par actif
    TARGET_VOLATILITY,       # 0.15 — vol cible annualisée
    RISK_FREE_RATE,
    TRANSACTION_COST_BPS,
    SLIPPAGE_BPS,
    MOMENTUM_WEIGHTS,
    SKIP_DAYS,
    LONG_QUANTILE,
    SHORT_QUANTILE,
    REBALANCE_THRESHOLD_DEFAULT,
    PROLONGED_UNDERWATER_ENABLED,
    PROLONGED_UNDERWATER_MIN_DAYS,
    PROLONGED_UNDERWATER_MIN_DD,
    PROLONGED_UNDERWATER_RISK_SCALE_MULT,
    SUSPENSION_COOLDOWN_CALENDAR_DAYS,
    SUSPENSION_REENTRY_DD_FROM_EXIT,
    SUSPENSION_REENTRY_FAST_CALENDAR_DAYS,
    SUSPENSION_REENTRY_FAST_DD_FROM_EXIT,
)

# Optionnels (non présents dans config.py = comportement historique inchangé)
SUSPENSION_REENTRY_REQUIRE_REGIME_CONFIRMATION = getattr(_cfg, "SUSPENSION_REENTRY_REQUIRE_REGIME_CONFIRMATION", False)
SUSPENSION_REENTRY_ALLOWED_RISK_REGIMES = getattr(_cfg, "SUSPENSION_REENTRY_ALLOWED_RISK_REGIMES", ("BULL", "NORMAL"))
SUSPENSION_REENTRY_MIN_CONSECUTIVE_RISK_DAYS = getattr(_cfg, "SUSPENSION_REENTRY_MIN_CONSECUTIVE_RISK_DAYS", 2)
SUSPENSION_REENTRY_RAMP_ENABLED = getattr(_cfg, "SUSPENSION_REENTRY_RAMP_ENABLED", False)
SUSPENSION_REENTRY_RAMP_SCALES = getattr(_cfg, "SUSPENSION_REENTRY_RAMP_SCALES", (0.30, 0.60, 1.00))
SUSPENSION_POST_REENTRY_GUARD_ENABLED = getattr(_cfg, "SUSPENSION_POST_REENTRY_GUARD_ENABLED", False)
SUSPENSION_POST_REENTRY_GUARD_CALENDAR_DAYS = getattr(_cfg, "SUSPENSION_POST_REENTRY_GUARD_CALENDAR_DAYS", 5)
SUSPENSION_POST_REENTRY_GUARD_DD = getattr(_cfg, "SUSPENSION_POST_REENTRY_GUARD_DD", -0.02)
SUSPENSION_POST_REENTRY_RECUT_ENABLED = getattr(_cfg, "SUSPENSION_POST_REENTRY_RECUT_ENABLED", False)
# Nombre de **séances** (appels update / jours de bourse dans le backtest), pas jours calendaires
# (sinon ven. → lun. saute la fenêtre alors qu’une seule séance a passé — typique début janvier).
SUSPENSION_POST_REENTRY_RECUT_SESSION_DAYS = getattr(
    _cfg,
    "SUSPENSION_POST_REENTRY_RECUT_SESSION_DAYS",
    getattr(_cfg, "SUSPENSION_POST_REENTRY_RECUT_CALENDAR_DAYS", 5),
)
SUSPENSION_POST_REENTRY_RECUT_LOSS = getattr(_cfg, "SUSPENSION_POST_REENTRY_RECUT_LOSS", 0.02)
# Si toujours pas de positions après réentrée (attente rebalance mensuel), abandonner le suivi recut.
SUSPENSION_POST_REENTRY_RECUT_MAX_CALENDAR_WAIT_NO_INVEST = getattr(
    _cfg, "SUSPENSION_POST_REENTRY_RECUT_MAX_CALENDAR_WAIT_NO_INVEST", 45
)
REBALANCE_FILL_SAME_BAR = getattr(_cfg, "REBALANCE_FILL_SAME_BAR", False)
REBALANCE_WINDOW_LOSS_CUT_ENABLED = getattr(_cfg, "REBALANCE_WINDOW_LOSS_CUT_ENABLED", False)
REBALANCE_WINDOW_LOSS_CUT_SESSION_DAYS = getattr(_cfg, "REBALANCE_WINDOW_LOSS_CUT_SESSION_DAYS", 5)
REBALANCE_WINDOW_LOSS_CUT_LOSS = getattr(_cfg, "REBALANCE_WINDOW_LOSS_CUT_LOSS", 0.02)
FAST_DRAWDOWN_CUT_ENABLED = getattr(_cfg, "FAST_DRAWDOWN_CUT_ENABLED", False)
FAST_DRAWDOWN_CUT_THRESHOLD = getattr(_cfg, "FAST_DRAWDOWN_CUT_THRESHOLD", 0.05)
FAST_DRAWDOWN_CUT_WINDOW_DAYS = getattr(_cfg, "FAST_DRAWDOWN_CUT_WINDOW_DAYS", 7)
FAST_DRAWDOWN_CUT_WINDOW_DAYS_RISK_OFF = getattr(_cfg, "FAST_DRAWDOWN_CUT_WINDOW_DAYS_RISK_OFF", 5)
# Profil C : n'évalue le fast cut que si le régime risk discret est déjà STRESS/CRISIS
# (état hystérétique en fin de jour précédent — pas BULL/NORMAL).
FAST_DRAWDOWN_CUT_ONLY_UNDER_STRESS = getattr(_cfg, "FAST_DRAWDOWN_CUT_ONLY_UNDER_STRESS", False)
FAST_DRAWDOWN_CUT_THRESHOLD_LONG = getattr(_cfg, "FAST_DRAWDOWN_CUT_THRESHOLD_LONG", 0.07)
FAST_DRAWDOWN_CUT_WINDOW_DAYS_LONG = getattr(_cfg, "FAST_DRAWDOWN_CUT_WINDOW_DAYS_LONG", 10)
FAST_DRAWDOWN_CUT_CONFIRM_DAYS = getattr(_cfg, "FAST_DRAWDOWN_CUT_CONFIRM_DAYS", 2)
RISK_OFF_ONLY_DERISK_ENABLED = getattr(_cfg, "RISK_OFF_ONLY_DERISK_ENABLED", True)
REBALANCE_FORCE_SIGN_FLIP_EXECUTION = getattr(_cfg, "REBALANCE_FORCE_SIGN_FLIP_EXECUTION", True)
REGIME_NET_EXPOSURE_TARGET_ENABLED = getattr(_cfg, "REGIME_NET_EXPOSURE_TARGET_ENABLED", True)
REGIME_NET_TARGET_RISK_OFF_MIN = getattr(_cfg, "REGIME_NET_TARGET_RISK_OFF_MIN", 0.0)
REGIME_NET_TARGET_RISK_OFF_MAX = getattr(_cfg, "REGIME_NET_TARGET_RISK_OFF_MAX", 0.15)
REGIME_NET_TARGET_TRANSITION_MIN = getattr(_cfg, "REGIME_NET_TARGET_TRANSITION_MIN", 0.0)
REGIME_NET_TARGET_TRANSITION_MAX = getattr(_cfg, "REGIME_NET_TARGET_TRANSITION_MAX", 0.40)
REGIME_NET_TARGET_TREND_MIN = getattr(_cfg, "REGIME_NET_TARGET_TREND_MIN", -1.0)
REGIME_NET_TARGET_TREND_MAX = getattr(_cfg, "REGIME_NET_TARGET_TREND_MAX", 1.0)
REGIME_NET_TARGET_RISK_ON_MIN = getattr(_cfg, "REGIME_NET_TARGET_RISK_ON_MIN", -1.0)
REGIME_NET_TARGET_RISK_ON_MAX = getattr(_cfg, "REGIME_NET_TARGET_RISK_ON_MAX", 1.0)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================
# SECTION 1 — TYPES ET STRUCTURES DE DONNÉES
# ============================================================
# On reprend les mêmes Enum que risk_manager.py pour la
# cohérence. Un Enum garantit qu'on ne peut pas écrire
# un régime invalide comme "STREESS".

class MarketRegime(Enum):
    """
    Régimes de marché avec leur facteur d'exposition associé.

    BULL   → 1.00 : exposition pleine, momentum très efficace
    NORMAL → 0.75 : conditions standard légèrement dégradées
    STRESS → 0.50 : vol élevée ou corrélations qui montent
    CRISIS → 0.25 : stress systémique, protection maximale
    """
    BULL      = 1.00
    NORMAL    = 0.75
    STRESS    = 0.50
    CRISIS    = 0.25
    SUSPENDED = 0.00


@dataclass
class RiskSnapshot:
    """
    Rapport de risque complet pour un jour donné.

    C'est l'équivalent event-driven de RiskReport dans risk_manager.py,
    mais enrichi des 4 scores de filtres de risk_enhanced.py.

    Produit par EventDrivenRiskManager.update() à chaque date.
    Consommé par MomentumSignalGenerator et PortfolioConstructor.
    """
    date             : pd.Timestamp

    # ── Scores des 4 filtres (chacun entre 0 et 1) ──────────
    # 1.0 = conditions idéales, 0.0 = crise extrême
    trend_score      : float = 1.0   # MA200 : est-on en bull market ?
    vol_score        : float = 1.0   # Ratio vol court/long
    corr_score       : float = 1.0   # Corrélation moyenne cross-actifs
    dd_score         : float = 1.0   # Circuit breaker sur drawdown portef.

    # ── Score composite (minimum des 4) ──────────────────────
    # On prend le min car le filtre le plus restrictif doit
    # dominer — un seul signal d'alarme suffit pour réduire.
    regime_score_raw : float = 1.0   # avant lissage
    regime_score     : float = 1.0   # après lissage 3j

    # ── Régime classifié ─────────────────────────────────────
    regime           : MarketRegime = MarketRegime.BULL

    # ── Métriques de volatilité ──────────────────────────────
    vol_short        : float = 0.15  # vol portef. 10j annualisée
    vol_long         : float = 0.15  # vol portef. 252j annualisée
    vol_ratio        : float = 1.0   # vol_short / vol_long

    # ── Métriques de drawdown ────────────────────────────────
    current_drawdown : float = 0.0   # drawdown courant (négatif)
    peak_value       : float = 0.0   # valeur maximale historique

    # ── Corrélation ──────────────────────────────────────────
    avg_correlation  : float = 0.0

    # ── Facteur de scaling final ─────────────────────────────
    # vol_scaling × regime_factor, clippé dans [0.10, 1.50]
    risk_scaling     : float = 1.0

    # ── Stop-loss individuels ────────────────────────────────
    positions_to_close : list = field(default_factory=list)

    # ── Circuit breaker global ───────────────────────────────
    trading_suspended : bool = False
    suspension_reason : str = ""
    suspended_days    : int = 0
    dd_max_stop       : bool = False   # alias pour compatibilité

    # ── Alertes texte ────────────────────────────────────────
    alerts : list = field(default_factory=list)

    # ── Sous l'eau prolongé (streak DD<0 + magnitude) ────────
    prolonged_underwater_active: bool = False


# ============================================================
# SECTION 2 — GESTIONNAIRE DE RISQUE EVENT-DRIVEN
# ============================================================

class EventDrivenRiskManager:
    """
    Équivalent event-driven de risk_manager.py + risk_enhanced.py.

    PRINCIPE DE FONCTIONNEMENT :
    Au lieu de calculer sur toute la période d'un coup (vectorisé),
    on maintient des fenêtres glissantes via des deques :
        deque(maxlen=N) → file circulaire de taille N
        → quand on ajoute un élément à droite et que la deque est pleine,
          l'élément le plus ancien sort automatiquement à gauche.

    C'est l'équivalent temps-réel de pandas.Series.rolling(N).

    USAGE :
        rm = EventDrivenRiskManager(initial_capital=100_000)
        for date, market_event in timeline:
            snapshot = rm.update(
                date=date,
                prices=market_event.prices,          # Series actuelle
                portfolio_value=portfolio.value,      # float
                current_positions=portfolio.positions, # dict
                entry_prices=portfolio.entry_prices,  # dict
            )
            # snapshot.risk_scaling → facteur à appliquer aux poids
            # snapshot.positions_to_close → stops à exécuter
    """

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.peak_value      = initial_capital

        # ── Hystérésis de régime (anti-thrashing) ──────────────
        # Le régime risk (BULL/NORMAL/STRESS/CRISIS) peut osciller
        # autour des frontières. On utilise une hystérésis :
        # - quand le régime devient pire (valeur Enum plus basse) :
        #   changement immédiat (protection tail risk).
        # - quand le régime s'améliore : changement seulement après
        #   N jours consécutifs.
        self._risk_regime_current = MarketRegime.BULL
        self._risk_regime_candidate = None
        self._risk_regime_candidate_days = 0
        self._risk_regime_up_days = 2

        # ── Buffers de volatilité du portefeuille ────────────
        # On stocke les rendements journaliers du portef. pour
        # calculer la vol rolling sans refaire tout l'historique.
        self._port_returns_short = deque(maxlen=10)   # 10j  — réactif
        self._port_returns_long  = deque(maxlen=252)  # 252j — baseline

        # ── Buffer de corrélation ────────────────────────────
        # On stocke les rendements cross-actifs (fenêtre 42j)
        self._returns_buffer = deque(maxlen=42)

        # ── Buffer du benchmark (MA200 pour trend filter) ────
        # La MA200 nécessite 200 observations → buffer de 200 jours
        self._benchmark_buffer = deque(maxlen=200)

        # ── Buffer de lissage du regime score ────────────────
        # Lissage sur 3 jours pour éviter le turnover excessif
        # (un régime qui oscille BULL/STRESS/BULL génère des trades inutiles)
        self._regime_buffer = deque(maxlen=3)

        # ── Historique des valeurs du portefeuille ───────────
        self._portfolio_values = deque(maxlen=252)

        # ── État de suspension ───────────────────────────────
        self.trading_suspended = False
        self._prev_portfolio_value = initial_capital

        # ── Compteur de jours de stress ─────────────────────
        self._stress_days = 0

        # ── Jours consécutifs avec drawdown < 0 (sous le pic) ─
        self._underwater_streak = 0
        self._reentry_regime_ok_days = 0
        self._fast_dd_breach_streak = 0

        logger.info(
            f"EventDrivenRiskManager initialisé | "
            f"Capital: {initial_capital:,.0f}$ | "
            f"Max DD: {MAX_PORTFOLIO_DRAWDOWN:.0%} | "
            f"Target vol: {TARGET_VOLATILITY:.0%}"
        )

    def _apply_regime_hysteresis(self, target_regime: MarketRegime) -> MarketRegime:
        """
        Applique une hystérésis simple sur la classification de régime.

        Règles :
        - Si le régime devient pire (valeur plus basse) : switch immédiat.
        - Si le régime devient meilleur : switch après N jours consécutifs.
        - Si le régime courant est SUSPENDED : switch immédiat (après cooldown).
        """
        current = getattr(self, "_risk_regime_current", target_regime)

        # Après SUSPENDED on permet un retour plus rapide (cooldown gère déjà le timing).
        if current == MarketRegime.SUSPENDED:
            self._risk_regime_current = target_regime
            self._risk_regime_candidate = None
            self._risk_regime_candidate_days = 0
            return target_regime

        if target_regime == current:
            self._risk_regime_candidate = None
            self._risk_regime_candidate_days = 0
            return current

        # Régime "pire" = exposition multiplicateur plus faible.
        if target_regime.value < current.value:
            self._risk_regime_current = target_regime
            self._risk_regime_candidate = None
            self._risk_regime_candidate_days = 0
            return target_regime

        # Régime "meilleur" => dwell time.
        if self._risk_regime_candidate != target_regime:
            self._risk_regime_candidate = target_regime
            self._risk_regime_candidate_days = 1
        else:
            self._risk_regime_candidate_days += 1

        if self._risk_regime_candidate_days >= self._risk_regime_up_days:
            self._risk_regime_current = target_regime
            self._risk_regime_candidate = None
            self._risk_regime_candidate_days = 0
            return target_regime

        return current

    def _disarm_rebalance_window_loss_cut(self) -> None:
        for name in (
            "_rbw_anchor",
            "_rbw_sessions",
            "_rbw_worst",
            "_rbw_last_eval_session_date",
            "_rbw_anchor_month",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def _arm_rebalance_window_loss_cut(self, date: pd.Timestamp, portfolio_value: float) -> None:
        if not REBALANCE_WINDOW_LOSS_CUT_ENABLED:
            return
        self._rbw_anchor = float(portfolio_value)
        self._rbw_sessions = 0
        self._rbw_worst = 0.0
        self._rbw_anchor_month = (int(date.year), int(date.month))
        if hasattr(self, "_rbw_last_eval_session_date"):
            del self._rbw_last_eval_session_date

    def _snapshot_suspend_post_reentry(
        self,
        date: pd.Timestamp,
        portfolio_value: float,
        reason: str,
        alert: str,
        log_message: str,
    ) -> RiskSnapshot:
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        current_drawdown = (portfolio_value - self.peak_value) / (self.peak_value + 1e-8)
        self.trading_suspended = True
        self._underwater_streak = 0
        self._fast_dd_breach_streak = 0
        self._suspension_date = date
        self._cash_at_exit = portfolio_value
        self._reentry_regime_ok_days = 0
        self._disarm_rebalance_window_loss_cut()
        snapshot = RiskSnapshot(date=date)
        snapshot.regime = MarketRegime.SUSPENDED
        snapshot.regime_score_raw = 0.0
        snapshot.trading_suspended = True
        snapshot.dd_max_stop = True
        snapshot.risk_scaling = 0.0
        snapshot.regime_score = 0.0
        snapshot.current_drawdown = current_drawdown
        snapshot.peak_value = self.peak_value
        snapshot.suspension_reason = reason
        snapshot.suspended_days = 0
        snapshot.alerts.append(alert)
        logger.critical(log_message)
        return snapshot

    def _eval_post_reentry_recut(
        self,
        date: pd.Timestamp,
        portfolio_value: float,
        current_positions: dict,
        *,
        same_bar_rebalance_followup: bool = False,
    ):
        """
        Évalue le recut post-réentrée (ancre = 1re séance investie, pire clôture vs ancre sur la fenêtre).
        Retourne un RiskSnapshot si suspension, sinon None.
        """
        if not SUSPENSION_POST_REENTRY_RECUT_ENABLED:
            return None
        if self.trading_suspended or not hasattr(self, "_reentry_date"):
            return None
        invested = bool(current_positions) and any(
            abs(float(q)) > 1e-9 for q in current_positions.values()
        )
        if not invested:
            return None
        recut_n = int(max(1, SUSPENSION_POST_REENTRY_RECUT_SESSION_DAYS))
        thr = float(SUSPENSION_POST_REENTRY_RECUT_LOSS)

        # Même jour : le matin update() a déjà incrémenté la séance avec l’ancien book ;
        # après fill rebalance même bar, on ne refait qu’actualiser le pire vs ancre (pas sess+1).
        if same_bar_rebalance_followup and getattr(self, "_reentry_last_eval_session_date", None) == date:
            anchor_v = getattr(self, "_reentry_anchor_value", None)
            if anchor_v is not None:
                anchor = float(anchor_v)
                r = float(portfolio_value) / anchor - 1.0 if anchor > 0 else 0.0
                self._reentry_worst_vs_anchor = min(
                    float(getattr(self, "_reentry_worst_vs_anchor", 0.0)), r
                )
                w = float(self._reentry_worst_vs_anchor)
                sess = int(getattr(self, "_reentry_sessions_since", 0))
                if anchor > 0 and 1 <= sess <= recut_n and w <= -thr:
                    return self._snapshot_suspend_post_reentry(
                        date,
                        portfolio_value,
                        "POST_REENTRY_RECUT_LOSS",
                        "POST_REENTRY_RECUT_TRIGGERED",
                        (
                            f"🚨 {date.date()} | POST-REENTRY RECUT | "
                            f"Pire clôture vs ancre: {w:.1%} <= -{thr:.1%} "
                            f"(séance investie {sess}/{recut_n}, post-rebal) | "
                            f"Trading SUSPENDU — toutes les positions liquidées en cash"
                        ),
                    )
            return None

        if getattr(self, "_reentry_anchor_value", None) is None:
            self._reentry_anchor_value = float(portfolio_value)
            self._reentry_sessions_since = 0
            self._reentry_worst_vs_anchor = 0.0
        self._reentry_sessions_since = int(getattr(self, "_reentry_sessions_since", 0)) + 1
        sess = int(self._reentry_sessions_since)
        anchor = float(self._reentry_anchor_value)
        r = float(portfolio_value) / anchor - 1.0 if anchor > 0 else 0.0
        self._reentry_worst_vs_anchor = min(float(getattr(self, "_reentry_worst_vs_anchor", 0.0)), r)
        w = float(self._reentry_worst_vs_anchor)
        self._reentry_last_eval_session_date = date
        if anchor > 0 and 1 <= sess <= recut_n and w <= -thr:
            return self._snapshot_suspend_post_reentry(
                date,
                portfolio_value,
                "POST_REENTRY_RECUT_LOSS",
                "POST_REENTRY_RECUT_TRIGGERED",
                (
                    f"🚨 {date.date()} | POST-REENTRY RECUT | "
                    f"Pire clôture vs ancre: {w:.1%} <= -{thr:.1%} "
                    f"(séance investie {sess}/{recut_n}) | "
                    f"Trading SUSPENDU — toutes les positions liquidées en cash"
                ),
            )
        return None

    def _eval_rebalance_window_loss_cut(
        self,
        date: pd.Timestamp,
        portfolio_value: float,
        current_positions: dict,
        *,
        same_bar_rebalance_followup: bool = False,
    ):
        """
        Fenêtre de perte après **chaque** rebalance mensuel (ancre = PV post-fill).
        Indépendant d'une suspension / réentrée risk.
        """
        if not REBALANCE_WINDOW_LOSS_CUT_ENABLED:
            return None
        if self.trading_suspended or not hasattr(self, "_rbw_anchor"):
            return None
        am = getattr(self, "_rbw_anchor_month", None)
        if am is not None:
            ym = (int(date.year), int(date.month))
            if ym > am:
                self._disarm_rebalance_window_loss_cut()
                return None
        invested = bool(current_positions) and any(
            abs(float(q)) > 1e-9 for q in current_positions.values()
        )
        if not invested:
            return None
        n = int(max(1, REBALANCE_WINDOW_LOSS_CUT_SESSION_DAYS))
        thr = float(REBALANCE_WINDOW_LOSS_CUT_LOSS)
        anchor = float(self._rbw_anchor)

        if same_bar_rebalance_followup and getattr(self, "_rbw_last_eval_session_date", None) == date:
            r = float(portfolio_value) / anchor - 1.0 if anchor > 0 else 0.0
            self._rbw_worst = min(float(getattr(self, "_rbw_worst", 0.0)), r)
            w = float(self._rbw_worst)
            sess = int(getattr(self, "_rbw_sessions", 0))
            if sess > n:
                self._disarm_rebalance_window_loss_cut()
                return None
            if anchor > 0 and 1 <= sess <= n and w <= -thr:
                return self._snapshot_suspend_post_reentry(
                    date,
                    portfolio_value,
                    "REBALANCE_WINDOW_LOSS_CUT",
                    "REBALANCE_WINDOW_LOSS_TRIGGERED",
                    (
                        f"🚨 {date.date()} | REBALANCE WINDOW LOSS | "
                        f"Pire clôture vs ancre rebalance: {w:.1%} <= -{thr:.1%} "
                        f"(séance {sess}/{n}, post-rebal) | "
                        f"Trading SUSPENDU — toutes les positions liquidées en cash"
                    ),
                )
            return None

        self._rbw_sessions = int(getattr(self, "_rbw_sessions", 0)) + 1
        sess = int(self._rbw_sessions)
        r = float(portfolio_value) / anchor - 1.0 if anchor > 0 else 0.0
        self._rbw_worst = min(float(getattr(self, "_rbw_worst", 0.0)), r)
        w = float(self._rbw_worst)
        self._rbw_last_eval_session_date = date
        if sess > n:
            self._disarm_rebalance_window_loss_cut()
            return None
        if anchor > 0 and 1 <= sess <= n and w <= -thr:
            return self._snapshot_suspend_post_reentry(
                date,
                portfolio_value,
                "REBALANCE_WINDOW_LOSS_CUT",
                "REBALANCE_WINDOW_LOSS_TRIGGERED",
                (
                    f"🚨 {date.date()} | REBALANCE WINDOW LOSS | "
                    f"Pire clôture vs ancre rebalance: {w:.1%} <= -{thr:.1%} "
                    f"(séance {sess}/{n}) | "
                    f"Trading SUSPENDU — toutes les positions liquidées en cash"
                ),
            )
        return None

    def apply_post_rebalance_recut_check(
        self,
        date: pd.Timestamp,
        portfolio_value: float,
        current_positions: dict,
    ):
        """Appelé par le moteur après exécution same-bar des ordres de rebalance."""
        if not REBALANCE_FILL_SAME_BAR:
            return None
        if REBALANCE_WINDOW_LOSS_CUT_ENABLED:
            self._arm_rebalance_window_loss_cut(date, float(portfolio_value))
            snap = self._eval_rebalance_window_loss_cut(
                date, portfolio_value, current_positions, same_bar_rebalance_followup=True
            )
            if snap is not None:
                return snap
        return self._eval_post_reentry_recut(
            date, portfolio_value, current_positions, same_bar_rebalance_followup=True
        )

    # ─────────────────────────────────────────────────────────
    # MÉTHODE PRINCIPALE : update() — à appeler chaque jour
    # ─────────────────────────────────────────────────────────
    def update(
        self,
        date              : pd.Timestamp,
        prices            : pd.Series,
        portfolio_value   : float,
        current_positions : dict,
        entry_prices      : dict,
        prev_prices       : pd.Series = None,
    ) -> RiskSnapshot:
        """
        Met à jour tous les indicateurs de risque pour la date t.

        ORDRE DES CALCULS :
        1. Rendement journalier du portefeuille → volatilité
        2. Mise à jour du peak → drawdown courant
        3. Mise à jour du buffer de corrélation
        4. Calcul des 4 filtres
        5. Score composite + lissage + régime
        6. Vol scaling + regime factor → risk_scaling final
        7. Stop-loss individuels
        8. Circuit breaker global

        Args:
            date              : date courante
            prices            : prix de tous les actifs à t
            portfolio_value   : valeur totale du portefeuille à t
            current_positions : dict { symbol: quantité }
            entry_prices      : dict { symbol: prix_entrée }
            prev_prices       : prix à t-1 (pour les rendements actifs)
                                Si None, on ne met pas à jour la corr.

        Returns:
            RiskSnapshot complet
        """
        snapshot = RiskSnapshot(date=date)

        # ── ÉTAPE 1 : Rendement journalier du portefeuille ───
        # r(t) = V(t) / V(t-1) - 1
        # On utilise le rendement simple ici (pas log) car on
        # travaille avec des valeurs de portefeuille entières.
        if self._prev_portfolio_value > 0:
            daily_ret = portfolio_value / self._prev_portfolio_value - 1
        else:
            daily_ret = 0.0

        self._prev_portfolio_value = portfolio_value
        self._port_returns_short.append(daily_ret)
        self._port_returns_long.append(daily_ret)
        self._portfolio_values.append(portfolio_value)

        # ── ÉTAPE 2 : Drawdown ───────────────────────────────
        # Peak = maximum cumulatif — ne peut que croître ou rester stable
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        current_drawdown = (portfolio_value - self.peak_value) / self.peak_value
        snapshot.current_drawdown = current_drawdown
        snapshot.peak_value       = self.peak_value

        def _compute_fast_dd(window_days: int) -> float:
            w = int(max(1, window_days))
            if len(self._portfolio_values) < 2:
                return 0.0
            arr = np.array(list(self._portfolio_values)[-min(len(self._portfolio_values), (w + 1)) :], dtype=float)
            local_peak = float(np.max(arr))
            if local_peak <= 0:
                return 0.0
            return float(arr[-1] / local_peak - 1.0)

        def _suspend(reason: str, alert: str, log_message: str) -> RiskSnapshot:
            self.trading_suspended = True
            self._underwater_streak = 0
            self._fast_dd_breach_streak = 0
            self._suspension_date = date
            self._cash_at_exit = portfolio_value
            self._reentry_regime_ok_days = 0
            snapshot.regime = MarketRegime.SUSPENDED
            snapshot.regime_score_raw = 0.0
            snapshot.trading_suspended = True
            snapshot.dd_max_stop = True
            snapshot.risk_scaling = 0.0
            snapshot.regime_score = 0.0
            snapshot.current_drawdown = current_drawdown
            snapshot.peak_value = self.peak_value
            snapshot.suspension_reason = reason
            snapshot.suspended_days = 0
            snapshot.alerts.append(alert)
            logger.critical(log_message)
            return snapshot

        # ── ÉTAPE 3 : Buffer rendements cross-actifs ─────────
        # Pour le filtre de corrélation, on a besoin des
        # rendements journaliers de TOUS les actifs.
        # On les calcule ici si prev_prices est fourni.
        if prev_prices is not None and len(prev_prices) > 0:
            common = prices.index.intersection(prev_prices.index)
            if len(common) > 1:
                asset_rets = (prices[common] / prev_prices[common] - 1).fillna(0)
                self._returns_buffer.append(asset_rets.values)

        # Benchmark proxy utilisé par le trend filter.
        bm_value = float(prices.mean())
        self._benchmark_buffer.append(bm_value)

        # ── Post-réentrée : recut (voir _eval_post_reentry_recut) + garde-fou pic local ─
        if (not self.trading_suspended) and hasattr(self, "_reentry_date"):
            days_since_reentry_cal = int((date - self._reentry_date).days)
            recut_n = int(max(1, SUSPENSION_POST_REENTRY_RECUT_SESSION_DAYS))
            max_wait_no_invest = int(max(1, SUSPENSION_POST_REENTRY_RECUT_MAX_CALENDAR_WAIT_NO_INVEST))

            snap_recut = self._eval_post_reentry_recut(date, portfolio_value, current_positions)
            if snap_recut is not None:
                return snap_recut

            if SUSPENSION_POST_REENTRY_GUARD_ENABLED and hasattr(self, "_reentry_peak_value"):
                guard_days = int(max(0, SUSPENSION_POST_REENTRY_GUARD_CALENDAR_DAYS))
                self._reentry_peak_value = max(float(self._reentry_peak_value), float(portfolio_value))
                dd_from_reentry_peak = float(portfolio_value - self._reentry_peak_value) / (
                    float(self._reentry_peak_value) + 1e-8
                )
                if days_since_reentry_cal <= guard_days and dd_from_reentry_peak <= float(
                    SUSPENSION_POST_REENTRY_GUARD_DD
                ):
                    return _suspend(
                        reason="POST_REENTRY_GUARDRAIL_BREACH",
                        alert="POST_REENTRY_GUARDRAIL_TRIGGERED",
                        log_message=(
                            f"🚨 {date.date()} | POST-REENTRY GUARDRAIL | "
                            f"DD post-reentry: {dd_from_reentry_peak:.1%} <= {float(SUSPENSION_POST_REENTRY_GUARD_DD):.1%} | "
                            f"Trading SUSPENDU — toutes les positions liquidées en cash"
                        ),
                    )

            sess = int(getattr(self, "_reentry_sessions_since", 0))
            anchor = getattr(self, "_reentry_anchor_value", None)
            recut_window_done = (not SUSPENSION_POST_REENTRY_RECUT_ENABLED) or (
                anchor is not None and sess > recut_n
            ) or (
                SUSPENSION_POST_REENTRY_RECUT_ENABLED
                and anchor is None
                and days_since_reentry_cal > max_wait_no_invest
            )
            guard_days = int(max(0, SUSPENSION_POST_REENTRY_GUARD_CALENDAR_DAYS))
            guard_window_done = (not SUSPENSION_POST_REENTRY_GUARD_ENABLED) or (
                days_since_reentry_cal > guard_days
            )
            if recut_window_done and guard_window_done:
                del self._reentry_date
                if hasattr(self, "_reentry_peak_value"):
                    del self._reentry_peak_value
                if hasattr(self, "_reentry_anchor_value"):
                    del self._reentry_anchor_value
                if hasattr(self, "_reentry_sessions_since"):
                    del self._reentry_sessions_since
                if hasattr(self, "_reentry_worst_vs_anchor"):
                    del self._reentry_worst_vs_anchor
                if hasattr(self, "_reentry_last_eval_session_date"):
                    del self._reentry_last_eval_session_date

        if not self.trading_suspended:
            snap_rbw = self._eval_rebalance_window_loss_cut(
                date, portfolio_value, current_positions, same_bar_rebalance_followup=False
            )
            if snap_rbw is not None:
                return snap_rbw

        # ── Circuit breaker : déjà suspendu depuis un jour précédent ─
        if self.trading_suspended:
            # CONDITION DE RÉENTRÉE :
            # Quand on est en cash, le drawdown pertinent est calculé
            # depuis la valeur de sortie (cash_at_exit), pas le peak historique.
            # On réentre si le marché a stagné (on est en cash → pas de perte)
            # ET que suffisamment de temps s'est écoulé (cooldown 21j minimum).
            #
            # En pratique : si portfolio_value ≈ cash_at_exit (Vol=0 en cash),
            # le DD depuis cash_at_exit = 0% → toujours > -10%.
            # On ajoute donc un cooldown obligatoire de 21 jours ouvrés.

            if not hasattr(self, '_suspension_date'):
                self._suspension_date  = date
                self._cash_at_exit     = portfolio_value

            days_suspended = (date - self._suspension_date).days
            dd_since_exit  = (portfolio_value - self._cash_at_exit) / (self._cash_at_exit + 1e-8)
            cd = int(SUSPENSION_COOLDOWN_CALENDAR_DAYS)
            r_dd = float(SUSPENSION_REENTRY_DD_FROM_EXIT)
            fd = int(SUSPENSION_REENTRY_FAST_CALENDAR_DAYS)
            f_dd = float(SUSPENSION_REENTRY_FAST_DD_FROM_EXIT)
            allowed_regimes = {str(x).upper() for x in SUSPENSION_REENTRY_ALLOWED_RISK_REGIMES}
            min_regime_days = int(max(1, SUSPENSION_REENTRY_MIN_CONSECUTIVE_RISK_DAYS))

            trend_score = self._compute_trend_score()
            vol_short, vol_long, vol_ratio = self._compute_vol_metrics()
            vol_score = self._compute_vol_score(vol_ratio)
            avg_corr, corr_score = self._compute_corr_metrics()
            # En suspension, la validation de réentrée doit regarder l'état
            # depuis la sortie cash, pas le drawdown global historique.
            reentry_dd_score = self._compute_dd_score(float(dd_since_exit))
            reentry_score = float(min(trend_score, vol_score, corr_score, reentry_dd_score))
            reentry_target_regime = self._classify_regime(reentry_score)
            if reentry_target_regime.name.upper() in allowed_regimes:
                self._reentry_regime_ok_days += 1
            else:
                self._reentry_regime_ok_days = 0

            reenter_ok = False
            if fd > 0 and days_suspended >= fd and dd_since_exit > f_dd:
                reenter_ok = True
            elif days_suspended >= cd and dd_since_exit > r_dd:
                reenter_ok = True
            if reenter_ok and SUSPENSION_REENTRY_REQUIRE_REGIME_CONFIRMATION:
                reenter_ok = self._reentry_regime_ok_days >= min_regime_days

            if reenter_ok:
                self.trading_suspended = False
                self._underwater_streak = 0
                self.peak_value        = portfolio_value  # reset peak au niveau cash
                self._reentry_date = date
                self._reentry_peak_value = float(portfolio_value)
                # Recut : ancre au 1er jour avec positions (souvent après le prochain rebalance)
                self._reentry_anchor_value = None
                self._reentry_sessions_since = 0
                self._reentry_regime_ok_days = 0
                del self._suspension_date
                del self._cash_at_exit
                logger.info(
                    f"  ✅ {date.date()} | RÉENTRÉE | "
                    f"{days_suspended}j de suspension | "
                    f"Trading repris | peak reset ${portfolio_value:,.0f}"
                )
                # On continue le calcul normal ci-dessous
            else:
                snapshot.regime            = MarketRegime.SUSPENDED
                snapshot.regime_score_raw  = 0.0
                snapshot.trading_suspended = True
                snapshot.dd_max_stop       = True
                snapshot.risk_scaling      = 0.0
                snapshot.regime_score      = 0.0
                snapshot.current_drawdown  = current_drawdown
                snapshot.peak_value        = self.peak_value
                cooldown_ok = (
                    (fd > 0 and days_suspended >= fd and dd_since_exit > f_dd)
                    or (days_suspended >= cd and dd_since_exit > r_dd)
                )
                if not cooldown_ok:
                    snapshot.suspension_reason = "COOLDOWN_ACTIVE"
                elif SUSPENSION_REENTRY_REQUIRE_REGIME_CONFIRMATION:
                    snapshot.suspension_reason = "REENTRY_REGIME_NOT_CONFIRMED"
                else:
                    snapshot.suspension_reason = "COOLDOWN_ACTIVE"
                snapshot.suspended_days    = int(days_suspended)
                snapshot.alerts.append("TRADING_SUSPENDED")
                snapshot.trend_score = trend_score
                snapshot.vol_score = vol_score
                snapshot.corr_score = corr_score
                snapshot.dd_score = reentry_dd_score
                snapshot.regime_score = reentry_score
                snapshot.regime = MarketRegime.SUSPENDED
                snapshot.vol_short = vol_short
                snapshot.vol_long = vol_long
                snapshot.vol_ratio = vol_ratio
                snapshot.avg_correlation = avg_corr
                return snapshot

        # ── Premier déclenchement du circuit breaker ──────────
        # On vérifie ICI, avant de calculer quoi que ce soit d'autre.
        # Si DD dépasse le seuil → suspension immédiate + log unique.
        risk_off_like = self._risk_regime_current in (
            MarketRegime.STRESS,
            MarketRegime.CRISIS,
            MarketRegime.SUSPENDED,
        )
        fast_cut_window = (
            int(FAST_DRAWDOWN_CUT_WINDOW_DAYS_RISK_OFF)
            if risk_off_like
            else int(FAST_DRAWDOWN_CUT_WINDOW_DAYS)
        )
        fast_dd = _compute_fast_dd(fast_cut_window)
        fast_dd_long = _compute_fast_dd(int(FAST_DRAWDOWN_CUT_WINDOW_DAYS_LONG))
        if current_drawdown < -MAX_PORTFOLIO_DRAWDOWN:
            return _suspend(
                reason="MAX_DRAWDOWN_BREACH",
                alert="CIRCUIT_BREAKER_TRIGGERED",
                log_message=(
                    f"🚨 {date.date()} | CIRCUIT BREAKER DÉCLENCHÉ | "
                    f"DD: {current_drawdown:.1%} > seuil -{MAX_PORTFOLIO_DRAWDOWN:.0%} | "
                    f"Trading SUSPENDU — toutes les positions liquidées en cash"
                ),
            )

        stress_like = self._risk_regime_current in (MarketRegime.STRESS, MarketRegime.CRISIS)
        fast_dd_eligible_short = (not FAST_DRAWDOWN_CUT_ONLY_UNDER_STRESS) or stress_like
        # Horizon long : toujours évalué (grind en BULL/NORMAL). Seul le court est soumis à ONLY_UNDER_STRESS.
        fast_dd_eligible_long = True
        fast_dd_short_breach = fast_dd_eligible_short and (
            fast_dd <= -float(FAST_DRAWDOWN_CUT_THRESHOLD)
        )
        fast_dd_long_breach = fast_dd_eligible_long and (
            fast_dd_long <= -float(FAST_DRAWDOWN_CUT_THRESHOLD_LONG)
        )
        fast_dd_breach = bool(fast_dd_short_breach or fast_dd_long_breach)
        if FAST_DRAWDOWN_CUT_ENABLED and fast_dd_breach:
            self._fast_dd_breach_streak += 1
        else:
            self._fast_dd_breach_streak = 0
        if (
            FAST_DRAWDOWN_CUT_ENABLED
            and fast_dd_breach
            and self._fast_dd_breach_streak >= int(max(1, FAST_DRAWDOWN_CUT_CONFIRM_DAYS))
        ):
            short_txt = f"{fast_cut_window}j {fast_dd:.1%}<=-{float(FAST_DRAWDOWN_CUT_THRESHOLD):.1%}"
            long_txt = (
                f"{int(FAST_DRAWDOWN_CUT_WINDOW_DAYS_LONG)}j {fast_dd_long:.1%}"
                f"<=-{float(FAST_DRAWDOWN_CUT_THRESHOLD_LONG):.1%}"
            )
            short_gate = "ok" if fast_dd_eligible_short else "blocked_stress_only"
            return _suspend(
                reason="FAST_DRAWDOWN_BREACH",
                alert="FAST_DRAWDOWN_BREAKER_TRIGGERED",
                log_message=(
                    f"🚨 {date.date()} | FAST DRAWDOWN CUT | "
                    f"{short_txt} | {long_txt} | "
                    f"confirm={self._fast_dd_breach_streak}j | "
                    f"(risk_regime={self._risk_regime_current.name} | short_gate={short_gate} | "
                    f"DD_global={current_drawdown:.1%}) | "
                    f"Trading SUSPENDU — toutes les positions liquidées en cash"
                ),
            )

        # ── ÉTAPE 4 : Calcul des 4 filtres ───────────────────

        # Filtre 1 — Trend (MA200 du benchmark)
        # Benchmark proxy : moyenne équipondérée des prix normalisés
        # (identique à risk_enhanced.py)
        trend_score = self._compute_trend_score()

        # Filtre 2 — Volatilité (ratio vol court / vol long)
        vol_short, vol_long, vol_ratio = self._compute_vol_metrics()
        vol_score = self._compute_vol_score(vol_ratio)

        # Filtre 3 — Corrélation cross-actifs
        avg_corr, corr_score = self._compute_corr_metrics()

        # Filtre 4 — Drawdown (circuit breaker progressif)
        dd_score = self._compute_dd_score(current_drawdown)

        # ── ÉTAPE 5 : Score composite + lissage ──────────────
        # En crise, le filtre le plus restrictif doit dominer l'évaluation.
        # Le lissage 3 jours (`self._regime_buffer`) évite les changements
        # trop brutaux d'un jour à l'autre.
        regime_score_raw = min(trend_score, vol_score, corr_score, dd_score)

        # Lissage sur 3 jours : évite les changements brusques
        # qui génèrent du turnover inutile et des coûts de transaction
        self._regime_buffer.append(regime_score_raw)
        regime_score_smoothed = float(np.mean(self._regime_buffer))

        # Contrôle "tail risk" : quand le drawdown devient dangereux,
        # on veut que dd_score domine immédiatement.
        # Sans cela, le lissage peut retarder le passage STRESS/CRISIS,
        # et laisse passer des épisodes de MaxDD.
        regime_score = float(min(regime_score_smoothed, dd_score))

        # Classification en régime discret + hystérésis (anti-thrashing)
        target_regime = self._classify_regime(regime_score)
        regime = self._apply_regime_hysteresis(target_regime)

        # ── ÉTAPE 6 : Risk scaling final ──────────────────────
        # COMPOSANT 1 : Vol targeting
        #   TARGET_VOL / vol_réalisée → si la vol est 2x la cible,
        #   on réduit l'expo de 50%
        vol_scaling = self._compute_vol_scaling(vol_short)

        # COMPOSANT 2 : Facteur régime
        # La réduction d'exposition liée au régime est appliquée côté signal
        # (event_driven.engine) via `apply_regime_weight_filter`, pour éviter
        # toute double pénalisation.
        # Ici, on ne garde que le vol-targeting.
        # Cap légèrement plus strict pour limiter le tail risk
        risk_scaling = float(np.clip(vol_scaling, 0.10, 1.40))

        prolonged_uw = False
        if PROLONGED_UNDERWATER_ENABLED:
            if current_drawdown < 0.0:
                self._underwater_streak += 1
            else:
                self._underwater_streak = 0
            if self._underwater_streak > int(PROLONGED_UNDERWATER_MIN_DAYS) and current_drawdown <= float(
                PROLONGED_UNDERWATER_MIN_DD
            ):
                risk_scaling = float(
                    np.clip(risk_scaling * float(PROLONGED_UNDERWATER_RISK_SCALE_MULT), 0.08, 1.40)
                )
                prolonged_uw = True

        if SUSPENSION_REENTRY_RAMP_ENABLED and hasattr(self, "_reentry_date"):
            ramp_scales = tuple(float(x) for x in SUSPENSION_REENTRY_RAMP_SCALES)
            if len(ramp_scales) > 0:
                days_since_reentry = max(0, int((date - self._reentry_date).days))
                idx = min(days_since_reentry, len(ramp_scales) - 1)
                risk_scaling = float(np.clip(risk_scaling * ramp_scales[idx], 0.08, 1.40))

        # ── ÉTAPE 7 : Stop-loss individuels ──────────────────
        positions_to_close = self._check_stop_losses(
            current_positions, prices, entry_prices
        )

        # ── ÉTAPE 8 : Remplissage du snapshot ────────────────
        snapshot.trend_score      = trend_score
        snapshot.vol_score        = vol_score
        snapshot.corr_score       = corr_score
        snapshot.dd_score         = dd_score
        snapshot.regime_score_raw = regime_score_raw
        snapshot.regime_score     = regime_score
        snapshot.regime           = regime
        snapshot.vol_short        = vol_short
        snapshot.vol_long         = vol_long
        snapshot.vol_ratio        = vol_ratio
        snapshot.avg_correlation  = avg_corr
        snapshot.risk_scaling     = risk_scaling
        snapshot.positions_to_close = positions_to_close
        snapshot.prolonged_underwater_active = prolonged_uw

        # Log résumé (tous les 21 jours environ)
        if len(self._port_returns_long) % 21 == 0:
            logger.info(
                f"  Risk {date.date()} | "
                f"Régime: {regime.name} | "
                f"Score: {regime_score:.2f} "
                f"[T:{trend_score:.2f} V:{vol_score:.2f} C:{corr_score:.2f} D:{dd_score:.2f}] | "
                f"Scaling: {risk_scaling:.2f}x | "
                f"DD: {current_drawdown:.1%}"
            )

        return snapshot

    # ─────────────────────────────────────────────────────────
    # FILTRE 1 : Trend (MA200)
    # ─────────────────────────────────────────────────────────
    def _compute_trend_score(self) -> float:
        """
        Score de tendance basé sur la position du benchmark vs sa MA200.

        LOGIQUE IDENTIQUE à risk_enhanced.py > compute_trend_filter() :
        - Benchmark au-dessus de MA200 → bull market → score = 1.0
        - Benchmark en dessous → bear market → score réduit progressivement
          vers 0.5 (on ne coupe jamais complètement sur ce seul filtre)

        PARTICULARITÉ EVENT-DRIVEN :
        On utilise self._benchmark_buffer (deque de 200 valeurs max)
        au lieu d'un .rolling(200) sur un DataFrame complet.
        Le calcul est identique, la structure est différente.

        Returns:
            float entre 0.5 et 1.0
        """
        n = len(self._benchmark_buffer)

        # Pas assez d'historique → score neutre
        if n < 50:
            return 1.0

        values   = list(self._benchmark_buffer)
        current  = values[-1]
        ma200    = float(np.mean(values))  # MA sur tout le buffer disponible

        # Distance relative : (prix - MA) / MA
        # Positif = au-dessus, Négatif = en dessous
        distance = (current - ma200) / (ma200 + 1e-8)

        if distance >= 0:
            return 1.0  # Au-dessus de la MA → pleine exposition

        # En dessous : transition progressive vers 0.5
        # À -5% sous la MA : score ≈ 0.65
        # À -10% sous la MA : score → 0.5 (plancher)
        score = 0.5 + 0.5 * (1 + distance * 5)
        return float(np.clip(score, 0.5, 1.0))

    # ─────────────────────────────────────────────────────────
    # FILTRE 2 : Volatilité
    # ─────────────────────────────────────────────────────────
    def _compute_vol_metrics(self):
        """
        Calcule les volatilités annualisées courte et longue.

        FENÊTRES :
          Court terme : 10 jours → détecte les chocs rapidement
                        (le crash COVID a mis 5 jours pour exploser)
          Long terme  : jusqu'à 252 jours → baseline "normale"

        ANNUALISATION :
          σ_annuelle = σ_journalière × √252
          Hypothèse : rendements i.i.d. (indépendants, même distribution)
          → la variance annuelle = 252 × variance journalière

        Returns:
            tuple (vol_short, vol_long, vol_ratio)
        """
        # Volatilité courte (10j)
        if len(self._port_returns_short) >= 5:
            vol_short = float(np.std(list(self._port_returns_short)) * np.sqrt(252))
        else:
            vol_short = TARGET_VOLATILITY  # fallback si pas assez de données

        # Volatilité longue (jusqu'à 252j)
        if len(self._port_returns_long) >= 20:
            vol_long = float(np.std(list(self._port_returns_long)) * np.sqrt(252))
        else:
            vol_long = TARGET_VOLATILITY  # fallback

        # Protection contre division par zéro
        vol_long  = max(vol_long, 0.01)
        vol_short = max(vol_short, 0.01)

        vol_ratio = vol_short / vol_long
        return vol_short, vol_long, vol_ratio

    def _compute_vol_score(self, vol_ratio: float) -> float:
        """
        Convertit le ratio de volatilité en score [0.10, 1.0].

        SEUILS (identiques à risk_enhanced.py) :
          ratio > 2.0 → CRISE  → score = 0.10
          ratio > 1.5 → STRESS → score = 0.25
          ratio > 1.2 → ÉLEVÉ  → score = 0.60
          ratio ≤ 1.2 → NORMAL → score = 1.00

        Pourquoi 10j au lieu de 20j pour la fenêtre courte ?
        Plus réactif aux chocs. Le COVID a exploité en 5 jours —
        avec 20j de fenêtre on aurait réagi 2 semaines trop tard.
        """
        if   vol_ratio > 2.0: return 0.10
        elif vol_ratio > 1.5: return 0.25
        elif vol_ratio > 1.2: return 0.60
        else:                 return 1.00

    # ─────────────────────────────────────────────────────────
    # FILTRE 3 : Corrélation cross-actifs
    # ─────────────────────────────────────────────────────────
    def _compute_corr_metrics(self):
        """
        Calcule la corrélation moyenne entre tous les actifs (fenêtre 42j).

        CONCEPT — "ALL CORRELATIONS GO TO 1 IN A CRISIS" :
        En temps de crise, toutes les corrélations convergent vers 1.
        La diversification disparaît exactement quand on en a besoin.
        C'est pourquoi on surveille les corrélations comme indicateur
        avancé de stress systémique.

        ALGORITHME :
        On a un buffer de 42 vecteurs de rendements journaliers.
        On construit la matrice de rendements (42 × n_actifs),
        puis on calcule la corrélation moyenne hors-diagonale.

        SEUILS :
          ρ > 0.45 → CRISE  → score = 0.10
          ρ > 0.35 → STRESS → transition progressive vers 0.10
          ρ ≤ 0.35 → NORMAL → score = 1.00

        Returns:
            tuple (avg_corr, corr_score)
        """
        n = len(self._returns_buffer)

        # Pas assez d'historique → score neutre
        if n < 20:
            return 0.0, 1.0

        # Construction de la matrice (jours × actifs)
        try:
            matrix = np.array(list(self._returns_buffer))  # shape (n, n_actifs)

            if matrix.shape[1] < 2:
                return 0.0, 1.0

            # Matrice de corrélation de Pearson
            corr_matrix = np.corrcoef(matrix.T)  # shape (n_actifs, n_actifs)

            # Extraction du triangle supérieur sans la diagonale
            # La diagonale = corrélation d'un actif avec lui-même = 1
            # → on l'exclut car elle biaiserait la moyenne
            n_assets = corr_matrix.shape[0]
            upper_tri = corr_matrix[np.triu_indices(n_assets, k=1)]

            avg_corr = float(np.nanmean(upper_tri))
            avg_corr = max(0.0, avg_corr)  # clip à 0 minimum

        except Exception:
            return 0.0, 1.0

        # Conversion en score
        if avg_corr > 0.45:
            corr_score = 0.10
        elif avg_corr > 0.35:
            # Transition progressive entre 1.0 et 0.10
            corr_score = float(np.interp(avg_corr, [0.35, 0.45], [1.0, 0.10]))
        else:
            corr_score = 1.0

        return avg_corr, corr_score

    # ─────────────────────────────────────────────────────────
    # FILTRE 4 : Drawdown (circuit breaker progressif)
    # ─────────────────────────────────────────────────────────
    def _compute_dd_score(self, current_drawdown: float) -> float:
        """
        Score basé sur le drawdown courant du portefeuille.

        LOGIQUE "POSITION SCALING PAR DRAWDOWN" :
        Quand la stratégie est en drawdown, on réduit progressivement
        l'exposition pour deux raisons :
          1. La stratégie ne fonctionne pas dans ce régime
          2. On protège le capital restant

        SEUILS PROGRESSIFS :
          DD > 20% → score = 0.00 (circuit breaker total)
          DD > 15.5% → score = 0.25 (réduction forte)
          DD > 12% → score = 0.50
          DD > 9% → score = 0.52 (< seuil NORMAL 0.55 → STRESS si ce filtre domine)
          DD > 6% → score = 0.75
          DD ≤ 6% → score = 1.00

        NOTE : current_drawdown est NÉGATIF (ex: -0.15 = -15%)
        On travaille avec la valeur absolue pour les comparaisons.
        """
        dd_abs = abs(current_drawdown)

        if dd_abs > 0.20:
            return 0.00
        if dd_abs > 0.155:
            return 0.25
        if dd_abs > 0.12:
            return 0.50
        if dd_abs > 0.09:
            return 0.52
        if dd_abs > 0.06:
            return 0.75
        return 1.00

    # ─────────────────────────────────────────────────────────
    # CLASSIFICATION DU RÉGIME
    # ─────────────────────────────────────────────────────────
    def _classify_regime(self, regime_score: float) -> MarketRegime:
        """
        Convertit le score composite en régime discret.

        SEUILS :
          score ≥ 0.80 → BULL   (1.00x exposition)
          score ≥ 0.55 → NORMAL (0.75x exposition)
          score ≥ 0.30 → STRESS (0.50x exposition)
          score < 0.30 → CRISIS (0.25x exposition)
        """
        if   regime_score >= 0.80: return MarketRegime.BULL
        elif regime_score >= 0.55: return MarketRegime.NORMAL
        elif regime_score >= 0.30: return MarketRegime.STRESS
        else:                      return MarketRegime.CRISIS

    # ─────────────────────────────────────────────────────────
    # VOL TARGETING (composant 1 du scaling)
    # ─────────────────────────────────────────────────────────
    def _compute_vol_scaling(self, realized_vol: float) -> float:
        """
        Calcule le facteur de vol targeting.

        FORMULE :
            scaling = TARGET_VOLATILITY / realized_vol

        INTUITION :
        Si la vol réalisée est 2x la cible → scaling = 0.5
          → on réduit l'exposition de 50%
        Si la vol réalisée est 0.5x la cible → scaling = 2.0
          → on augmenterait, mais on clippe à 1.5 max

        CONTRAINTES [0.25, 1.40] :
          Min 0.25 : on garde toujours 25% d'exposition
          Max 1.40 : on ne dépasse pas 1.40x en période très calme
        """
        if realized_vol < 0.001:
            return 1.0

        raw_scaling = TARGET_VOLATILITY / realized_vol
        return float(np.clip(raw_scaling, 0.25, 1.40))

    # ─────────────────────────────────────────────────────────
    # STOP-LOSS INDIVIDUELS
    # ─────────────────────────────────────────────────────────
    def _check_stop_losses(
        self,
        current_positions : dict,
        current_prices    : pd.Series,
        entry_prices      : dict,
    ) -> list:
        """
        Vérifie si des positions individuelles doivent être fermées.

        LOGIQUE :
        Si une position perd plus de MAX_POSITION_LOSS (15%)
        depuis son prix d'entrée → on la ferme immédiatement.

        ASYMÉTRIE LONG / SHORT :
          Long  : perte si prix baisse → P&L = (P_courant - P_entrée) / P_entrée
          Short : perte si prix monte  → P&L = -(P_courant - P_entrée) / P_entrée

        Args:
            current_positions : dict { symbol: quantité (+ long, - short) }
            current_prices    : Series des prix actuels
            entry_prices      : dict { symbol: prix d'entrée moyen }

        Returns:
            list des symboles à fermer
        """
        to_close = []

        for symbol, qty in current_positions.items():
            if qty == 0 or symbol not in entry_prices:
                continue
            if symbol not in current_prices.index:
                continue

            entry   = entry_prices[symbol]
            current = float(current_prices[symbol])

            if entry <= 0 or current <= 0:
                continue

            # P&L en % depuis l'entrée
            pnl_pct = (current - entry) / entry

            # Pour une position short, le P&L est inversé
            if qty < 0:
                pnl_pct = -pnl_pct

            # Déclenchement du stop-loss
            if pnl_pct < -MAX_POSITION_LOSS:
                to_close.append(symbol)
                logger.warning(
                    f"  🛑 STOP-LOSS {symbol} | "
                    f"P&L: {pnl_pct:.1%} | "
                    f"Entrée: {entry:.2f}$ | Courant: {current:.2f}$"
                )

        return to_close


# ============================================================
# SECTION 3 — GÉNÉRATEUR DE SIGNAL AVEC RISQUE INTÉGRÉ
# ============================================================

class MomentumSignalGeneratorV2:
    """
    Version améliorée du MomentumSignalGenerator (event_driven.signal_generator).

    AMÉLIORATIONS vs l'original :
      1. Vol EWMA par actif → vol parity weighting (comme momentum_signal.py)
      2. Scaling des poids par le risk_scaling du RiskSnapshot
      3. Contraintes complètes (cap, levier max) depuis config.py
      4. Seuil de rebalancement pour éviter les micro-trades

    L'objet EventDrivenRiskManager est passé en paramètre
    → séparation des responsabilités : le risk manager gère le risque,
    le signal generator gère le signal.

    USAGE :
        risk_manager = EventDrivenRiskManager(100_000)
        signal_gen   = MomentumSignalGeneratorV2(data_handler, risk_manager)

        for date, market_event in ...:
            risk_snapshot = risk_manager.update(date, prices, ...)
            if rebal_today:
                weights = signal_gen.compute_weights(date, risk_snapshot)
    """

    def __init__(
        self,
        data_handler,
        risk_manager    : EventDrivenRiskManager,
        momentum_weights: dict  = None,
        skip_days       : int   = SKIP_DAYS,
        long_quantile   : float = LONG_QUANTILE,
        short_quantile  : float = SHORT_QUANTILE,
    ):
        self.data        = data_handler
        self.risk_mgr    = risk_manager
        self.weights     = momentum_weights or MOMENTUM_WEIGHTS
        self.skip_days   = skip_days
        self.long_q      = long_quantile
        self.short_q     = short_quantile
        self.max_window  = max(self.weights.keys())

        # Buffer EWMA des volatilités par actif
        # λ = 0.94 (standard RiskMetrics — JP Morgan, 1994)
        # On maintient la variance EWMA pour chaque actif séparément
        # σ²(t) = λ × σ²(t-1) + (1-λ) × r²(t)
        self._ewma_var   = {}    # { symbol: variance_ewma_courante }
        self._ewma_lambda = 0.94

        # Mémoire des poids du mois précédent
        # Utilisée par _filter_by_rebal_threshold pour éviter
        # de retrader des positions dont le signal n'a pas changé.
        self._prev_weights = {}
        self.last_diagnostics = {}

        logger.info("MomentumSignalGeneratorV2 initialisé")

    def update_ewma_vol(self, prices: pd.Series, prev_prices: pd.Series):
        """
        Met à jour la variance EWMA pour chaque actif.

        FORMULE EWMA (RiskMetrics) :
            σ²(t) = λ × σ²(t-1) + (1-λ) × r²(t)

        POURQUOI EWMA ET PAS UNE SIMPLE MOYENNE MOBILE ?
        La volatilité est PERSISTANTE — après un choc, elle reste
        élevée puis revient graduellement à la normale.
        EWMA capte cette dynamique en donnant plus de poids aux
        observations récentes (λ=0.94 → demi-vie ≈ 12 jours).

        APPELER CETTE MÉTHODE CHAQUE JOUR avant compute_weights().
        """
        if prev_prices is None:
            return

        common = prices.index.intersection(prev_prices.index)
        for symbol in common:
            p_curr = float(prices[symbol])
            p_prev = float(prev_prices[symbol])

            if p_prev <= 0 or p_curr <= 0:
                continue

            # Log return journalier
            r = np.log(p_curr / p_prev)

            # Mise à jour récursive de la variance EWMA
            lam = self._ewma_lambda
            if symbol in self._ewma_var:
                self._ewma_var[symbol] = lam * self._ewma_var[symbol] + (1 - lam) * r**2
            else:
                # Initialisation : variance = r² du premier jour
                self._ewma_var[symbol] = r**2

    def get_ewma_vol(self, symbol: str) -> float:
        """
        Retourne la volatilité EWMA annualisée pour un actif.

        ANNUALISATION : σ_annuelle = √(252 × σ²_journalière)
        Facteur √252 car on suppose des rendements i.i.d.

        Returns:
            float : vol annualisée (ex: 0.20 = 20%)
                    Si pas encore calculée → TARGET_VOLATILITY (15%)
        """
        if symbol not in self._ewma_var:
            return TARGET_VOLATILITY  # fallback conservateur

        vol_annual = float(np.sqrt(self._ewma_var[symbol] * 252))
        return max(vol_annual, 0.01)  # minimum 1% pour éviter /0

    def compute_weights(
        self,
        date          : pd.Timestamp,
        risk_snapshot : RiskSnapshot,
        market_regime_state: str = "",
    ) -> dict:
        """
        Calcule les poids cibles du portefeuille pour la date t.

        PIPELINE — IDENTIQUE AU VECTORISÉ :
          1. Historique jusqu'à t (no look-ahead strict)
          2. MomentumSignalGenerator complet : log returns → momentum brut
             → score composite → vol EWMA → z-score CS → signal CS + TS
             → signal final combiné
          3. Vol parity weighting depuis la vol EWMA calculée par le générateur
          4. Contraintes (cap, levier) depuis config.py
          5. Scaling par risk_snapshot.risk_scaling

        POURQUOI RÉUTILISER MomentumSignalGenerator ?
        Le vectorisé (Sharpe 0.643) utilise ce générateur avec la pipeline
        complète CS + TS. Notre première implémentation utilisait un score
        momentum brut sans z-score ni combinaison CS/TS → signal différent
        et moins performant. En réutilisant le même générateur avec
        l'historique disponible à t, on garantit l'alignement des signaux
        sans look-ahead (get_history filtre strictement ≤ t).

        Args:
            date          : date du calcul (t)
            risk_snapshot : sortie de EventDrivenRiskManager.update()

        Returns:
            dict { symbol: poids } ou {} si pas de signal valide
        """
        # Si le trading est suspendu → aucune position
        rebal_threshold = float(REBALANCE_THRESHOLD_DEFAULT)
        rebal_threshold_context = "DEFAULT"
        # DD-aware rebalancing threshold (asymétrique) :
        # - si dd_score est mauvais (drawdown dangereux) -> on baisse le threshold => on retrade
        #   davantage pour désendetter/resserrer rapidement (meilleur contrôle du tail risk).
        # - si dd_score est bon -> on conserve le threshold d'origine (sinon on peut perdre du Sharpe,
        #   car on réduit trop le "churn" et on rate du rééquilibrage utile).
        dd_score_for_rebal = float(getattr(risk_snapshot, "dd_score", 1.0) or 1.0)
        dd_score_for_rebal = float(np.clip(dd_score_for_rebal, 0.0, 1.0))

        # danger = 0 quand dd_score >= 0.6, danger = 1 quand dd_score <= 0.0
        danger = float(np.clip((0.60 - dd_score_for_rebal) / 0.60, 0.0, 1.0))
        # multiplier dans [0.75, 1.0] : only decrease threshold when danger > 0
        dd_rebal_multiplier = float(1.0 - 0.25 * danger)
        rebal_threshold_dd_aware = float(np.clip(rebal_threshold * dd_rebal_multiplier, 0.75 * rebal_threshold, rebal_threshold))
        diagnostics = {
            "date": date,
            "signal_reason": "OK",
            "risk_scaling": float(getattr(risk_snapshot, "risk_scaling", np.nan)),
            "rebal_threshold": float(rebal_threshold),
            "rebal_threshold_context": str(rebal_threshold_context),
            "dd_score": dd_score_for_rebal,
            "dd_rebal_multiplier": float(dd_rebal_multiplier),
            "rebal_threshold_dd_aware": float(rebal_threshold_dd_aware),
            "gross_signal_raw": 0.0,
            "gross_after_constraints": 0.0,
            "gross_after_risk_manager": 0.0,
            "gross_after_rebal_threshold": 0.0,
            "gross_after_signal_generator": 0.0,
            "n_long_candidates": 0,
            "n_short_candidates": 0,
            "n_selected_positions": 0,
        }

        if risk_snapshot.trading_suspended:
            diagnostics["signal_reason"] = "TRADING_SUSPENDED"
            self.last_diagnostics = diagnostics
            return {}

        # ── ÉTAPE 1 : Historique disponible jusqu'à t ─────────
        # On prend max_window + skip + warmup = ~285 jours minimum.
        # get_history() filtre strictement index <= date → no look-ahead.
        # On prend plus large (520j) pour que le générateur ait assez
        # pour calculer le z-score cross-sectionnel (au moins 252j valides).
        required = max(self.weights.keys()) + self.skip_days + 270
        prices_hist = self.data.get_history(date, required)

        if len(prices_hist) < max(self.weights.keys()) + self.skip_days + 10:
            diagnostics["signal_reason"] = "INSUFFICIENT_HISTORY"
            self.last_diagnostics = diagnostics
            return {}

        # ── ÉTAPE 2 : Pipeline signal complète (identique au vectorisé) ──
        # On importe ici (pas au module) pour éviter les imports circulaires
        # et parce que cet import n'est nécessaire qu'au rebalancement mensuel.
        try:
            import sys, os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from strategies.momentum.momentum_signal import MomentumSignalGenerator
        except ImportError:
            logger.error("❌ Impossible d'importer MomentumSignalGenerator")
            diagnostics["signal_reason"] = "IMPORT_ERROR"
            self.last_diagnostics = diagnostics
            return {}

        # On désactive les logs du générateur pour ne pas polluer les logs
        # de la boucle event-driven (le générateur logue beaucoup)
        import logging
        logging.disable(logging.CRITICAL)
        try:
            generator   = MomentumSignalGenerator(prices_hist)
            sig_results = generator.run_full_pipeline(
                cs_weight=0.5,   # même paramètre que dans run_backtest.py
                ts_weight=0.5,
            )
        except Exception as e:
            logging.disable(logging.NOTSET)
            logger.warning(f"  Signal generator échec : {e}")
            diagnostics["signal_reason"] = "PIPELINE_ERROR"
            self.last_diagnostics = diagnostics
            return {}
        finally:
            logging.disable(logging.NOTSET)

        # ── ÉTAPE 3 : Extraire le signal final à la date t ────
        # signal_final est un DataFrame (dates × actifs).
        # On prend la DERNIÈRE ligne disponible = signal à t.
        # Jamais de ligne future car prices_hist ne contient que ≤ t.
        signal_final = sig_results["signal_final"]
        ewma_vol     = sig_results["ewma_vol"]

        if signal_final.empty:
            diagnostics["signal_reason"] = "EMPTY_SIGNAL"
            self.last_diagnostics = diagnostics
            return {}

        signal_today = signal_final.iloc[-1].dropna()
        vol_today    = ewma_vol.iloc[-1].clip(lower=0.01)

        # ── ÉTAPE 4 : Vol parity weighting ────────────────────
        # On sélectionne les N meilleurs actifs longs et shorts
        # plutôt qu'un seuil continu — cela contrôle directement
        # le nombre de positions et donc le turnover.
        # N_LONG = 8, N_SHORT = 5 → ~13 positions totales
        # Cohérent avec MAX_POSITION_SIZE=10% et levier 1.0x max.
        N_LONG  = 6
        N_SHORT = 4

        # Trier par signal décroissant
        signal_sorted = signal_today.sort_values(ascending=False)
        top_longs  = signal_sorted.head(N_LONG)
        top_shorts = signal_sorted.tail(N_SHORT)
        diagnostics["n_long_candidates"] = int(len(top_longs))
        diagnostics["n_short_candidates"] = int(len(top_shorts))

        raw_weights = {}

        for symbol, sig in top_longs.items():
            if sig <= 0.02:          # signal trop faible → pas de long
                continue
            if symbol not in vol_today.index:
                continue
            vol = float(vol_today[symbol])
            raw_weights[symbol] = sig * (TARGET_VOLATILITY / vol)

        for symbol, sig in top_shorts.items():
            if sig >= -0.02:         # signal trop faible → pas de short
                continue
            if symbol not in vol_today.index:
                continue
            vol  = float(vol_today[symbol])
            # Shorts à 50% — momentum short moins persistant
            raw_weights[symbol] = sig * (TARGET_VOLATILITY / vol) * 0.5

        diagnostics["gross_signal_raw"] = float(sum(abs(w) for w in raw_weights.values()))
        if not raw_weights:
            diagnostics["signal_reason"] = "NO_RAW_WEIGHTS"
            self.last_diagnostics = diagnostics
            return {}

        # ── ÉTAPE 5 : Contraintes ─────────────────────────────
        weights_new = self._apply_constraints(raw_weights)
        diagnostics["gross_after_constraints"] = float(sum(abs(w) for w in weights_new.values()))

        # ── ÉTAPE 6 : Application du risk scaling ─────────────
        # risk_scaling ∈ [0.10, 1.50] — calculé par EventDrivenRiskManager
        # Réduit l'exposition en régime STRESS/CRISIS ou vol élevée.
        scaling     = risk_snapshot.risk_scaling
        weights_new = {s: w * scaling for s, w in weights_new.items()}
        diagnostics["gross_after_risk_manager"] = float(sum(abs(w) for w in weights_new.values()))

        # ── ÉTAPE 7 : Filtre de rebalancement ─────────────────
        # On ne retrade que les positions qui ont changé significativement
        # (Δw > 1.5% du capital). Réduit le turnover de ~80%.
        weights = self._filter_by_rebal_threshold(
            new_weights  = weights_new,
            prev_weights = self._prev_weights,
            threshold    = rebal_threshold_dd_aware,
            market_regime_state=market_regime_state,
        )
        weights = self._apply_regime_net_exposure_target(weights, market_regime_state)
        diagnostics["rebal_threshold"] = float(rebal_threshold_dd_aware)
        diagnostics["rebal_threshold_context"] = str(rebal_threshold_context)
        diagnostics["gross_after_rebal_threshold"] = float(sum(abs(w) for w in weights.values()))

        # Mémoriser les poids pour le prochain rebalancement
        self._prev_weights = {s: w for s, w in weights.items() if abs(w) > 0.002}

        # Fermeture des positions en stop-loss (force override du filtre)
        for symbol in risk_snapshot.positions_to_close:
            if symbol in weights:
                weights[symbol] = 0.0
                self._prev_weights.pop(symbol, None)
                logger.info(f"  Position {symbol} → 0 (stop-loss)")

        diagnostics["gross_after_signal_generator"] = float(sum(abs(w) for w in weights.values()))
        diagnostics["n_selected_positions"] = int(len(weights))
        self.last_diagnostics = diagnostics

        # Log du snapshot du signal au rebalancement
        n_long  = sum(1 for w in weights.values() if w > 0.01)
        n_short = sum(1 for w in weights.values() if w < -0.01)
        gross   = sum(abs(w) for w in weights.values())
        logger.info(
            f"  Signal {date.date()} | "
            f"L:{n_long} S:{n_short} | "
            f"Gross: {gross:.2f}x | "
            f"Scaling: {scaling:.2f}x | "
            f"Rebal thr: {rebal_threshold:.3f}"
        )

        return weights

    def _apply_constraints(self, raw_weights: dict) -> dict:
        """
        Applique les contraintes de risque sur les poids bruts.

        CONTRAINTE 1 — Position cap :
          Aucune position ne dépasse MAX_POSITION_SIZE (10%)
          → Diversification minimale, protection idiosyncratique

        CONTRAINTE 2 — Levier max :
          gross_exposure = Σ|w_i| ≤ MAX_LEVERAGE (1.5x)
          Si dépassé → on scale tous les poids à la baisse

        CONTRAINTE 3 — Seuil de rebalancement :
          On ne retrade pas si |Δw| < 0.005 du capital
          → évite les micro-trades coûteux (10bps + 5bps par trade)

        Args:
            raw_weights : dict { symbol: poids brut }

        Returns:
            dict { symbol: poids contraint }
        """
        weights = dict(raw_weights)

        # Contrainte 1 : cap individuel
        weights = {
            s: float(np.clip(w, -MAX_POSITION_SIZE, MAX_POSITION_SIZE))
            for s, w in weights.items()
        }

        # Contrainte 2 : levier max event-driven = 1.0x
        # On est plus conservateur que le vectorisé (1.5x) car l'event-driven
        # tourne sur données réelles avec des coûts à chaque trade.
        # 1.0x = pas de levier → exposition nette ≤ 100% du capital.
        ED_MAX_LEVERAGE = 1.0
        gross = sum(abs(w) for w in weights.values())
        if gross > ED_MAX_LEVERAGE:
            scale = ED_MAX_LEVERAGE / gross
            weights = {s: w * scale for s, w in weights.items()}

        # Retirer les poids proches de zéro (évite les micro-positions)
        weights = {s: w for s, w in weights.items() if abs(w) > 0.005}

        return weights

    def _filter_by_rebal_threshold(
        self,
        new_weights   : dict,
        prev_weights  : dict,
        threshold     : float = 0.015,
        market_regime_state: str = "",
    ) -> dict:
        """
        Ne retourne que les poids qui ont changé au-delà du seuil.

        CONCEPT — SEUIL DE REBALANCEMENT :
        On ne retrade une position que si la variation de poids dépasse
        threshold (1.5% du capital par défaut).
        En dessous, les frais de transaction (15bps) ne valent pas le coût.

        RÉSULTAT :
        Au lieu de retrader 33 actifs chaque mois, on ne retrade que ceux
        dont le signal a vraiment changé → turnover réduit de ~80%.

        Args:
            new_weights  : poids cibles calculés ce mois-ci
            prev_weights : poids du mois précédent
            threshold    : variation minimale pour déclencher un trade

        Returns:
            dict des poids finaux (anciens poids conservés si Δw < seuil)
        """
        filtered = {}
        regime_name = str(market_regime_state or "").strip().upper()
        candidate_weights = self._apply_risk_off_derisk_only(new_weights, prev_weights, regime_name)

        # Actifs avec un nouveau signal
        for symbol, w_new in candidate_weights.items():
            w_prev = prev_weights.get(symbol, 0.0)
            if (
                REBALANCE_FORCE_SIGN_FLIP_EXECUTION
                and abs(w_prev) > 0.002
                and abs(w_new) > 0.002
                and (w_new * w_prev) < 0.0
            ):
                filtered[symbol] = w_new
                continue
            if abs(w_new - w_prev) >= threshold:
                filtered[symbol] = w_new   # changement significatif → on retrade
            else:
                if abs(w_prev) > 0.002:
                    filtered[symbol] = w_prev  # on garde le poids précédent

        # Actifs à fermer (dans prev mais plus dans new)
        for symbol, w_prev in prev_weights.items():
            if symbol not in candidate_weights and abs(w_prev) > 0.002:
                # Le signal a disparu → on ferme (Δw = w_prev → toujours > seuil)
                filtered[symbol] = 0.0

        return {s: w for s, w in filtered.items() if abs(w) > 0.002}

    def _apply_risk_off_derisk_only(self, new_weights: dict, prev_weights: dict, regime_name: str) -> dict:
        if not RISK_OFF_ONLY_DERISK_ENABLED or regime_name != "RISK_OFF":
            return dict(new_weights)

        out: dict[str, float] = {}
        for symbol, w_new in new_weights.items():
            w_prev = float(prev_weights.get(symbol, 0.0))
            w_new = float(w_new)
            if abs(w_prev) <= 0.002:
                continue
            if w_prev * w_new < 0.0:
                continue
            if abs(w_new) <= abs(w_prev):
                out[symbol] = w_new
            else:
                out[symbol] = w_prev
        return {s: w for s, w in out.items() if abs(w) > 0.002}

    def _apply_regime_net_exposure_target(self, weights: dict, market_regime_state: str) -> dict:
        if not REGIME_NET_EXPOSURE_TARGET_ENABLED or not weights:
            return dict(weights)

        regime = str(market_regime_state or "").strip().upper()
        regime_bounds = {
            "RISK_OFF": (float(REGIME_NET_TARGET_RISK_OFF_MIN), float(REGIME_NET_TARGET_RISK_OFF_MAX)),
            "TRANSITION": (float(REGIME_NET_TARGET_TRANSITION_MIN), float(REGIME_NET_TARGET_TRANSITION_MAX)),
            "TREND": (float(REGIME_NET_TARGET_TREND_MIN), float(REGIME_NET_TARGET_TREND_MAX)),
            "RISK_ON": (float(REGIME_NET_TARGET_RISK_ON_MIN), float(REGIME_NET_TARGET_RISK_ON_MAX)),
        }
        if regime not in regime_bounds:
            return dict(weights)

        min_net, max_net = regime_bounds[regime]
        out = dict(weights)
        net = float(sum(out.values()))

        if net > max_net:
            excess = net - max_net
            long_sum = float(sum(w for w in out.values() if w > 0))
            if long_sum > 1e-12:
                keep = max(0.0, (long_sum - excess) / long_sum)
                out = {k: (v * keep if v > 0 else v) for k, v in out.items()}
        elif net < min_net:
            deficit = min_net - net
            short_abs_sum = float(sum(-w for w in out.values() if w < 0))
            if short_abs_sum > 1e-12:
                keep = max(0.0, (short_abs_sum - deficit) / short_abs_sum)
                out = {k: (v * keep if v < 0 else v) for k, v in out.items()}

        return {s: w for s, w in out.items() if abs(w) > 0.002}


# ============================================================
# SECTION 4 — INTÉGRATION DANS L'ENGINE EVENT-DRIVEN
# ============================================================
# Instructions pour modifier event_driven/engine.py :
#
# REMPLACER :
#   self.signal_gen = MomentumSignalGenerator(self.data_handler)
# PAR :
#   from event_driven_risk import (
#       EventDrivenRiskManager, MomentumSignalGeneratorV2
#   )
#   self.risk_manager = EventDrivenRiskManager(initial_capital)
#   self.signal_gen   = MomentumSignalGeneratorV2(
#       data_handler  = self.data_handler,
#       risk_manager  = self.risk_manager,
#   )
#
# REMPLACER dans la boucle principale :
#   signal_event = self.signal_gen.compute_signal(date)
# PAR :
#   risk_snapshot = self.risk_manager.update(
#       date               = date,
#       prices             = prices,
#       portfolio_value    = self.portfolio.portfolio_value,
#       current_positions  = self.portfolio.positions,
#       entry_prices       = self.portfolio.entry_prices,
#       prev_prices        = prev_prices,  # à maintenir dans la boucle
#   )
#   self.signal_gen.update_ewma_vol(prices, prev_prices)
#
#   if rebal_today:
#       weights = self.signal_gen.compute_weights(date, risk_snapshot)
#       signal_event = SignalEvent(date=date, weights=weights,
#                                  regime=risk_snapshot.regime_score,
#                                  signal=Signal.FLAT if not weights else Signal.HOLD)
#
# SUPPRIMER :
#   self._compute_dd_multiplier()  # plus nécessaire
#   Le circuit breaker est maintenant dans risk_snapshot.trading_suspended
# ============================================================


# ============================================================
# SCRIPT PRINCIPAL — Test unitaire
# ============================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("  TEST — EventDrivenRiskManager")
    print("=" * 60)

    np.random.seed(42)

    # ── Simulation d'un historique de prix ────────────────────
    n_days   = 504
    n_assets = 8
    assets   = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "XOM", "GC", "CL"]

    mu    = 0.08 / 252
    sigma = 0.20 / np.sqrt(252)
    rets  = np.random.normal(mu, sigma, (n_days, n_assets))
    rets[200:250] = np.random.normal(-0.003, sigma * 3, (50, n_assets))

    prices_arr = 100 * np.exp(np.cumsum(rets, axis=0))
    dates      = pd.bdate_range("2022-01-01", periods=n_days)
    prices_df  = pd.DataFrame(prices_arr, index=dates, columns=assets)

    # ── Simulation valeur portefeuille réaliste ───────────────
    # On simule un portefeuille equal-weighted sur les prix
    # → la valeur évolue avec les prix, pas indépendamment.
    # BUG CORRIGÉ : avant, port_vals était recalculé depuis rets.mean()
    # indépendamment du circuit breaker → le DD continuait après
    # le déclenchement. Maintenant, dès que suspended=True,
    # la valeur reste figée (le portefeuille est en cash).
    port_vals = np.zeros(n_days)
    port_vals[0] = 100_000
    for i in range(1, n_days):
        # rendement equal-weighted du jour
        port_vals[i] = port_vals[i-1] * (1 + rets[i].mean())

    # ── Positions et prix d'entrée simulés réalistes ──────────
    # entry_prices : prix d'entrée moyen par position.
    # En vrai backtest, Portfolio.fill_order() met à jour ce dict
    # à chaque trade. Dans le test, on simule le comportement réel :
    # dès qu'un stop est déclenché, on retire la position du dict
    # ET on met à jour entry_prices au prix de ré-entrée (rebal mensuel).
    entry_prices = {a: 100.0 for a in assets}
    positions    = {a: 100 for a in assets}

    # ── Lancement du risk manager ──────────────────────────────
    rm = EventDrivenRiskManager(initial_capital=100_000)

    snapshots   = []
    prev_prices = None
    suspended   = False
    last_rebal_month = -1

    for i, date in enumerate(dates):
        prices     = prices_df.loc[date]
        port_value = port_vals[i]

        if suspended:
            port_value = port_vals[i-1]

        snapshot = rm.update(
            date              = date,
            prices            = prices,
            portfolio_value   = port_value,
            current_positions = {} if suspended else positions,
            entry_prices      = entry_prices,
            prev_prices       = prev_prices,
        )
        snapshots.append(snapshot)

        if snapshot.trading_suspended:
            suspended = True

        # SIMULATION RÉALISTE DU STOP-LOSS :
        # Quand un stop est déclenché → on ferme la position
        # → on retire du dict positions pour ne pas re-déclencher.
        # En vrai backtest, c'est Portfolio.fill_order() qui fait ça.
        for symbol in snapshot.positions_to_close:
            if symbol in positions:
                del positions[symbol]
                # entry_prices reste tel quel (la position est fermée)

        # SIMULATION DU REBALANCEMENT MENSUEL :
        # Chaque mois, on rouvre les positions fermées par stop-loss
        # au prix courant → entry_price est mis à jour.
        current_month = date.month + date.year * 12
        if current_month != last_rebal_month and not suspended:
            last_rebal_month = current_month
            # Réouverture des positions fermées + mise à jour entry_prices
            for a in assets:
                positions[a]    = 100
                entry_prices[a] = float(prices[a])  # nouveau prix d'entrée

        prev_prices = prices

    # ── Résumé ────────────────────────────────────────────────
    print(f"\n  {len(snapshots)} jours simulés\n")

    from collections import Counter
    regimes = [s.regime.name for s in snapshots]
    counts  = Counter(regimes)
    for r, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {r:8s} : {c:4d} jours ({c/len(snapshots):.1%})")

    non_suspended = [s for s in snapshots if not s.trading_suspended]
    print(f"\n  Scaling moyen (hors suspension) : {np.mean([s.risk_scaling for s in non_suspended]):.3f}x")
    print(f"  Scaling min                     : {np.min([s.risk_scaling for s in snapshots]):.3f}x")
    print(f"  DD max                          : {np.min([s.current_drawdown for s in snapshots]):.2%}")
    print(f"  Jours suspendus                 : {sum(1 for s in snapshots if s.trading_suspended)}")

    # ── Snapshots pendant la phase de stress ──────────────────
    print("\n  === Phase de stress (jours 200-250) ===")
    for s in snapshots[200:220:5]:
        print(
            f"  {s.date.date()} | "
            f"{s.regime.name:8s} | "
            f"score: {s.regime_score:.2f} "
            f"[T:{s.trend_score:.2f} V:{s.vol_score:.2f} "
            f"C:{s.corr_score:.2f} D:{s.dd_score:.2f}] | "
            f"scaling: {s.risk_scaling:.2f}x | "
            f"DD: {s.current_drawdown:.1%}"
        )
