# ============================================================
# risk_manager.py — Risk Management Layer
# ============================================================
# RÔLE DE CE FICHIER :
# Surveiller et contrôler le risque à tous les niveaux :
#   1. Position individuelle (stop-loss par actif)
#   2. Portefeuille global (drawdown, levier, volatilité)
#   3. Régime de marché (détection de stress)
#
# C'est le "disjoncteur" de la stratégie — il protège le
# capital en conditions de marché extrêmes.
#
# PHILOSOPHIE :
# Le risk manager ne cherche PAS à maximiser les rendements.
# Il cherche à SURVIVRE aux périodes difficiles pour être
# présent quand les opportunités reviennent.
#
# DÉPENDANCES :
#   pip install pandas numpy scipy
# ============================================================

import os
import sys
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    INITIAL_CAPITAL,
    MAX_PORTFOLIO_DRAWDOWN,
    MAX_POSITION_LOSS,
    MAX_LEVERAGE,
    TARGET_VOLATILITY,
    RISK_FREE_RATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================
# ÉNUMÉRATIONS — États et Régimes
# ============================================================
# Une Enum (énumération) est un ensemble de constantes nommées.
# Avantage vs strings : Python vérifie à la compilation que
# la valeur est valide. "NORML" passerait inaperçu en string,
# MarketRegime.NORML lèverait une erreur immédiate.

class MarketRegime(Enum):
    """
    Régimes de marché détectés par le risk manager.

    NORMAL   : conditions standard, exposition pleine
    ELEVATED : volatilité élevée, exposition réduite de 25%
    STRESS   : stress systémique, exposition réduite de 50%
    CRISIS   : crise majeure, exposition minimale (25%)
    """
    NORMAL   = "NORMAL"
    ELEVATED = "ELEVATED"
    STRESS   = "STRESS"
    CRISIS   = "CRISIS"


class RiskStatus(Enum):
    """
    Statut global du risk manager.

    ACTIVE      : trading normal autorisé
    REDUCED     : exposition réduite suite à un signal de risque
    SUSPENDED   : trading suspendu (drawdown max atteint)
    """
    ACTIVE    = "ACTIVE"
    REDUCED   = "REDUCED"
    SUSPENDED = "SUSPENDED"


# ============================================================
# DATACLASS — Rapport de risque
# ============================================================
# Un dataclass est une classe Python spécialisée pour stocker
# des données structurées. Plus lisible qu'un dict, plus léger
# qu'une classe complète.
# @dataclass génère automatiquement __init__, __repr__, etc.

@dataclass
class RiskReport:
    """
    Rapport de risque complet généré à chaque vérification.
    Contient toutes les métriques et décisions du risk manager.
    """
    date                : pd.Timestamp
    status              : RiskStatus
    regime              : MarketRegime

    # Métriques de drawdown
    current_drawdown    : float = 0.0
    max_drawdown        : float = 0.0
    peak_value          : float = 0.0

    # Métriques de volatilité
    realized_vol        : float = 0.0
    vol_ratio           : float = 1.0   # vol_court / vol_long
    risk_scaling        : float = 1.0   # facteur d'ajustement expo

    # Métriques de corrélation
    avg_correlation     : float = 0.0

    # Positions à fermer (stop-loss individuels)
    positions_to_close  : list = field(default_factory=list)

    # Alertes et messages
    alerts              : list = field(default_factory=list)

    def add_alert(self, message: str, level: str = "WARNING"):
        """Ajoute une alerte au rapport."""
        self.alerts.append({"level": level, "message": message})
        if level == "CRITICAL":
            logger.critical(f"🚨 {message}")
        elif level == "WARNING":
            logger.warning(f"⚠️  {message}")
        else:
            logger.info(f"ℹ️  {message}")


# ============================================================
# CLASSE PRINCIPALE : RiskManager
# ============================================================

class RiskManager:
    """
    Surveille et contrôle le risque de la stratégie momentum.

    Responsabilités :
      - Tracker la valeur du portefeuille et les drawdowns
      - Détecter les régimes de marché (normal/stress/crise)
      - Appliquer le vol targeting dynamique
      - Déclencher les stop-loss (position et portefeuille)
      - Produire des rapports de risque détaillés
    """

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        """
        Initialise le risk manager.

        Args:
            initial_capital : capital de départ en USD
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # Peak value = valeur maximale atteinte par le portefeuille
        # Utilisé pour calculer le drawdown courant
        # Au départ, le peak = capital initial
        self.peak_value = initial_capital

        # Historique de la valeur du portefeuille
        # Liste de tuples (date, valeur) — permet de recalculer
        # n'importe quelle métrique historique
        self.portfolio_values = []

        # Statut et régime courants
        self.status = RiskStatus.ACTIVE
        self.regime = MarketRegime.NORMAL

        # Prix d'entrée par position
        # Dictionnaire { symbol: prix_d_entrée }
        # Utilisé pour calculer les P&L individuels
        self.entry_prices = {}

        # Historique des rapports de risque
        self.risk_reports = []

        # Compteur de jours en régime de stress
        # Si on est en stress depuis trop longtemps → on réduit encore plus
        self.stress_days_count = 0

        logger.info(
            f"RiskManager initialisé | "
            f"Capital: {initial_capital:,.0f}$ | "
            f"Max DD: {MAX_PORTFOLIO_DRAWDOWN:.0%} | "
            f"Max pos loss: {MAX_POSITION_LOSS:.0%}"
        )

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 1 : Mise à jour de la valeur du portefeuille
    # ─────────────────────────────────────────────────────────
    def update_portfolio_value(
        self,
        date: pd.Timestamp,
        portfolio_value: float
    ):
        """
        Met à jour la valeur courante du portefeuille et
        recalcule le drawdown.

        CONCEPT — DRAWDOWN :
        DD(t) = (V(t) - Peak(t)) / Peak(t)

        Le peak est le MAXIMUM historique de la valeur du portefeuille.
        Il ne peut qu'augmenter ou rester stable — jamais baisser.
        Quand la valeur courante est sous le peak → drawdown négatif.

        UNDERWATER PERIOD :
        La période entre le début d'un drawdown et son recovery
        s'appelle la "période underwater" ou "recovery period".
        En institution, on surveille autant la profondeur du drawdown
        que sa durée — être en drawdown 2 ans est très problématique
        même si le DD maximum n'est "que" de 15%.

        Args:
            date            : date courante
            portfolio_value : valeur totale du portefeuille en $
        """
        self.current_capital = portfolio_value
        self.portfolio_values.append((date, portfolio_value))

        # Mise à jour du peak (ne peut qu'augmenter)
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        # Calcul du drawdown courant
        current_drawdown = (portfolio_value - self.peak_value) / self.peak_value

        # Vérification du circuit breaker global
        if current_drawdown < -MAX_PORTFOLIO_DRAWDOWN:
            self.status = RiskStatus.SUSPENDED
            logger.critical(
                f"🚨 CIRCUIT BREAKER DÉCLENCHÉ | "
                f"Drawdown: {current_drawdown:.1%} | "
                f"Seuil: -{MAX_PORTFOLIO_DRAWDOWN:.0%} | "
                f"Trading SUSPENDU"
            )

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 2 : Calcul du drawdown historique complet
    # ─────────────────────────────────────────────────────────
    def compute_drawdown_series(self) -> pd.Series:
        """
        Calcule la série temporelle complète des drawdowns.

        UTILITÉ :
        Permet de visualiser l'évolution du drawdown dans le temps
        et d'identifier les périodes underwater.

        ALGORITHME :
          1. Calculer le peak cumulatif (expanding maximum)
          2. DD(t) = (V(t) - peak(t)) / peak(t)

        Returns:
            Series des drawdowns (valeurs négatives ou nulles)
        """
        if not self.portfolio_values:
            return pd.Series(dtype=float)

        dates  = [x[0] for x in self.portfolio_values]
        values = [x[1] for x in self.portfolio_values]

        values_series = pd.Series(values, index=dates)

        # expanding().max() = maximum cumulatif
        # À chaque date t, on prend le max de toutes les valeurs
        # de la date 0 jusqu'à t
        # C'est exactement la définition du "peak"
        peak_series = values_series.expanding().max()

        # Drawdown = écart relatif entre valeur courante et peak
        drawdown_series = (values_series - peak_series) / peak_series

        return drawdown_series

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 3 : Calcul de la volatilité réalisée
    # ─────────────────────────────────────────────────────────
    def compute_realized_volatility(
        self,
        returns: pd.Series,
        window_short: int = 20,
        window_long: int  = 252
    ) -> dict:
        """
        Calcule la volatilité réalisée sur deux horizons.

        COURT TERME (20j) : capture la vol récente, réactive aux chocs
        LONG TERME (252j) : capture la vol "normale" historique

        Le RATIO vol_court / vol_long est notre indicateur de stress :
          Ratio > 1.0 : vol récente > vol normale → tension
          Ratio > 1.5 : vol récente 50% au-dessus de la normale → stress
          Ratio > 2.0 : vol récente double de la normale → crise

        ANNUALISATION :
        Vol journalière × √252 = Vol annualisée
        Cette formule suppose des rendements i.i.d. (indépendants
        et identiquement distribués) — hypothèse simplificatrice
        mais standard dans l'industrie.

        Args:
            returns      : Series des rendements journaliers du portfolio
            window_short : fenêtre courte (défaut 20 jours)
            window_long  : fenêtre longue (défaut 252 jours)

        Returns:
            dict { vol_short, vol_long, vol_ratio }
        """
        if len(returns) < window_short:
            return {"vol_short": TARGET_VOLATILITY,
                    "vol_long":  TARGET_VOLATILITY,
                    "vol_ratio": 1.0}

        # Volatilité annualisée sur fenêtre courte
        # std() calcule l'écart-type des rendements → × √252 pour annualiser
        vol_short = returns.tail(window_short).std() * np.sqrt(252)

        # Volatilité annualisée sur fenêtre longue
        vol_long = returns.tail(window_long).std() * np.sqrt(252)

        # Protection contre division par zéro
        if vol_long < 0.001:
            vol_ratio = 1.0
        else:
            vol_ratio = vol_short / vol_long

        return {
            "vol_short" : vol_short,
            "vol_long"  : vol_long,
            "vol_ratio" : vol_ratio
        }

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 4 : Calcul du risk scaling (vol targeting)
    # ─────────────────────────────────────────────────────────
    def compute_risk_scaling(self, realized_vol: float) -> float:
        """
        Calcule le facteur de scaling de l'exposition basé sur
        la volatilité réalisée.

        FORMULE :
            scaling = TARGET_VOL / realized_vol

        INTUITION :
        Si la vol réalisée est 2x la vol cible → on réduit l'expo de 50%
        Si la vol réalisée est 0.5x la vol cible → on augmente l'expo de 100%

        CONTRAINTES :
        On clippe le scaling entre 0.25 et 1.5 :
          - Minimum 0.25 : on garde toujours au moins 25% d'exposition
          - Maximum 1.50 : on ne dépasse pas 1.5x notre exposition normale
                           (évite l'over-leveraging en période très calme)

        Args:
            realized_vol : volatilité réalisée annualisée

        Returns:
            float : facteur multiplicatif pour l'exposition [0.25, 1.5]
        """
        if realized_vol < 0.001:
            return 1.0

        raw_scaling = TARGET_VOLATILITY / realized_vol

        # Clip entre 0.25 et 1.5
        # np.clip(valeur, min, max)
        scaling = float(np.clip(raw_scaling, 0.25, 1.50))

        if scaling < 0.75:
            logger.info(
                f"  Vol scaling réduit : {scaling:.2f}x "
                f"(vol réalisée: {realized_vol:.1%} vs cible: {TARGET_VOLATILITY:.1%})"
            )

        return scaling

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 5 : Détection du régime de marché
    # ─────────────────────────────────────────────────────────
    def detect_market_regime(
        self,
        vol_ratio: float,
        avg_correlation: float,
        current_drawdown: float
    ) -> MarketRegime:
        """
        Détecte le régime de marché courant.

        RÈGLES DE CLASSIFICATION :

        CRISIS   : drawdown > 15% OU (vol_ratio > 2.5 ET corr > 0.7)
                   → Les actifs chutent ensemble massivement
                   → Exposition réduite à 25%

        STRESS   : vol_ratio > 2.0 OU (vol_ratio > 1.5 ET corr > 0.6)
                   → Stress systémique, corrélations élevées
                   → Exposition réduite à 50%

        ELEVATED : vol_ratio > 1.5 OU drawdown > 8%
                   → Tensions sur le marché
                   → Exposition réduite à 75%

        NORMAL   : conditions standard
                   → Exposition pleine

        POURQUOI LES CORRÉLATIONS ?
        En temps de crise, toutes les corrélations convergent vers 1.
        C'est le phénomène "all correlations go to 1 in a crisis".
        Un portefeuille diversifié perd sa diversification exactement
        quand on en a le plus besoin. C'est pourquoi on surveille
        les corrélations comme indicateur avancé de crise.

        Args:
            vol_ratio        : vol_court / vol_long
            avg_correlation  : corrélation moyenne entre actifs
            current_drawdown : drawdown courant du portefeuille

        Returns:
            MarketRegime enum
        """
        # CRISIS
        if (abs(current_drawdown) > 0.15 or
            (vol_ratio > 2.5 and avg_correlation > 0.70)):
            if self.regime != MarketRegime.CRISIS:
                logger.critical(
                    f"🚨 RÉGIME CRISIS DÉTECTÉ | "
                    f"DD: {current_drawdown:.1%} | "
                    f"Vol ratio: {vol_ratio:.2f} | "
                    f"Corr: {avg_correlation:.2f}"
                )
            self.stress_days_count += 1
            return MarketRegime.CRISIS

        # STRESS
        if (vol_ratio > 2.0 or
            (vol_ratio > 1.5 and avg_correlation > 0.60)):
            if self.regime != MarketRegime.STRESS:
                logger.warning(
                    f"⚠️  RÉGIME STRESS DÉTECTÉ | "
                    f"Vol ratio: {vol_ratio:.2f} | "
                    f"Corr: {avg_correlation:.2f}"
                )
            self.stress_days_count += 1
            return MarketRegime.STRESS

        # ELEVATED
        if vol_ratio > 1.5 or abs(current_drawdown) > 0.08:
            if self.regime != MarketRegime.ELEVATED:
                logger.warning(
                    f"⚠️  RÉGIME ELEVATED DÉTECTÉ | "
                    f"Vol ratio: {vol_ratio:.2f} | "
                    f"DD: {current_drawdown:.1%}"
                )
            self.stress_days_count = max(0, self.stress_days_count - 1)
            return MarketRegime.ELEVATED

        # NORMAL
        self.stress_days_count = max(0, self.stress_days_count - 1)
        return MarketRegime.NORMAL

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 6 : Facteur d'exposition par régime
    # ─────────────────────────────────────────────────────────
    def get_regime_exposure_factor(self, regime: MarketRegime) -> float:
        """
        Retourne le facteur multiplicatif d'exposition selon le régime.

        NORMAL   → 1.00 (exposition pleine)
        ELEVATED → 0.75 (réduit de 25%)
        STRESS   → 0.50 (réduit de 50%)
        CRISIS   → 0.25 (exposition minimale)

        Args:
            regime : MarketRegime détecté

        Returns:
            float : facteur multiplicatif [0.25, 1.0]
        """
        factors = {
            MarketRegime.NORMAL   : 1.00,
            MarketRegime.ELEVATED : 0.75,
            MarketRegime.STRESS   : 0.50,
            MarketRegime.CRISIS   : 0.25,
        }
        return factors[regime]

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 7 : Vérification des stop-loss individuels
    # ─────────────────────────────────────────────────────────
    def check_position_stop_losses(
        self,
        current_positions: dict,
        current_prices: pd.Series
    ) -> list:
        """
        Vérifie si des positions individuelles doivent être fermées
        suite à un stop-loss.

        STOP-LOSS INDIVIDUEL :
        Si une position perd plus de MAX_POSITION_LOSS (15%) depuis
        son prix d'entrée → on la ferme immédiatement.

        ASYMÉTRIE LONG/SHORT :
          Long  : perte si prix baisse → stop si prix < entrée × (1 - seuil)
          Short : perte si prix monte  → stop si prix > entrée × (1 + seuil)

        POURQUOI UN STOP-LOSS INDIVIDUEL ?
        Même avec un bon signal de portefeuille, un actif peut subir
        un choc idiosyncratique (fraude comptable, désastre, FDA rejection).
        Le stop-loss individuel protège contre ce risque spécifique
        qui n'est pas capturé par le signal momentum.

        Args:
            current_positions : dict { symbol: quantité }
            current_prices    : Series des prix actuels

        Returns:
            list des symboles dont la position doit être fermée
        """
        positions_to_close = []

        for symbol, qty in current_positions.items():
            if qty == 0:
                continue

            if symbol not in self.entry_prices:
                # On n'a pas de prix d'entrée enregistré → on skip
                continue

            if symbol not in current_prices.index:
                continue

            entry_price   = self.entry_prices[symbol]
            current_price = current_prices[symbol]

            if entry_price <= 0:
                continue

            # Calcul du P&L en % depuis l'entrée
            pnl_pct = (current_price - entry_price) / entry_price

            # Pour une position SHORT, le P&L est inversé
            if qty < 0:
                pnl_pct = -pnl_pct

            # Vérification du stop-loss
            if pnl_pct < -MAX_POSITION_LOSS:
                positions_to_close.append(symbol)
                logger.warning(
                    f"🛑 STOP-LOSS déclenché : {symbol} | "
                    f"P&L: {pnl_pct:.1%} | "
                    f"Entrée: {entry_price:.2f}$ | "
                    f"Courant: {current_price:.2f}$"
                )

        return positions_to_close

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 8 : Calcul de la corrélation moyenne
    # ─────────────────────────────────────────────────────────
    def compute_average_correlation(
        self,
        returns_matrix: pd.DataFrame,
        window: int = 60
    ) -> float:
        """
        Calcule la corrélation moyenne entre tous les actifs.

        CONCEPT :
        La matrice de corrélation est une matrice N×N où chaque
        élément (i,j) représente la corrélation entre l'actif i
        et l'actif j sur une fenêtre de temps.

        On prend la MOYENNE de tous les éléments hors-diagonale
        (la diagonale = corrélation d'un actif avec lui-même = 1).

        INTERPRÉTATION :
        ρ_moy = 0.1-0.3 : corrélations normales en marché diversifié
        ρ_moy = 0.4-0.6 : corrélations élevées, tensions
        ρ_moy > 0.7     : corrélations de crise, "all-to-one"

        Args:
            returns_matrix : DataFrame des rendements (dates × actifs)
            window         : fenêtre de calcul en jours (défaut 60)

        Returns:
            float : corrélation moyenne [0, 1]
        """
        if len(returns_matrix) < window or returns_matrix.shape[1] < 2:
            return 0.0

        # On prend les window dernières observations
        recent_returns = returns_matrix.tail(window)

        # .corr() calcule la matrice de corrélation de Pearson
        corr_matrix = recent_returns.corr()

        # On extrait seulement les éléments hors-diagonale
        # np.triu(matrix, k=1) : triangle supérieur sans la diagonale
        # k=1 exclut la diagonale principale
        n = corr_matrix.shape[0]
        upper_triangle = corr_matrix.values[
            np.triu_indices(n, k=1)
        ]

        if len(upper_triangle) == 0:
            return 0.0

        avg_corr = float(np.nanmean(upper_triangle))
        return max(0.0, avg_corr)  # clip à 0 minimum

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 9 : Vérification complète du risque
    # ─────────────────────────────────────────────────────────
    def check_risk(
        self,
        date: pd.Timestamp,
        portfolio_value: float,
        weights: pd.Series,
        returns_matrix: pd.DataFrame,
        current_positions: dict,
        current_prices: pd.Series,
    ) -> RiskReport:
        """
        Effectue la vérification complète du risque.
        C'est la méthode principale appelée par le backtest
        à chaque date de rebalancement.

        PIPELINE :
          1. Mise à jour valeur portefeuille → drawdown
          2. Calcul volatilité réalisée → vol ratio
          3. Calcul corrélation moyenne
          4. Détection régime de marché
          5. Risk scaling (vol targeting)
          6. Check stop-loss individuels
          7. Vérification levier
          8. Génération du rapport

        Args:
            date              : date courante
            portfolio_value   : valeur totale du portefeuille
            weights           : poids actuels du portefeuille
            returns_matrix    : DataFrame des rendements (historique)
            current_positions : dict { symbol: quantité }
            current_prices    : Series des prix actuels

        Returns:
            RiskReport complet
        """
        # Initialisation du rapport
        report = RiskReport(
            date   = date,
            status = self.status,
            regime = self.regime,
        )

        # ── ÉTAPE 1 : Mise à jour valeur et drawdown ──────────
        self.update_portfolio_value(date, portfolio_value)

        drawdown_series  = self.compute_drawdown_series()
        current_drawdown = float(drawdown_series.iloc[-1]) if len(drawdown_series) > 0 else 0.0
        max_drawdown     = float(drawdown_series.min()) if len(drawdown_series) > 0 else 0.0

        report.current_drawdown = current_drawdown
        report.max_drawdown     = max_drawdown
        report.peak_value       = self.peak_value

        # ── ÉTAPE 2 : Volatilité réalisée ─────────────────────
        # On calcule les rendements du portefeuille depuis l'historique
        if len(self.portfolio_values) > 1:
            port_values = pd.Series(
                [x[1] for x in self.portfolio_values],
                index=[x[0] for x in self.portfolio_values]
            )
            port_returns = port_values.pct_change().dropna()
        else:
            port_returns = pd.Series(dtype=float)

        vol_metrics = self.compute_realized_volatility(port_returns)

        report.realized_vol = vol_metrics["vol_short"]
        report.vol_ratio    = vol_metrics["vol_ratio"]

        # ── ÉTAPE 3 : Corrélation moyenne ─────────────────────
        if not returns_matrix.empty:
            avg_corr = self.compute_average_correlation(returns_matrix)
        else:
            avg_corr = 0.0

        report.avg_correlation = avg_corr

        # ── ÉTAPE 4 : Détection du régime ─────────────────────
        new_regime  = self.detect_market_regime(
            vol_ratio        = vol_metrics["vol_ratio"],
            avg_correlation  = avg_corr,
            current_drawdown = current_drawdown
        )

        # Log si changement de régime
        if new_regime != self.regime:
            report.add_alert(
                f"Changement de régime : {self.regime.value} → {new_regime.value}",
                level="WARNING" if new_regime != MarketRegime.NORMAL else "INFO"
            )

        self.regime  = new_regime
        report.regime = new_regime

        # ── ÉTAPE 5 : Risk scaling ─────────────────────────────
        vol_scaling    = self.compute_risk_scaling(vol_metrics["vol_short"])
        regime_factor  = self.get_regime_exposure_factor(new_regime)

        # Le scaling final combine les deux facteurs
        # Vol scaling : ajuste selon la vol réalisée vs cible
        # Regime factor : réduit en période de stress
        final_scaling = vol_scaling * regime_factor
        final_scaling = float(np.clip(final_scaling, 0.10, 1.50))

        report.risk_scaling = final_scaling

        # ── ÉTAPE 6 : Stop-loss individuels ───────────────────
        stops = self.check_position_stop_losses(
            current_positions, current_prices
        )
        report.positions_to_close = stops

        if stops:
            report.add_alert(
                f"Stop-loss déclenché sur : {', '.join(stops)}",
                level="WARNING"
            )

        # ── ÉTAPE 7 : Vérification du levier ──────────────────
        if not weights.empty:
            gross_exposure = weights.abs().sum()
            if gross_exposure > MAX_LEVERAGE:
                report.add_alert(
                    f"Levier excessif détecté : {gross_exposure:.2f}x > {MAX_LEVERAGE}x",
                    level="WARNING"
                )

        # ── ÉTAPE 8 : Vérification circuit breaker ────────────
        if self.status == RiskStatus.SUSPENDED:
            report.add_alert(
                f"TRADING SUSPENDU — Drawdown max atteint : {current_drawdown:.1%}",
                level="CRITICAL"
            )
            report.status = RiskStatus.SUSPENDED

        elif final_scaling < 0.75:
            report.status = RiskStatus.REDUCED
            self.status   = RiskStatus.REDUCED
        else:
            report.status = RiskStatus.ACTIVE
            self.status   = RiskStatus.ACTIVE

        # ── Sauvegarde du rapport ──────────────────────────────
        self.risk_reports.append(report)

        # ── Log résumé ─────────────────────────────────────────
        logger.info(
            f"🛡️  Risk Check {date.date()} | "
            f"Status: {report.status.value} | "
            f"Régime: {report.regime.value} | "
            f"DD: {current_drawdown:.1%} | "
            f"Vol: {vol_metrics['vol_short']:.1%} | "
            f"Scaling: {final_scaling:.2f}x"
        )

        return report

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 10 : Appliquer le risk scaling aux poids
    # ─────────────────────────────────────────────────────────
    def apply_risk_scaling_to_weights(
        self,
        weights: pd.Series,
        risk_report: RiskReport,
    ) -> pd.Series:
        """
        Applique le facteur de risk scaling aux poids du portefeuille.

        C'est la méthode qui "connecte" le risk manager au portfolio
        constructor. Elle prend les poids calculés par le portfolio
        et les ajuste selon les conditions de marché.

        EXEMPLE :
          Poids AAPL avant scaling : +0.10 (10% du capital)
          Risk scaling : 0.50 (régime STRESS)
          Poids AAPL après scaling : +0.05 (5% du capital)

        Elle ferme aussi les positions en stop-loss.

        Args:
            weights     : Series des poids calculés par le portfolio
            risk_report : rapport de risque généré par check_risk()

        Returns:
            Series des poids ajustés
        """
        adjusted_weights = weights.copy()

        # Application du scaling global
        adjusted_weights = adjusted_weights * risk_report.risk_scaling

        # Fermeture des positions en stop-loss
        for symbol in risk_report.positions_to_close:
            if symbol in adjusted_weights.index:
                adjusted_weights[symbol] = 0.0
                logger.info(f"  Position {symbol} mise à 0 (stop-loss)")

        # Si trading suspendu → tout à zéro
        if risk_report.status == RiskStatus.SUSPENDED:
            adjusted_weights = adjusted_weights * 0.0
            logger.critical("🚨 Tous les poids mis à 0 (trading suspendu)")

        if risk_report.risk_scaling < 1.0:
            logger.info(
                f"  Poids scalés : {risk_report.risk_scaling:.2f}x "
                f"(régime {risk_report.regime.value})"
            )

        return adjusted_weights

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 11 : Historique des rapports en DataFrame
    # ─────────────────────────────────────────────────────────
    def get_risk_history(self) -> pd.DataFrame:
        """
        Retourne l'historique complet des métriques de risque
        sous forme de DataFrame.

        Utile pour analyser l'évolution du risque sur la durée
        du backtest et identifier les périodes problématiques.
        """
        if not self.risk_reports:
            return pd.DataFrame()

        rows = []
        for r in self.risk_reports:
            rows.append({
                "date"            : r.date,
                "status"          : r.status.value,
                "regime"          : r.regime.value,
                "drawdown"        : r.current_drawdown,
                "max_drawdown"    : r.max_drawdown,
                "realized_vol"    : r.realized_vol,
                "vol_ratio"       : r.vol_ratio,
                "risk_scaling"    : r.risk_scaling,
                "avg_correlation" : r.avg_correlation,
                "nb_stops"        : len(r.positions_to_close),
                "nb_alerts"       : len(r.alerts),
            })

        return pd.DataFrame(rows).set_index("date")


# ============================================================
# SCRIPT PRINCIPAL — Test du risk manager
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  TEST — RiskManager")
    print("=" * 60)

    np.random.seed(42)

    # ── Simulation d'un historique de portefeuille ─────────────
    # On simule 2 ans de valeurs quotidiennes avec :
    #   - Phase 1 (1 an)  : rendements normaux (+8% annuel)
    #   - Phase 2 (3 mois): stress avec drawdown important (-20%)
    #   - Phase 3 (9 mois): recovery progressive

    n_days = 504
    dates  = pd.bdate_range(start="2022-01-01", periods=n_days)

    # Génération des rendements par phase
    returns_phase1 = np.random.normal(0.08/252, 0.12/np.sqrt(252), 252)
    returns_phase2 = np.random.normal(-0.30/252, 0.35/np.sqrt(252), 63)
    returns_phase3 = np.random.normal(0.15/252, 0.15/np.sqrt(252), 189)

    all_returns  = np.concatenate([returns_phase1, returns_phase2, returns_phase3])
    port_values  = INITIAL_CAPITAL * np.exp(np.cumsum(all_returns))

    # ── Simulation d'une matrice de rendements d'actifs ────────
    n_assets = 8
    assets   = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "XOM", "GC", "CL"]

    asset_returns = pd.DataFrame(
        np.random.normal(0.0003, 0.015, (n_days, n_assets)),
        index=dates, columns=assets
    )

    # ── Initialisation du Risk Manager ─────────────────────────
    rm = RiskManager(initial_capital=INITIAL_CAPITAL)

    # Prix d'entrée simulés pour les stop-loss
    rm.entry_prices = {a: 100.0 for a in assets}

    # Prix courants (on simule une baisse sur certains actifs)
    current_prices = pd.Series({
        "AAPL": 95.0,   # -5%  → pas de stop
        "MSFT": 88.0,   # -12% → pas de stop (< 15%)
        "GOOGL": 82.0,  # -18% → STOP-LOSS déclenché
        "AMZN": 100.0,  # 0%   → pas de stop
        "JPM": 105.0,   # +5%  → pas de stop
        "XOM": 78.0,    # -22% → STOP-LOSS déclenché
        "GC": 103.0,    # +3%  → pas de stop
        "CL": 91.0,     # -9%  → pas de stop
    })

    current_positions = {a: 100 for a in assets}
    weights = pd.Series({a: 0.10 for a in assets})

    # ── Test sur plusieurs dates ────────────────────────────────
    print("\n🔍 Vérifications de risque successives...\n")

    # Phase 1 : marché normal
    for i in [0, 50, 100, 150, 200, 251]:
        date  = dates[i]
        value = port_values[i]
        rm.check_risk(
            date              = date,
            portfolio_value   = value,
            weights           = weights,
            returns_matrix    = asset_returns.iloc[:i+1],
            current_positions = current_positions,
            current_prices    = current_prices,
        )

    # Phase 2 : stress (drawdown important)
    for i in [252, 280, 314]:
        date  = dates[i]
        value = port_values[i]
        rm.check_risk(
            date              = date,
            portfolio_value   = value,
            weights           = weights,
            returns_matrix    = asset_returns.iloc[:i+1],
            current_positions = current_positions,
            current_prices    = current_prices,
        )

    # ── Historique du risque ────────────────────────────────────
    print("\n📊 Historique des métriques de risque :")
    history = rm.get_risk_history()
    print(history.round(4).to_string())

    # ── Test apply_risk_scaling ─────────────────────────────────
    print("\n\n🔧 Test du risk scaling sur les poids...")
    last_report = rm.risk_reports[-1]
    print(f"  Régime        : {last_report.regime.value}")
    print(f"  Risk scaling  : {last_report.risk_scaling:.2f}x")
    print(f"  Stops         : {last_report.positions_to_close}")

    adjusted = rm.apply_risk_scaling_to_weights(weights, last_report)
    print(f"\n  Poids avant : {weights.to_dict()}")
    print(f"  Poids après : {adjusted.round(4).to_dict()}")

    # ── Drawdown series ─────────────────────────────────────────
    print("\n📉 Drawdown max historique :")
    dd_series = rm.compute_drawdown_series()
    print(f"  Max Drawdown  : {dd_series.min():.2%}")
    print(f"  DD final      : {dd_series.iloc[-1]:.2%}")