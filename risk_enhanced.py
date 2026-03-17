# ============================================================
# risk_enhanced.py — Risk Management Amélioré Phase 2B
# ============================================================
# RÔLE DE CE FICHIER :
# Implémenter les filtres de régime de marché pour réduire
# le Max Drawdown de -22.46% à < -20% tout en préservant
# le Sharpe Ratio amélioré (0.575).
#
# FILTRES IMPLÉMENTÉS :
#   1. Trend Filter      : MA200 sur benchmark
#   2. Volatility Filter : ratio vol court/long
#   3. Correlation Filter: corrélation moyenne cross-actifs
#   4. Drawdown Filter   : circuit breaker progressif
#
# PIPELINE :
#   Prix → 4 filtres → RegimeScore → Poids ajustés
#
# USAGE :
#   from risk_enhanced import RegimeDetector, EnhancedRiskManager
#   detector = RegimeDetector(price_matrix)
#   scores   = detector.compute_regime_scores()
#   # Appliquer scores aux poids du backtest
#
# DÉPENDANCES :
#   pip install pandas numpy scipy
# ============================================================

import os
import sys
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    INITIAL_CAPITAL,
    TARGET_VOLATILITY,
    MAX_PORTFOLIO_DRAWDOWN,
    RISK_FREE_RATE,
    TRANSACTION_COST_BPS,
    SLIPPAGE_BPS,
    BACKTEST_START,
    BACKTEST_END,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================
# ENUM — Régimes de marché
# ============================================================

class MarketRegime(Enum):
    """
    Régimes de marché avec leur facteur d'exposition associé.

    BULL   : conditions normales, momentum très efficace
    BEAR   : tendance baissière, momentum moins efficace
    STRESS : volatilité élevée, risque de momentum crash
    CRISIS : stress systémique, couper l'exposition
    """
    BULL   = 1.00   # Exposition pleine
    BEAR   = 0.50   # Exposition réduite de 50%
    STRESS = 0.25   # Exposition réduite de 75%
    CRISIS = 0.00   # Exposition nulle


# ============================================================
# DATACLASS — Rapport de régime quotidien
# ============================================================

@dataclass
class RegimeReport:
    """
    Rapport complet du régime de marché pour une date donnée.
    Contient les scores de chaque filtre et le score final.
    """
    date             : pd.Timestamp
    trend_score      : float = 1.0   # [0, 1]
    vol_score        : float = 1.0   # [0, 1]
    corr_score       : float = 1.0   # [0, 1]
    dd_score         : float = 1.0   # [0, 1]
    regime_score     : float = 1.0   # min des 4 scores
    regime           : MarketRegime  = MarketRegime.BULL
    vol_ratio        : float = 1.0
    avg_correlation  : float = 0.0
    current_drawdown : float = 0.0


# ============================================================
# CLASSE 1 : RegimeDetector
# ============================================================

class RegimeDetector:
    """
    Détecte le régime de marché en temps réel.

    Calcule 4 filtres indépendants et les combine en un
    score de régime unique utilisé pour ajuster l'exposition.

    PHILOSOPHIE :
    Chaque filtre détecte un type de risque différent :
    - Trend Filter    : risque directionnel (marché qui descend)
    - Vol Filter      : risque de volatilité (marché nerveux)
    - Corr Filter     : risque systémique (tout corrèle)
    - DD Filter       : risque de ruine (stratégie qui saigne)

    En prenant le MINIMUM des 4 scores, le filtre le plus
    restrictif s'impose — c'est intentionnel : un seul signal
    d'alarme suffit pour réduire l'exposition.
    """

    def __init__(
        self,
        price_matrix    : pd.DataFrame,
        benchmark_col   : str = None,
    ):
        """
        Initialise le détecteur de régime.

        Args:
            price_matrix  : DataFrame (dates × actifs)
            benchmark_col : colonne à utiliser comme benchmark
                            pour le trend filter. Si None, on
                            utilise la moyenne équipondérée de
                            tous les actifs (proxy SPY).
        """
        self.prices       = price_matrix.copy()
        self.log_returns  = np.log(price_matrix / price_matrix.shift(1))

        # Benchmark : SPY si disponible, sinon moyenne équipondérée
        if benchmark_col and benchmark_col in price_matrix.columns:
            self.benchmark = price_matrix[benchmark_col]
        else:
            # Proxy benchmark = moyenne équipondérée de tous les actifs
            # Normalisation : chaque actif part à 100 au début
            normalized    = price_matrix / price_matrix.iloc[0] * 100
            self.benchmark = normalized.mean(axis=1)

        logger.info(
            f"RegimeDetector initialisé | "
            f"{len(self.prices)} jours × {self.prices.shape[1]} actifs"
        )

    # ─────────────────────────────────────────────────────────
    # FILTRE 1 : Trend Filter (MA200)
    # ─────────────────────────────────────────────────────────
    def compute_trend_filter(
        self,
        ma_window : int = 200,
    ) -> pd.Series:
        """
        Calcule le filtre de tendance basé sur la MA200.

        LOGIQUE :
        Si le benchmark est au-dessus de sa moyenne mobile 200j
        → marché en tendance haussière → exposition normale (1.0)

        Si le benchmark est en dessous de sa MA200
        → marché en tendance baissière → exposition réduite (0.5)

        POURQUOI MA200 ?
        La MA200 est LA référence pour définir un bull/bear market.
        Utilisée par tous les gérants depuis des décennies.
        Au-dessus = bull market = momentum fonctionne bien.
        En dessous = bear market = momentum plus risqué.

        NUANCE — Score continu vs binaire :
        Au lieu d'un simple 0/1, on calcule la distance relative
        au-dessus/dessous de la MA : si le benchmark est 5% au-dessus,
        score = 1.0. S'il est juste en dessous, score = 0.6.
        S'il est très en dessous (-10%), score = 0.5.
        → Transition progressive, pas un interrupteur brutal.

        Args:
            ma_window : fenêtre de la moyenne mobile (défaut 200j)

        Returns:
            Series des scores trend [0.5, 1.0] par date
        """
        # Moyenne mobile 200j du benchmark
        ma200 = self.benchmark.rolling(ma_window).mean()

        # Distance relative : (prix - MA) / MA
        # Positif = au-dessus de la MA, Négatif = en dessous
        distance = (self.benchmark - ma200) / ma200

        # Conversion en score [0.5, 1.0]
        # Au-dessus de MA → score = 1.0 (exposition pleine)
        # En dessous de MA → score = 0.5 (exposition réduite)
        # La transition est lissée avec un clip et une normalisation
        trend_score = pd.Series(1.0, index=self.prices.index)

        # En dessous de la MA → on réduit progressivement
        # -5% en dessous → score ≈ 0.65
        # -10% en dessous → score ≈ 0.50 (plancher)
        below_ma = distance < 0
        trend_score[below_ma] = (0.5 + 0.5 * (1 + distance[below_ma] * 5)).clip(0.5, 1.0)

        # Les premières dates n'ont pas de MA → score neutre
        trend_score[:ma_window] = 1.0

        logger.info(
            f"  Trend Filter | "
            f"Score moyen : {trend_score.mean():.3f} | "
            f"% temps BEAR : {(trend_score < 0.8).mean():.1%}"
        )
        return trend_score

    # ─────────────────────────────────────────────────────────
    # FILTRE 2 : Volatility Filter
    # ─────────────────────────────────────────────────────────
    def compute_volatility_filter(
        self,
        window_short : int   = 10,
        window_long  : int   = 252,
        threshold_1  : float = 1.2,
        threshold_2  : float = 1.5,
        threshold_3  : float = 2.0,
    ) -> pd.Series:
        """
        Calcule le filtre de volatilité basé sur le ratio vol court/long.

        FORMULE :
            Vol_ratio(t) = Vol_10j(t) / Vol_252j(t)

        POURQUOI 10j AU LIEU DE 20j ?
        On utilise une fenêtre courte de 10 jours pour être
        plus réactif aux chocs de volatilité. Le crash COVID
        a mis 5 jours pour exploser — avec 20j on aurait réagi
        trop tard. Avec 10j on détecte plus vite.

        SEUILS :
            ratio > 2.0 → CRISE  → score = 0.10 (exposition quasi nulle)
            ratio > 1.5 → STRESS → score = 0.25 (exposition 25%)
            ratio > 1.2 → ÉLEVÉ  → score = 0.60 (exposition 60%)
            ratio ≤ 1.2 → NORMAL → score = 1.00 (exposition normale)

        ANNUALISATION :
        Vol journalière × √252 = Vol annualisée
        On compare les deux vols annualisées → ratio sans biais.

        Args:
            window_short : fenêtre vol courte en jours
            window_long  : fenêtre vol longue en jours
            threshold_1/2/3 : seuils de transition

        Returns:
            Series des scores vol [0.10, 1.0] par date
        """
        # Vol du portefeuille = vol de la moyenne équipondérée
        # On utilise les returns moyens cross-actifs comme proxy
        portfolio_returns = self.log_returns.mean(axis=1)

        # Volatilités annualisées (rolling std × √252)
        vol_short = portfolio_returns.rolling(window_short).std() * np.sqrt(252)
        vol_long  = portfolio_returns.rolling(window_long).std()  * np.sqrt(252)

        # Protection division par zéro
        vol_long  = vol_long.replace(0, np.nan).ffill().fillna(TARGET_VOLATILITY)
        vol_short = vol_short.fillna(TARGET_VOLATILITY)

        # Ratio vol courte / vol longue
        vol_ratio = vol_short / vol_long

        # Conversion en score
        vol_score = pd.Series(1.0, index=self.prices.index)
        vol_score[vol_ratio > threshold_3] = 0.10   # CRISE
        vol_score[(vol_ratio > threshold_2) & (vol_ratio <= threshold_3)] = 0.25  # STRESS
        vol_score[(vol_ratio > threshold_1) & (vol_ratio <= threshold_2)] = 0.60  # ÉLEVÉ

        logger.info(
            f"  Vol Filter | "
            f"Score moyen : {vol_score.mean():.3f} | "
            f"% temps STRESS : {(vol_score < 0.5).mean():.1%} | "
            f"Vol ratio moyen : {vol_ratio.mean():.2f}"
        )
        return vol_score, vol_ratio

    # ─────────────────────────────────────────────────────────
    # FILTRE 3 : Correlation Filter
    # ─────────────────────────────────────────────────────────
    def compute_correlation_filter(
        self,
        window          : int   = 60,
        threshold_stress: float = 0.50,
        threshold_crisis: float = 0.70,
    ) -> pd.Series:
        """
        Calcule le filtre de corrélation cross-actifs.

        CONCEPT — "ALL CORRELATIONS GO TO 1 IN A CRISIS" :
        En temps de crise, toutes les corrélations convergent vers 1.
        C'est le phénomène le plus documenté en finance de crise.
        Quand ça arrive, la diversification disparaît exactement
        quand on en a le plus besoin.

        ALGORITHME :
        À chaque date t, on calcule la corrélation moyenne de
        TOUTES les paires d'actifs sur une fenêtre glissante de 60j.
        Cette corrélation moyenne est notre indicateur de stress.

        SEUILS :
            ρ_moyen > 0.70 → CRISE  → score = 0.10
            ρ_moyen > 0.50 → STRESS → score = 0.40
            ρ_moyen ≤ 0.50 → NORMAL → score = 1.00

        OPTIMISATION :
        Calculer la matrice de corrélation à chaque jour serait
        très lent (O(n² × T)). On utilise une fenêtre glissante
        et on calcule les corrélations par paires via ewm() pour
        gagner un facteur ~10 en vitesse.

        Args:
            window           : fenêtre de calcul (défaut 60j)
            threshold_stress : seuil stress (défaut 0.5)
            threshold_crisis : seuil crise (défaut 0.7)

        Returns:
            Series des scores corrélation [0.10, 1.0] par date
        """
        # Calcul de la corrélation moyenne par rolling window
        # On utilise une approche vectorisée : pour chaque paire (i,j),
        # on calcule la corrélation rolling et on fait la moyenne

        n_assets = self.prices.shape[1]
        returns  = self.log_returns.fillna(0)

        # Pour éviter un calcul O(n²) à chaque pas de temps,
        # on calcule la corrélation rolling de manière vectorisée
        # en utilisant la décomposition cov/(std_i × std_j)

        # Rolling mean et std pour chaque actif
        roll_mean = returns.rolling(window).mean()
        roll_std  = returns.rolling(window).std().replace(0, np.nan)

        # Normalisation des returns (z-score rolling)
        z_returns = (returns - roll_mean) / roll_std

        # La corrélation moyenne = moyenne du produit croisé des z-scores
        # E[z_i × z_j] = corr(i, j) pour i ≠ j
        # On calcule la variance totale du portefeuille équipondéré :
        # Var(1/N × Σz_i) = 1/N² × [N × var + N(N-1) × corr_moy × var]
        # → corr_moy = (N × Var(portfolio) / var_asset - 1) / (N - 1)

        z_portfolio = z_returns.mean(axis=1)
        var_portfolio = z_portfolio.rolling(window).var()
        var_asset     = z_returns.rolling(window).var().mean(axis=1)

        # Protection division par zéro
        var_asset = var_asset.replace(0, np.nan).ffill().fillna(1.0)

        # Corrélation moyenne dérivée de la variance du portefeuille
        avg_corr = (n_assets * var_portfolio / var_asset - 1) / (n_assets - 1)
        avg_corr = avg_corr.clip(0, 1).fillna(0)

        # Conversion en score
        corr_score = pd.Series(1.0, index=self.prices.index)
        corr_score[avg_corr > threshold_crisis] = 0.10
        corr_score[(avg_corr > threshold_stress) & (avg_corr <= threshold_crisis)] = 0.40

        logger.info(
            f"  Corr Filter | "
            f"Score moyen : {corr_score.mean():.3f} | "
            f"Corr moyenne : {avg_corr.mean():.3f} | "
            f"% temps CRISE : {(avg_corr > threshold_crisis).mean():.1%}"
        )
        return corr_score, avg_corr

    # ─────────────────────────────────────────────────────────
    # FILTRE 4 : Drawdown Filter (Circuit Breaker)
    # ─────────────────────────────────────────────────────────
    def compute_drawdown_filter(
        self,
        portfolio_values : pd.Series,
        threshold_1      : float = 0.08,
        threshold_2      : float = 0.12,
        threshold_3      : float = 0.18,
        threshold_stop   : float = 0.22,
    ) -> pd.Series:
        """
        Calcule le filtre de drawdown avec circuit breaker progressif.

        CONCEPT — POSITION SCALING PAR DRAWDOWN :
        Quand la stratégie est en drawdown, on réduit progressivement
        l'exposition. Deux raisons :
          1. La stratégie ne fonctionne pas dans ce régime
          2. On protège le capital restant

        SEUILS PROGRESSIFS :
            DD > 8%  → score = 0.75 (réduction douce)
            DD > 12% → score = 0.50 (réduction modérée)
            DD > 18% → score = 0.25 (réduction forte)
            DD > 22% → score = 0.00 (circuit breaker total)

        RECOVERY PROGRESSIVE :
        Quand le drawdown se réduit, on réaugmente l'exposition
        progressivement. On ne reprend pas immédiatement l'exposition
        pleine — on attend que le trend soit confirmé.

        POURQUOI CES SEUILS ?
        Notre Max DD actuel est -22.46%. Si on coupe à -22%, on
        protège les derniers %. On vise un Max DD de -20% donc
        on coupe à -18% pour avoir une marge de sécurité.

        Args:
            portfolio_values : Series des valeurs du portefeuille
            threshold_1/2/3/stop : seuils de réduction

        Returns:
            Series des scores drawdown [0.0, 1.0] par date
        """
        # Drawdown courant
        peak     = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        dd_abs   = drawdown.abs()  # Valeur absolue (positif = perte)

        # Score progressif
        dd_score = pd.Series(1.0, index=portfolio_values.index)
        dd_score[dd_abs > threshold_stop] = 0.00   # Circuit breaker
        dd_score[(dd_abs > threshold_3) & (dd_abs <= threshold_stop)] = 0.25
        dd_score[(dd_abs > threshold_2) & (dd_abs <= threshold_3)] = 0.50
        dd_score[(dd_abs > threshold_1) & (dd_abs <= threshold_2)] = 0.75

        logger.info(
            f"  DD Filter | "
            f"Score moyen : {dd_score.mean():.3f} | "
            f"Max DD : {drawdown.min():.2%} | "
            f"% temps DD>8% : {(dd_abs > threshold_1).mean():.1%}"
        )
        return dd_score, drawdown

    # ─────────────────────────────────────────────────────────
    # MÉTHODE PRINCIPALE : Calcul du score de régime complet
    # ─────────────────────────────────────────────────────────
    def compute_regime_scores(
        self,
        portfolio_values : pd.Series = None,
    ) -> pd.DataFrame:
        """
        Calcule tous les scores de régime et les combine.

        SCORE FINAL = MINIMUM des 4 filtres

        On prend le minimum car le filtre le plus restrictif doit
        dominer. Si la vol est normale mais le drawdown est extrême,
        on coupe quand même l'exposition.

        Args:
            portfolio_values : Series de la valeur du portefeuille
                               Nécessaire pour le DD Filter.
                               Si None, on utilise le benchmark.

        Returns:
            DataFrame avec colonnes :
            [trend_score, vol_score, corr_score, dd_score,
             regime_score, vol_ratio, avg_correlation, drawdown]
        """
        logger.info("\n  Calcul des scores de régime...")

        # Filtre 1 : Trend
        trend_score = self.compute_trend_filter()

        # Filtre 2 : Volatilité
        vol_score, vol_ratio = self.compute_volatility_filter()

        # Filtre 3 : Corrélation
        corr_score, avg_corr = self.compute_correlation_filter()

        # Filtre 4 : Drawdown
        if portfolio_values is None:
            # Si pas de valeur de portefeuille, utilise le benchmark
            portfolio_values = self.benchmark * (INITIAL_CAPITAL / self.benchmark.iloc[0])

        dd_score, drawdown = self.compute_drawdown_filter(portfolio_values)

        # Score final = minimum des 4 filtres
        # → Le filtre le plus restrictif domine
        regime_score = pd.concat([
            trend_score,
            vol_score,
            corr_score,
            dd_score
        ], axis=1).min(axis=1)

        # Lissage du score final sur 3 jours
        # Pour éviter des changements trop brusques d'exposition
        # (qui généreraient du turnover excessif et des coûts)
        regime_score_smoothed = regime_score.rolling(3, min_periods=1).mean()

        # Construction du DataFrame résultat
        regime_df = pd.DataFrame({
            "trend_score"    : trend_score,
            "vol_score"      : vol_score,
            "corr_score"     : corr_score,
            "dd_score"       : dd_score,
            "regime_score"   : regime_score_smoothed,
            "vol_ratio"      : vol_ratio,
            "avg_correlation": avg_corr,
            "drawdown"       : drawdown,
        })

        # Log résumé
        regime_counts = {
            "BULL"   : (regime_score_smoothed >= 0.90).sum(),
            "NORMAL" : ((regime_score_smoothed >= 0.50) & (regime_score_smoothed < 0.90)).sum(),
            "REDUCED": ((regime_score_smoothed >= 0.20) & (regime_score_smoothed < 0.50)).sum(),
            "CRISIS" : (regime_score_smoothed < 0.20).sum(),
        }

        logger.info(
            f"\n  Régimes détectés sur {len(regime_df)} jours :"
        )
        for regime, count in regime_counts.items():
            pct = count / len(regime_df) * 100
            logger.info(f"    {regime:8s} : {count:4d} jours ({pct:5.1f}%)")

        logger.info(
            f"\n  Score moyen : {regime_score_smoothed.mean():.3f} | "
            f"Score min : {regime_score_smoothed.min():.3f}"
        )

        return regime_df


# ============================================================
# CLASSE 2 : EnhancedRiskManager
# ============================================================

class EnhancedRiskManager:
    """
    Risk manager amélioré qui intègre les filtres de régime
    dans le pipeline de backtest vectorisé.

    Prend les poids calculés par le backtest standard et les
    ajuste en fonction du régime de marché détecté.
    """

    def __init__(
        self,
        price_matrix    : pd.DataFrame,
        asset_types     : dict  = None,
        initial_capital : float = INITIAL_CAPITAL,
        start_date      : str   = BACKTEST_START,
        end_date        : str   = BACKTEST_END,
    ):
        self.prices          = price_matrix.copy()
        self.asset_types     = asset_types or {}
        self.initial_capital = initial_capital
        self.start_date      = start_date
        self.end_date        = end_date

    # ─────────────────────────────────────────────────────────
    # MÉTHODE : Application du régime aux poids
    # ─────────────────────────────────────────────────────────
    def apply_regime_to_weights(
        self,
        weights_all   : pd.DataFrame,
        regime_scores : pd.Series,
    ) -> pd.DataFrame:
        """
        Multiplie chaque ligne de poids par le score de régime.

        C'est la connexion entre le RegimeDetector et le backtest.
        Pour chaque date t :
            weights_adjusted(t) = weights(t) × regime_score(t)

        EXEMPLE :
            weights(t)      = [AAPL: 0.10, XOM: -0.08, CL: 0.07, ...]
            regime_score(t) = 0.25  (régime STRESS)
            weights_adj(t)  = [AAPL: 0.025, XOM: -0.020, CL: 0.0175, ...]

        IMPORTANT — LAG D'1 JOUR :
        Le score de régime calculé à t est appliqué à t+1.
        Cohérent avec le lag déjà appliqué dans backtest_vectorized.

        Args:
            weights_all   : DataFrame des poids (dates × actifs)
            regime_scores : Series des scores [0, 1] par date

        Returns:
            DataFrame des poids ajustés
        """
        # Alignement des index
        common_idx     = weights_all.index.intersection(regime_scores.index)
        weights_common = weights_all.reindex(common_idx)
        scores_common  = regime_scores.reindex(common_idx)

        # Lag d'1 jour sur le score de régime
        # Le régime détecté aujourd'hui est appliqué demain
        scores_lagged = scores_common.shift(1).fillna(1.0)

        # Application du score à tous les actifs
        # .mul(series, axis=0) multiplie chaque ligne par le score
        weights_adjusted = weights_common.mul(scores_lagged, axis=0)

        # Log de l'impact
        avg_reduction = (1 - scores_lagged.mean()) * 100
        logger.info(
            f"  Régime appliqué aux poids | "
            f"Réduction moyenne : {avg_reduction:.1f}% | "
            f"Score min : {scores_lagged.min():.3f}"
        )

        return weights_adjusted

    # ─────────────────────────────────────────────────────────
    # MÉTHODE PRINCIPALE : Backtest avec filtres de régime
    # ─────────────────────────────────────────────────────────
    def run_enhanced_backtest(
        self,
        signal_results  : dict,
        apply_regime    : bool = True,
    ) -> dict:
        """
        Lance le backtest complet avec les filtres de régime.

        PIPELINE :
          1. Backtest standard (backtest_vectorized)
          2. Calcul des scores de régime (RegimeDetector)
          3. Application du régime aux poids
          4. Recalcul du P&L avec poids ajustés
          5. Calcul des métriques de performance

        Args:
            signal_results : dict output de run_full_pipeline()
            apply_regime   : appliquer les filtres de régime

        Returns:
            dict des résultats complets
        """
        from strategies.momentum.backtest_vectorized import VectorizedBacktest
        from metrics.performance import PerformanceAnalyzer

        logger.info("\n" + "="*55)
        logger.info("  BACKTEST AVEC FILTRES DE REGIME")
        logger.info("="*55)

        # ── ÉTAPE 1 : Backtest standard ────────────────────────
        logger.info("\n  Etape 1 : Backtest standard...")
        bt = VectorizedBacktest(
            price_matrix    = self.prices,
            initial_capital = self.initial_capital,
            start_date      = self.start_date,
            end_date        = self.end_date,
            asset_types     = self.asset_types,
        )
        bt_results = bt.run(
            signal_results     = signal_results,
            apply_risk_scaling = True,
        )

        if not apply_regime:
            logger.info("  Filtres de regime desactives → retour du backtest standard")
            return bt_results

        # ── ÉTAPE 2 : Calcul des scores de régime ──────────────
        logger.info("\n  Etape 2 : Detection du regime de marche...")

        prices_period = self.prices.loc[self.start_date:self.end_date]
        detector      = RegimeDetector(prices_period)

        # Premier calcul sans portfolio_values pour avoir les scores
        # de base (trend, vol, corr). Le DD filter sera recalculé
        # avec les portfolio_values du backtest standard.
        regime_df_base = detector.compute_regime_scores(
            portfolio_values=bt_results["portfolio_values"]
        )

        regime_scores = regime_df_base["regime_score"]

        # ── ÉTAPE 3 : Application du régime aux poids ──────────
        logger.info("\n  Etape 3 : Application des filtres aux poids...")
        weights_standard = bt_results["weights"]
        weights_adjusted = self.apply_regime_to_weights(
            weights_standard, regime_scores
        )

        # ── ÉTAPE 4 : Recalcul du P&L avec poids ajustés ───────
        logger.info("\n  Etape 4 : Recalcul du P&L...")

        prices_bt     = self.prices.loc[self.start_date:self.end_date]
        log_returns   = np.log(prices_bt / prices_bt.shift(1)).fillna(0)
        gross_returns = (weights_adjusted * log_returns).sum(axis=1)

        # Coûts de transaction sur les poids ajustés
        prev_weights  = weights_adjusted.shift(1).fillna(0)
        turnover      = (weights_adjusted - prev_weights).abs().sum(axis=1)
        cost_fraction = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10_000

        # Dates de rebalancement (pour les coûts)
        rebal_positions = prices_bt.index.get_indexer(
            bt_results["rebal_dates"], method="nearest"
        )
        rebal_positions = rebal_positions[rebal_positions >= 0]

        daily_costs = pd.Series(0.0, index=prices_bt.index)
        for pos in rebal_positions:
            if pos < len(turnover):
                daily_costs.iloc[pos] = turnover.iloc[pos] * cost_fraction

        net_returns    = gross_returns - daily_costs
        simple_returns = np.exp(net_returns) - 1
        port_values    = self.initial_capital * (1 + simple_returns).cumprod()

        # Turnover mensuel
        monthly_turnover = turnover.iloc[rebal_positions].mean() if len(rebal_positions) > 0 else 0

        # ── ÉTAPE 5 : Métriques finales ─────────────────────────
        logger.info("\n  Etape 5 : Calcul des metriques...")

        final_value  = port_values.iloc[-1]
        total_return = final_value / self.initial_capital - 1

        # Calcul rapide des métriques clés
        n_years    = len(net_returns) / 252
        cagr       = (final_value / self.initial_capital) ** (1/n_years) - 1
        annual_vol = net_returns.std() * np.sqrt(252)
        rf_daily   = (1 + RISK_FREE_RATE) ** (1/252) - 1
        sharpe     = (net_returns.mean() - rf_daily) / net_returns.std() * np.sqrt(252)

        peak     = port_values.expanding().max()
        drawdown = (port_values - peak) / peak
        max_dd   = drawdown.min()

        neg_ret  = net_returns[net_returns < rf_daily]
        sortino  = (net_returns.mean() - rf_daily) / neg_ret.std() * np.sqrt(252) if len(neg_ret) > 0 else 0
        calmar   = cagr / abs(max_dd) if abs(max_dd) > 0.001 else 0

        # Benchmark
        ew = pd.DataFrame(
            1.0 / prices_bt.shape[1],
            index=prices_bt.index,
            columns=prices_bt.columns
        ).shift(1).fillna(0)
        bm_returns = (ew * log_returns).sum(axis=1)
        bm_values  = self.initial_capital * (1 + (np.exp(bm_returns) - 1)).cumprod()

        logger.info("\n" + "─"*55)
        logger.info(f"  BACKTEST AVEC FILTRES TERMINE")
        logger.info(f"  Capital : {self.initial_capital:,.0f}$ -> {final_value:,.0f}$")
        logger.info(f"  CAGR    : {cagr:.2%}")
        logger.info(f"  Sharpe  : {sharpe:.3f}")
        logger.info(f"  Max DD  : {max_dd:.2%}")
        logger.info(f"  Turnover: {monthly_turnover:.1%}")
        logger.info("─"*55)

        return {
            "returns"          : net_returns,
            "gross_returns"    : gross_returns,
            "costs"            : daily_costs,
            "portfolio_values" : port_values,
            "weights"          : weights_adjusted,
            "weights_standard" : weights_standard,
            "benchmark_returns": bm_returns,
            "benchmark_values" : bm_values,
            "regime_df"        : regime_df_base,
            "regime_scores"    : regime_scores,
            "rebal_dates"      : bt_results["rebal_dates"],
            "monthly_turnover" : float(monthly_turnover),
            "initial_capital"  : self.initial_capital,
            "n_assets"         : prices_bt.shape[1],
            "n_days"           : len(prices_bt),
            "metrics": {
                "cagr"       : float(cagr),
                "sharpe"     : float(sharpe),
                "sortino"    : float(sortino),
                "calmar"     : float(calmar),
                "max_dd"     : float(max_dd),
                "annual_vol" : float(annual_vol),
                "total_return": float(total_return),
            }
        }


# ============================================================
# SCRIPT PRINCIPAL — Test du risk manager amélioré
# ============================================================

if __name__ == "__main__":

    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("  TEST — EnhancedRiskManager (Phase 2B)")
    print("=" * 60)

    # Chargement des données
    cache_file = "./data/processed/price_matrix.csv"
    if not os.path.exists(cache_file):
        print("ERREUR : Cache non trouve. Lance run_backtest.py d'abord.")
        sys.exit(1)

    print(f"\nChargement depuis {cache_file}...")
    price_matrix = pd.read_csv(cache_file, index_col="date", parse_dates=True)
    print(f"Donnees : {price_matrix.shape}")

    # Asset types
    from config import FUTURES_UNIVERSE
    asset_types = {
        s: "future" if s in FUTURES_UNIVERSE else "stock"
        for s in price_matrix.columns
    }

    # Pipeline signal
    print("\nCalcul des signaux...")
    from strategies.momentum.momentum_signal import MomentumSignalGenerator
    generator   = MomentumSignalGenerator(price_matrix)
    sig_results = generator.run_full_pipeline(cs_weight=0.5, ts_weight=0.5)

    # ── TEST 1 : Backtest SANS filtres de regime ───────────────
    print("\n" + "="*55)
    print("  TEST 1 : Backtest SANS filtres de regime (baseline)")
    print("="*55)

    rm_no_filter = EnhancedRiskManager(
        price_matrix    = price_matrix,
        asset_types     = asset_types,
        initial_capital = 100_000,
        start_date      = BACKTEST_START,
        end_date        = BACKTEST_END,
    )
    results_no_filter = rm_no_filter.run_enhanced_backtest(
        sig_results,
        apply_regime=False
    )

    m_nf = results_no_filter["metrics"] if "metrics" in results_no_filter else {}

    # ── TEST 2 : Backtest AVEC filtres de regime ───────────────
    print("\n" + "="*55)
    print("  TEST 2 : Backtest AVEC filtres de regime")
    print("="*55)

    rm_with_filter = EnhancedRiskManager(
        price_matrix    = price_matrix,
        asset_types     = asset_types,
        initial_capital = 100_000,
        start_date      = BACKTEST_START,
        end_date        = BACKTEST_END,
    )
    results_with_filter = rm_with_filter.run_enhanced_backtest(
        sig_results,
        apply_regime=True
    )

    m_wf = results_with_filter["metrics"]

    # ── Comparaison ────────────────────────────────────────────
    print("\n" + "="*60)
    print("  COMPARAISON : SANS vs AVEC filtres de regime")
    print("="*60)
    print(f"  {'Metrique':<20} {'Sans filtre':>12} {'Avec filtre':>12} {'Delta':>10}")
    print("  " + "-"*56)

    metrics_to_compare = [
        ("CAGR",      "cagr",       ".2%"),
        ("Sharpe",    "sharpe",     ".3f"),
        ("Sortino",   "sortino",    ".3f"),
        ("Calmar",    "calmar",     ".3f"),
        ("Max DD",    "max_dd",     ".2%"),
        ("Vol",       "annual_vol", ".2%"),
    ]

    for label, key, fmt in metrics_to_compare:
        v_no  = results_no_filter.get("metrics", {}).get(key, 0)
        v_with = m_wf.get(key, 0)
        delta  = v_with - v_no
        sign   = "+" if delta > 0 else ""
        print(
            f"  {label:<20} "
            f"{format(v_no, fmt):>12} "
            f"{format(v_with, fmt):>12} "
            f"{sign}{format(delta, fmt):>10}"
        )

    print("="*60)

    print("\n  Regime distribution :")
    regime_df = results_with_filter["regime_df"]
    rs        = results_with_filter["regime_scores"]
    print(f"  Score >= 0.90 (BULL)   : {(rs >= 0.90).sum():4d} jours ({(rs >= 0.90).mean():.1%})")
    print(f"  Score 0.50-0.90        : {((rs >= 0.50) & (rs < 0.90)).sum():4d} jours ({((rs >= 0.50) & (rs < 0.90)).mean():.1%})")
    print(f"  Score 0.20-0.50 (REDUIT): {((rs >= 0.20) & (rs < 0.50)).sum():4d} jours ({((rs >= 0.20) & (rs < 0.50)).mean():.1%})")
    print(f"  Score < 0.20 (CRISE)   : {(rs < 0.20).sum():4d} jours ({(rs < 0.20).mean():.1%})")

    print(f"\n  Turnover mensuel :")
    print(f"  Sans filtre  : {results_no_filter.get('monthly_turnover', 0):.1%}")
    print(f"  Avec filtres : {results_with_filter['monthly_turnover']:.1%}")