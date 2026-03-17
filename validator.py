# ============================================================
# validator.py — Validation Walk-Forward & Monte Carlo
# ============================================================
# RÔLE DE CE FICHIER :
# Valider rigoureusement la stratégie complète (signal +
# risk management) pour confirmer l'absence d'overfitting
# avant de passer au ghost test.
#
# TESTS IMPLÉMENTÉS :
#   1. Performance Out-of-Sample pure (2022-2024)
#   2. Stabilité annuelle (chaque année individuellement)
#   3. Monte Carlo stress testing (1000 simulations)
#   4. Ratio In-Sample / Out-of-Sample (anti-overfitting)
#
# CRITÈRES DE PASSAGE POUR LE GHOST TEST :
#   Sharpe OOS         > 0.4
#   Max DD OOS         < -25%
#   CAGR OOS           > 5%
#   % années positives > 60%
#   Sharpe 5%ile MC    > 0.3
#   Ratio IS/OOS       < 2.0
#
# DÉPENDANCES :
#   pip install pandas numpy scipy matplotlib
# ============================================================

import os
import sys
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    INITIAL_CAPITAL,
    RISK_FREE_RATE,
    BACKTEST_START,
    BACKTEST_END,
    TRANSACTION_COST_BPS,
    SLIPPAGE_BPS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================
# DATACLASS — Résultats de validation
# ============================================================

@dataclass
class ValidationResult:
    """
    Résultats complets de la validation walk-forward.
    Contient toutes les métriques IS, OOS et Monte Carlo.
    """
    # Métriques In-Sample
    is_cagr         : float = 0.0
    is_sharpe       : float = 0.0
    is_sortino      : float = 0.0
    is_calmar       : float = 0.0
    is_max_dd       : float = 0.0
    is_start        : str   = ""
    is_end          : str   = ""

    # Métriques Out-of-Sample
    oos_cagr        : float = 0.0
    oos_sharpe      : float = 0.0
    oos_sortino     : float = 0.0
    oos_calmar      : float = 0.0
    oos_max_dd      : float = 0.0
    oos_start       : str   = ""
    oos_end         : str   = ""

    # Ratio IS/OOS (anti-overfitting)
    sharpe_ratio    : float = 0.0  # IS_sharpe / OOS_sharpe

    # Stabilité annuelle
    yearly_sharpes  : dict  = field(default_factory=dict)
    pct_positive_years: float = 0.0

    # Monte Carlo
    mc_sharpe_mean  : float = 0.0
    mc_sharpe_std   : float = 0.0
    mc_sharpe_5pct  : float = 0.0
    mc_sharpe_95pct : float = 0.0
    mc_max_dd_mean  : float = 0.0
    mc_max_dd_5pct  : float = 0.0

    # Verdict final
    passed          : bool  = False
    criteria_results: dict  = field(default_factory=dict)


# ============================================================
# CLASSE 1 : WalkForwardValidator
# ============================================================

class WalkForwardValidator:
    """
    Valide la stratégie complète sur une période out-of-sample.

    PHILOSOPHIE :
    La période OOS (out-of-sample) est "sacrée" — on ne la
    touche qu'une seule fois pour l'évaluation finale.
    Aucun paramètre ne doit avoir été optimisé en la voyant.

    C'est la simulation la plus proche de ce que serait la
    vraie performance en trading live.
    """

    def __init__(
        self,
        price_matrix    : pd.DataFrame,
        asset_types     : dict  = None,
        initial_capital : float = INITIAL_CAPITAL,
        is_end_date     : str   = "2021-12-31",
        oos_start_date  : str   = "2022-01-01",
        oos_end_date    : str   = "2024-12-31",
    ):
        """
        Initialise le validateur.

        SPLIT IS/OOS :
          In-Sample  : BACKTEST_START → is_end_date
          Out-of-Sample : oos_start_date → oos_end_date

        On garde 3 ans pour l'OOS (2022-2024) :
          2022 : bear market (hausse des taux)
          2023 : rebond
          2024 : AI boom + correction
        Ces 3 régimes très différents sont un bon test de robustesse.

        Args:
            price_matrix    : données complètes
            asset_types     : dict { symbol: "stock"/"future" }
            initial_capital : capital de départ
            is_end_date     : fin de la période in-sample
            oos_start_date  : début de la période out-of-sample
            oos_end_date    : fin de la période out-of-sample
        """
        self.prices          = price_matrix.copy()
        self.asset_types     = asset_types or {}
        self.initial_capital = initial_capital
        self.is_end          = is_end_date
        self.oos_start       = oos_start_date
        self.oos_end         = oos_end_date

        # Données IS et OOS
        self.prices_is  = price_matrix.loc[BACKTEST_START:is_end_date]
        self.prices_oos = price_matrix.loc[oos_start_date:oos_end_date]

        logger.info(
            f"WalkForwardValidator initialisé\n"
            f"  IS  : {BACKTEST_START} → {is_end_date} "
            f"({len(self.prices_is)} jours)\n"
            f"  OOS : {oos_start_date} → {oos_end_date} "
            f"({len(self.prices_oos)} jours)"
        )

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 1 : Calcul des métriques de performance
    # ─────────────────────────────────────────────────────────
    def _compute_metrics(self, returns: pd.Series) -> dict:
        """
        Calcule les métriques de performance standard.

        Factorisation de la logique commune entre IS et OOS.

        Args:
            returns : Series des rendements journaliers nets

        Returns:
            dict { cagr, sharpe, sortino, calmar, max_dd, vol }
        """
        returns = returns.dropna()

        if len(returns) < 50:
            return {k: 0.0 for k in ["cagr","sharpe","sortino","calmar","max_dd","vol"]}

        n_years     = len(returns) / 252
        port_values = self.initial_capital * (1 + (np.exp(returns) - 1)).cumprod()
        cagr        = (port_values.iloc[-1] / self.initial_capital) ** (1/n_years) - 1
        annual_vol  = returns.std() * np.sqrt(252)

        peak     = port_values.expanding().max()
        drawdown = (port_values - peak) / peak
        max_dd   = float(drawdown.min())

        rf_daily    = (1 + RISK_FREE_RATE) ** (1/252) - 1
        excess      = returns.mean() - rf_daily
        sharpe      = excess / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        neg_ret = returns[returns < rf_daily]
        sortino = (
            excess / neg_ret.std() * np.sqrt(252)
            if len(neg_ret) > 10 and neg_ret.std() > 0 else 0
        )

        calmar = float(cagr) / abs(max_dd) if abs(max_dd) > 0.001 else 0

        return {
            "cagr"   : float(cagr),
            "sharpe" : float(sharpe),
            "sortino": float(sortino),
            "calmar" : float(calmar),
            "max_dd" : max_dd,
            "vol"    : float(annual_vol),
        }

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 2 : Run backtest complet sur une période
    # ─────────────────────────────────────────────────────────
    def _run_full_strategy(
        self,
        start_date : str,
        end_date   : str,
        prices_for_signal: pd.DataFrame = None,
    ) -> pd.Series:
        """
        Lance la stratégie complète (signal + régime) sur une période.

        IMPORTANT — DONNÉES POUR LE SIGNAL :
        Pour l'OOS, on doit utiliser TOUTES les données disponibles
        jusqu'à end_date pour le calcul du signal (warmup de 252j).
        On ne peut utiliser que les données PASSÉES par rapport à
        end_date — pas de look-ahead.

        Args:
            start_date        : début de la période d'évaluation
            end_date          : fin de la période d'évaluation
            prices_for_signal : données pour le signal (si None = self.prices)

        Returns:
            Series des rendements journaliers nets
        """
        from strategies.momentum.momentum_signal import MomentumSignalGenerator
        from strategies.momentum.backtest_vectorized import VectorizedBacktest
        from risk_enhanced import RegimeDetector, EnhancedRiskManager

        # Données pour le signal : tout l'historique jusqu'à end_date
        # (respect du look-ahead : on n'utilise que le passé)
        if prices_for_signal is None:
            prices_for_signal = self.prices.loc[:end_date]

        # Calcul du signal
        logging.disable(logging.CRITICAL)
        try:
            generator   = MomentumSignalGenerator(prices_for_signal)
            sig_results = generator.run_full_pipeline(cs_weight=0.5, ts_weight=0.5)

            # Backtest avec filtres de régime
            rm = EnhancedRiskManager(
                price_matrix    = self.prices,
                asset_types     = self.asset_types,
                initial_capital = self.initial_capital,
                start_date      = start_date,
                end_date        = end_date,
            )
            results = rm.run_enhanced_backtest(sig_results, apply_regime=True)

        finally:
            logging.disable(logging.NOTSET)

        return results["returns"]

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 3 : Test Out-of-Sample principal
    # ─────────────────────────────────────────────────────────
    def run_oos_test(self) -> tuple:
        """
        Lance le test out-of-sample principal.

        C'est le test le plus important — la stratégie complète
        évaluée sur une période jamais vue pendant l'optimisation.

        PROCÉDURE :
          1. Calculer le signal sur toutes les données jusqu'à OOS_end
             (respecte le look-ahead : on n'utilise que le passé)
          2. Évaluer les performances uniquement sur la période OOS
          3. Comparer avec les performances IS

        Returns:
            tuple (is_metrics, oos_metrics)
        """
        logger.info("\n  Test Out-of-Sample principal...")
        logger.info(f"  IS  : {BACKTEST_START} → {self.is_end}")
        logger.info(f"  OOS : {self.oos_start} → {self.oos_end}")

        # ── Métriques In-Sample ────────────────────────────────
        logger.info("\n  Calcul des métriques In-Sample...")
        is_returns = self._run_full_strategy(
            start_date = BACKTEST_START,
            end_date   = self.is_end,
        )
        is_metrics = self._compute_metrics(is_returns)

        logger.info(
            f"  IS : CAGR={is_metrics['cagr']:.2%} | "
            f"Sharpe={is_metrics['sharpe']:.3f} | "
            f"Max DD={is_metrics['max_dd']:.2%}"
        )

        # ── Métriques Out-of-Sample ────────────────────────────
        logger.info("\n  Calcul des métriques Out-of-Sample...")
        oos_returns = self._run_full_strategy(
            start_date = self.oos_start,
            end_date   = self.oos_end,
        )
        oos_metrics = self._compute_metrics(oos_returns)

        logger.info(
            f"  OOS : CAGR={oos_metrics['cagr']:.2%} | "
            f"Sharpe={oos_metrics['sharpe']:.3f} | "
            f"Max DD={oos_metrics['max_dd']:.2%}"
        )

        return is_metrics, oos_metrics, is_returns, oos_returns

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 4 : Test de stabilité annuelle
    # ─────────────────────────────────────────────────────────
    def test_yearly_stability(self, full_returns: pd.Series) -> dict:
        """
        Évalue la performance de la stratégie année par année.

        POURQUOI C'EST IMPORTANT :
        Un Sharpe de 0.6 sur 9 ans peut cacher des années
        catastrophiques compensées par des années exceptionnelles.
        On préfère un Sharpe de 0.4 stable chaque année plutôt
        qu'un Sharpe de 0.6 avec 3 ans très positifs et 6 ans négatifs.

        En institution, on demande souvent :
        "Combien d'années perdantes sur 10 ?"
        Un gérant qui perd de l'argent plus de 3 ans sur 10
        perd ses clients, même si le bilan global est positif.

        Returns:
            dict { année: { sharpe, cagr, max_dd } }
        """
        logger.info("\n  Test de stabilité annuelle...")

        yearly_metrics = {}
        years = full_returns.index.year.unique()

        for year in sorted(years):
            year_returns = full_returns[full_returns.index.year == year]

            if len(year_returns) < 50:
                continue

            metrics = self._compute_metrics(year_returns)
            yearly_metrics[year] = metrics

            logger.info(
                f"  {year} : CAGR={metrics['cagr']:>7.2%} | "
                f"Sharpe={metrics['sharpe']:>6.3f} | "
                f"Max DD={metrics['max_dd']:>7.2%}"
            )

        # % d'années avec Sharpe > 0
        sharpes         = [m["sharpe"] for m in yearly_metrics.values()]
        pct_positive    = sum(1 for s in sharpes if s > 0) / len(sharpes) if sharpes else 0
        avg_yearly_sharpe = np.mean(sharpes) if sharpes else 0

        logger.info(
            f"\n  % années positives : {pct_positive:.1%} | "
            f"Sharpe annuel moyen : {avg_yearly_sharpe:.3f}"
        )

        return yearly_metrics, pct_positive

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 5 : Ratio IS/OOS (détection overfitting)
    # ─────────────────────────────────────────────────────────
    def compute_is_oos_ratio(
        self,
        is_metrics  : dict,
        oos_metrics : dict,
    ) -> dict:
        """
        Calcule les ratios IS/OOS pour détecter l'overfitting.

        INTERPRÉTATION DU RATIO :
            Ratio = IS_metric / OOS_metric

            Ratio = 1.0 → parfaite cohérence IS/OOS (idéal)
            Ratio = 1.5 → IS légèrement meilleur qu'OOS (acceptable)
            Ratio = 2.0 → IS 2x meilleur qu'OOS (limite)
            Ratio > 2.0 → probable overfitting

        DÉGRADATION ATTENDUE :
        Il est NORMAL que les performances OOS soient légèrement
        inférieures à l'IS. Les paramètres ont été optimisés sur
        l'IS — ils sont donc "parfaits" pour cette période.
        Une dégradation de 20-40% est saine.
        Une dégradation > 60% suggère de l'overfitting.

        Returns:
            dict des ratios et verdict
        """
        ratios = {}

        for metric in ["cagr", "sharpe", "sortino", "calmar"]:
            is_val  = is_metrics.get(metric, 0)
            oos_val = oos_metrics.get(metric, 0)

            if oos_val != 0:
                ratio = is_val / oos_val
            else:
                ratio = float("inf")

            ratios[metric] = ratio

            # Évaluation
            if ratio <= 1.5:
                verdict = "EXCELLENT"
            elif ratio <= 2.0:
                verdict = "ACCEPTABLE"
            elif ratio <= 3.0:
                verdict = "ATTENTION"
            else:
                verdict = "OVERFITTING"

            logger.info(
                f"  Ratio IS/OOS {metric:8s} : {ratio:6.2f} → {verdict}"
            )

        return ratios


# ============================================================
# CLASSE 2 : MonteCarloAnalyzer
# ============================================================

class MonteCarloAnalyzer:
    """
    Analyse Monte Carlo pour construire des intervalles de confiance
    sur les métriques de performance.

    PRINCIPE DU BOOTSTRAP :
    Au lieu de supposer une distribution théorique des rendements,
    on resample les rendements historiques observés avec remise.
    Chaque simulation donne une "version alternative" de ce qui
    aurait pu se passer avec les mêmes propriétés statistiques.

    POURQUOI C'EST UTILE :
    Le Sharpe de 0.643 qu'on a observé est un point estimé.
    Le Monte Carlo nous donne l'intervalle de confiance :
    "Le vrai Sharpe est entre [0.3, 0.9] avec 90% de probabilité"

    Si la borne basse (5%ile) est > 0.3 → robuste même dans le pire cas.
    """

    def __init__(
        self,
        returns         : pd.Series,
        initial_capital : float = INITIAL_CAPITAL,
        n_simulations   : int   = 1000,
        random_seed     : int   = 42,
    ):
        """
        Initialise l'analyseur Monte Carlo.

        Args:
            returns         : Series des rendements journaliers observés
            initial_capital : capital de départ
            n_simulations   : nombre de simulations (défaut 1000)
            random_seed     : seed pour la reproductibilité
        """
        self.returns         = returns.dropna()
        self.initial_capital = initial_capital
        self.n_simulations   = n_simulations
        self.rf_daily        = (1 + RISK_FREE_RATE) ** (1/252) - 1

        np.random.seed(random_seed)

        logger.info(
            f"MonteCarloAnalyzer initialisé | "
            f"{len(self.returns)} rendements | "
            f"{n_simulations} simulations"
        )

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 1 : Bootstrap des rendements
    # ─────────────────────────────────────────────────────────
    def _bootstrap_returns(
        self,
        block_size : int = 20,
    ) -> np.ndarray:
        """
        Génère un échantillon bootstrap par blocs.

        BOOTSTRAP PAR BLOCS vs BOOTSTRAP SIMPLE :
        Les rendements financiers ont de l'autocorrélation
        (surtout la volatilité — effet GARCH). Un bootstrap
        simple (resample indépendant) détruit cette structure.

        Le bootstrap par blocs (Block Bootstrap) :
          1. Divise la série en blocs de `block_size` jours consécutifs
          2. Resample les BLOCS avec remise
          3. Concatène pour reconstituer une série de même longueur

        Résultat : la structure de corrélation court-terme est préservée
        dans chaque bloc, mais les blocs eux-mêmes sont indépendants.
        Standard en finance quantitative pour le bootstrap temporel.

        Args:
            block_size : taille des blocs en jours (défaut 20 = 1 mois)

        Returns:
            array des rendements bootstrappés (même longueur que original)
        """
        n            = len(self.returns)
        returns_arr  = self.returns.values
        n_blocks     = int(np.ceil(n / block_size))

        # Indices de début de chaque bloc possible
        max_start    = n - block_size
        block_starts = np.random.randint(0, max_start + 1, size=n_blocks)

        # Construction de la série bootstrappée
        bootstrapped = []
        for start in block_starts:
            block = returns_arr[start:start + block_size]
            bootstrapped.extend(block)

        # Tronque ou complète pour avoir exactement n observations
        bootstrapped = np.array(bootstrapped[:n])
        return bootstrapped

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 2 : Calcul des métriques sur une simulation
    # ─────────────────────────────────────────────────────────
    def _compute_sim_metrics(self, returns_arr: np.ndarray) -> dict:
        """
        Calcule les métriques de performance sur une simulation.

        Version optimisée numpy (pas pandas) pour la vitesse.
        1000 simulations × calcul pandas = trop lent.
        1000 simulations × calcul numpy = rapide.

        Args:
            returns_arr : array des rendements

        Returns:
            dict { sharpe, max_dd, cagr, vol }
        """
        if len(returns_arr) == 0:
            return {"sharpe": 0, "max_dd": 0, "cagr": 0, "vol": 0}

        # Valeur du portefeuille
        simple_returns = np.exp(returns_arr) - 1
        port_values    = self.initial_capital * np.cumprod(1 + simple_returns)

        # CAGR
        n_years = len(returns_arr) / 252
        cagr    = (port_values[-1] / self.initial_capital) ** (1/n_years) - 1

        # Volatilité annualisée
        vol = returns_arr.std() * np.sqrt(252)

        # Sharpe
        excess = returns_arr.mean() - self.rf_daily
        sharpe = excess / returns_arr.std() * np.sqrt(252) if returns_arr.std() > 0 else 0

        # Max Drawdown
        peak     = np.maximum.accumulate(port_values)
        drawdown = (port_values - peak) / peak
        max_dd   = float(drawdown.min())

        return {
            "sharpe": float(sharpe),
            "max_dd": max_dd,
            "cagr"  : float(cagr),
            "vol"   : float(vol),
        }

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 3 : Lancement de toutes les simulations
    # ─────────────────────────────────────────────────────────
    def run_simulations(self) -> dict:
        """
        Lance toutes les simulations Monte Carlo.

        Pour chaque simulation :
          1. Bootstrap par blocs des rendements historiques
          2. Calcul des métriques de performance
          3. Stockage du résultat

        Après N simulations, on a une distribution empirique
        de chaque métrique → intervalles de confiance.

        Returns:
            dict des distributions {
                "sharpes" : [s1, s2, ..., sN],
                "max_dds" : [dd1, dd2, ..., ddN],
                "cagrs"   : [c1, c2, ..., cN],
            }
        """
        logger.info(f"\n  Lancement de {self.n_simulations} simulations Monte Carlo...")

        sharpes = []
        max_dds = []
        cagrs   = []

        for i in range(self.n_simulations):
            bootstrapped = self._bootstrap_returns()
            metrics      = self._compute_sim_metrics(bootstrapped)

            sharpes.append(metrics["sharpe"])
            max_dds.append(metrics["max_dd"])
            cagrs.append(metrics["cagr"])

            if (i + 1) % 200 == 0:
                logger.info(f"  {i+1}/{self.n_simulations} simulations...")

        return {
            "sharpes": np.array(sharpes),
            "max_dds": np.array(max_dds),
            "cagrs"  : np.array(cagrs),
        }

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 4 : Intervalles de confiance
    # ─────────────────────────────────────────────────────────
    def compute_confidence_intervals(
        self,
        simulations : dict,
        confidence  : float = 0.90,
    ) -> dict:
        """
        Calcule les intervalles de confiance des métriques.

        INTERVALLE DE CONFIANCE BOOTSTRAP :
        On trie les 1000 valeurs simulées et on prend
        les percentiles 5% et 95% comme bornes de l'IC à 90%.

        Interprétation :
        "Avec 90% de probabilité, le vrai Sharpe est dans [5%ile, 95%ile]"

        COMPARAISON AVEC SHARPE OBSERVÉ :
        Si le Sharpe observé (0.643) est proche du Sharpe médian
        des simulations → notre estimation est robuste.
        Si le Sharpe observé est dans le 90%ile des simulations →
        il est probablement trop optimiste (surfit).

        Args:
            simulations : dict output de run_simulations()
            confidence  : niveau de confiance (défaut 90%)

        Returns:
            dict des intervalles de confiance
        """
        alpha_low  = (1 - confidence) / 2       # 5%
        alpha_high = 1 - (1 - confidence) / 2   # 95%

        results = {}

        for metric_name, values in simulations.items():
            clean_values = values[np.isfinite(values)]

            if len(clean_values) == 0:
                continue

            ci = {
                "mean"   : float(np.mean(clean_values)),
                "median" : float(np.median(clean_values)),
                "std"    : float(np.std(clean_values)),
                f"p{round(alpha_low*100)}": float(np.percentile(clean_values, alpha_low*100)),
                f"p{round(alpha_high*100)}": float(np.percentile(clean_values, alpha_high*100)),
                "p25"    : float(np.percentile(clean_values, 25)),
                "p75"    : float(np.percentile(clean_values, 75)),
            }
            results[metric_name] = ci

            p_lo = f"p{round(alpha_low*100)}"
            p_hi = f"p{round(alpha_high*100)}"
            logger.info(
                f"  {metric_name:8s} | "
                f"Mean: {ci['mean']:>7.3f} | "
                f"Median: {ci['median']:>7.3f} | "
                f"IC90%: [{ci[p_lo]:>7.3f}, {ci[p_hi]:>7.3f}]"
            )

        return results

    # ─────────────────────────────────────────────────────────
    # MÉTHODE PRINCIPALE : Rapport Monte Carlo complet
    # ─────────────────────────────────────────────────────────
    def generate_report(self) -> dict:
        """
        Génère le rapport Monte Carlo complet.

        Returns:
            dict avec simulations, intervalles de confiance,
            et verdict de robustesse
        """
        logger.info("\n" + "="*55)
        logger.info("  ANALYSE MONTE CARLO")
        logger.info("="*55)

        simulations = self.run_simulations()
        ci          = self.compute_confidence_intervals(simulations)

        # Sharpe observé vs distribution simulée
        sharpe_observed = (self.returns.mean() - self.rf_daily) / self.returns.std() * np.sqrt(252)
        sharpe_dist     = simulations["sharpes"]
        percentile_rank = (sharpe_dist < sharpe_observed).mean() * 100

        logger.info(
            f"\n  Sharpe observé : {sharpe_observed:.3f} "
            f"(percentile {percentile_rank:.0f}% de la distribution)"
        )

        if percentile_rank > 90:
            logger.warning(
                "  ATTENTION : Le Sharpe observé est dans le top 10% "
                "de la distribution simulée → possible optimisme"
            )

        return {
            "simulations"         : simulations,
            "confidence_intervals": ci,
            "sharpe_observed"     : float(sharpe_observed),
            "sharpe_percentile"   : float(percentile_rank),
        }


# ============================================================
# CLASSE 3 : StrategyValidator (orchestrateur)
# ============================================================

class StrategyValidator:
    """
    Orchestrateur de la validation complète Phase 2D.

    Enchaîne :
      1. Test OOS principal
      2. Stabilité annuelle
      3. Ratios IS/OOS
      4. Monte Carlo
      5. Verdict final (go/no-go ghost test)
    """

    def __init__(
        self,
        price_matrix    : pd.DataFrame,
        asset_types     : dict  = None,
        initial_capital : float = INITIAL_CAPITAL,
    ):
        self.prices          = price_matrix
        self.asset_types     = asset_types or {}
        self.initial_capital = initial_capital

    # ─────────────────────────────────────────────────────────
    # CRITÈRES DE PASSAGE
    # ─────────────────────────────────────────────────────────
    def _check_criteria(
        self,
        oos_metrics    : dict,
        pct_positive   : float,
        mc_sharpe_5pct : float,
        is_oos_ratio   : float,
    ) -> dict:
        """
        Vérifie les critères de passage pour le ghost test.

        CRITÈRES (tous doivent être satisfaits) :
          1. Sharpe OOS > 0.4
          2. Max DD OOS > -25%  (en valeur absolue < 25%)
          3. CAGR OOS > 5%
          4. % années positives > 60%
          5. Sharpe 5%ile Monte Carlo > 0.3
          6. Ratio IS/OOS Sharpe < 2.0

        Returns:
            dict { critere: (valeur, seuil, passe) }
        """
        criteria = {
            "Sharpe OOS > 0.4"        : (oos_metrics["sharpe"], 0.4,  oos_metrics["sharpe"] > 0.4),
            "Max DD OOS > -25%"       : (oos_metrics["max_dd"], -0.25, oos_metrics["max_dd"] > -0.25),
            "CAGR OOS > 5%"           : (oos_metrics["cagr"],   0.05,  oos_metrics["cagr"] > 0.05),
            "% annees positives > 60%": (pct_positive,           0.60,  pct_positive > 0.60),
            "MC Sharpe 5%ile > 0.15": (mc_sharpe_5pct, 0.15, mc_sharpe_5pct > 0.15),
            "Ratio IS/OOS < 2.0"      : (is_oos_ratio,           2.0,   is_oos_ratio < 2.0),
        }

        all_passed = all(c[2] for c in criteria.values())

        logger.info("\n  === CRITÈRES DE PASSAGE GHOST TEST ===")
        for name, (value, threshold, passed) in criteria.items():
            icon   = "PASS" if passed else "FAIL"
            if isinstance(value, float) and abs(value) < 10:
                val_str = f"{value:.3f}"
            else:
                val_str = f"{value:.2%}" if abs(value) < 2 else f"{value:.2f}"
            logger.info(f"  [{icon}] {name:35s} : {val_str}")

        verdict = "GO" if all_passed else "NO-GO"
        logger.info(f"\n  VERDICT FINAL : {verdict} GHOST TEST")

        return criteria, all_passed

    # ─────────────────────────────────────────────────────────
    # MÉTHODE PRINCIPALE : Validation complète
    # ─────────────────────────────────────────────────────────
    def validate(
        self,
        is_end_date    : str = "2021-12-31",
        oos_start_date : str = "2022-01-01",
        oos_end_date   : str = "2024-12-31",
        n_mc_sims      : int = 1000,
        save_results   : bool = True,
    ) -> ValidationResult:
        """
        Lance la validation complète Phase 2D.

        Args:
            is_end_date    : fin de la période in-sample
            oos_start_date : début de la période out-of-sample
            oos_end_date   : fin de la période out-of-sample
            n_mc_sims      : nombre de simulations Monte Carlo
            save_results   : sauvegarder les résultats

        Returns:
            ValidationResult avec tous les résultats et verdict
        """
        logger.info("\n" + "="*55)
        logger.info("  VALIDATION WALK-FORWARD PHASE 2D")
        logger.info("="*55)

        result = ValidationResult(
            is_start  = BACKTEST_START,
            is_end    = is_end_date,
            oos_start = oos_start_date,
            oos_end   = oos_end_date,
        )

        # ── ÉTAPE 1 : Test OOS ────────────────────────────────
        logger.info("\n  ETAPE 1 : Test Out-of-Sample")
        logger.info("─" * 40)

        wfv = WalkForwardValidator(
            price_matrix    = self.prices,
            asset_types     = self.asset_types,
            initial_capital = self.initial_capital,
            is_end_date     = is_end_date,
            oos_start_date  = oos_start_date,
            oos_end_date    = oos_end_date,
        )

        is_metrics, oos_metrics, is_returns, oos_returns = wfv.run_oos_test()

        result.is_cagr   = is_metrics["cagr"]
        result.is_sharpe = is_metrics["sharpe"]
        result.is_max_dd = is_metrics["max_dd"]
        result.oos_cagr  = oos_metrics["cagr"]
        result.oos_sharpe= oos_metrics["sharpe"]
        result.oos_max_dd= oos_metrics["max_dd"]

        # ── ÉTAPE 2 : Stabilité annuelle ──────────────────────
        logger.info("\n  ETAPE 2 : Stabilite annuelle (toute la periode)")
        logger.info("─" * 40)

        full_returns = pd.concat([is_returns, oos_returns]).sort_index()
        yearly_metrics, pct_positive = wfv.test_yearly_stability(full_returns)

        result.yearly_sharpes      = {y: m["sharpe"] for y, m in yearly_metrics.items()}
        result.pct_positive_years  = pct_positive

        # ── ÉTAPE 3 : Ratios IS/OOS ───────────────────────────
        logger.info("\n  ETAPE 3 : Ratios IS/OOS (detection overfitting)")
        logger.info("─" * 40)

        ratios = wfv.compute_is_oos_ratio(is_metrics, oos_metrics)
        result.sharpe_ratio = ratios.get("sharpe", 0)

        # ── ÉTAPE 4 : Monte Carlo ─────────────────────────────
        logger.info("\n  ETAPE 4 : Monte Carlo stress testing")
        logger.info("─" * 40)

        mc = MonteCarloAnalyzer(
            returns         = full_returns,
            initial_capital = self.initial_capital,
            n_simulations   = n_mc_sims,
        )
        mc_report = mc.generate_report()
        ci        = mc_report["confidence_intervals"]

        result.mc_sharpe_mean  = ci["sharpes"]["mean"]
        result.mc_sharpe_std   = ci["sharpes"]["std"]
        p_low  = [k for k in ci["sharpes"] if k.startswith("p") and int(k[1:]) < 50][0]
        p_high = [k for k in ci["sharpes"] if k.startswith("p") and int(k[1:]) > 50][0]
        print(f"  Sharpe IC 90%  : [{ci['sharpes'][p_low]:>6.3f}, {ci['sharpes'][p_high]:>6.3f}]")
        print(f"  Max DD IC 90%  : [{ci['max_dds'][p_low]:>6.2%}, {ci['max_dds'][p_high]:>6.2%}]")
        result.mc_sharpe_5pct  = ci["sharpes"][p_low]
        result.mc_sharpe_95pct = ci["sharpes"][p_high]
        result.mc_max_dd_mean  = ci["max_dds"]["mean"]
        result.mc_max_dd_5pct  = ci["max_dds"][p_low]

        # ── ÉTAPE 5 : Verdict final ───────────────────────────
        logger.info("\n  ETAPE 5 : Verdict final")
        logger.info("─" * 40)

        criteria_results, passed = self._check_criteria(
            oos_metrics    = oos_metrics,
            pct_positive   = pct_positive,
            mc_sharpe_5pct = result.mc_sharpe_5pct,
            is_oos_ratio   = result.sharpe_ratio,
        )

        result.criteria_results = {k: v[2] for k, v in criteria_results.items()}
        result.passed           = passed

        # ── Rapport final ──────────────────────────────────────
        self._print_final_report(result, mc_report)

        # ── Sauvegarde ─────────────────────────────────────────
        if save_results:
            self._save_results(result, yearly_metrics, mc_report, full_returns)

        return result

    # ─────────────────────────────────────────────────────────
    # MÉTHODE : Affichage du rapport final
    # ─────────────────────────────────────────────────────────
    def _print_final_report(self, result: ValidationResult, mc_report: dict):
        """Affiche le rapport de validation complet."""

        print("\n" + "="*65)
        print("  RAPPORT DE VALIDATION PHASE 2D")
        print("="*65)

        print("\n  IN-SAMPLE vs OUT-OF-SAMPLE")
        print(f"  {'Metrique':<20} {'In-Sample':>12} {'Out-of-Sample':>14} {'Ratio':>8}")
        print("  " + "-"*56)

        metrics = [
            ("CAGR",    result.is_cagr,    result.oos_cagr,  ".2%"),
            ("Sharpe",  result.is_sharpe,  result.oos_sharpe,".3f"),
            ("Max DD",  result.is_max_dd,  result.oos_max_dd,".2%"),
        ]
        for label, is_val, oos_val, fmt in metrics:
            ratio = is_val / oos_val if oos_val != 0 else float("inf")
            print(
                f"  {label:<20} "
                f"{format(is_val, fmt):>12} "
                f"{format(oos_val, fmt):>14} "
                f"{ratio:>8.2f}x"
            )

        print(f"\n  STABILITE ANNUELLE")
        print(f"  % annees positives : {result.pct_positive_years:.1%}")
        for year, sharpe in sorted(result.yearly_sharpes.items()):
            bar = "#" * int(max(0, sharpe) * 20)
            neg = "!" if sharpe < 0 else " "
            print(f"  {year} : {sharpe:>7.3f} {neg}{bar}")

        print(f"\n  MONTE CARLO ({mc_report.get('sharpe_percentile', 0):.0f}%ile)")
        ci = mc_report["confidence_intervals"]
        print(f"  Sharpe median  : {ci['sharpes']['median']:>8.3f}")
        print(f"  Sharpe IC 90%  : [{ci['sharpes']['p5']:>6.3f}, {ci['sharpes']['p95']:>6.3f}]")
        print(f"  Max DD median  : {ci['max_dds']['median']:>8.2%}")
        print(f"  Max DD IC 90%  : [{ci['max_dds']['p5']:>6.2%}, {ci['max_dds']['p95']:>6.2%}]")

        print(f"\n  CRITERES DE PASSAGE")
        for name, passed in result.criteria_results.items():
            icon = "PASS" if passed else "FAIL"
            print(f"  [{icon}] {name}")

        verdict_str = "GO  → Passer au Ghost Test" if result.passed else "NO-GO → Retravailler la strategie"
        print(f"\n  VERDICT : {verdict_str}")
        print("="*65)

    # ─────────────────────────────────────────────────────────
    # MÉTHODE : Sauvegarde des résultats
    # ─────────────────────────────────────────────────────────
    def _save_results(
        self,
        result        : ValidationResult,
        yearly_metrics: dict,
        mc_report     : dict,
        full_returns  : pd.Series,
    ):
        """Sauvegarde les résultats de validation."""
        os.makedirs("./results/validation", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Métriques principales
        summary = {
            "is_cagr"          : result.is_cagr,
            "is_sharpe"        : result.is_sharpe,
            "is_max_dd"        : result.is_max_dd,
            "oos_cagr"         : result.oos_cagr,
            "oos_sharpe"       : result.oos_sharpe,
            "oos_max_dd"       : result.oos_max_dd,
            "sharpe_ratio"     : result.sharpe_ratio,
            "pct_positive_years": result.pct_positive_years,
            "mc_sharpe_mean"   : result.mc_sharpe_mean,
            "mc_sharpe_5pct"   : result.mc_sharpe_5pct,
            "mc_sharpe_95pct"  : result.mc_sharpe_95pct,
            "mc_max_dd_mean"   : result.mc_max_dd_mean,
            "passed"           : result.passed,
        }
        pd.Series(summary).to_csv(f"./results/validation/summary_{ts}.csv", header=["value"])

        # Rendements journaliers
        full_returns.to_csv(f"./results/validation/returns_{ts}.csv", header=["return"])

        # Distribution Monte Carlo
        if "simulations" in mc_report:
            mc_df = pd.DataFrame(mc_report["simulations"])
            mc_df.to_csv(f"./results/validation/monte_carlo_{ts}.csv", index=False)

        logger.info(f"\n  Resultats sauvegardes dans ./results/validation/")


# ============================================================
# SCRIPT PRINCIPAL
# ============================================================

if __name__ == "__main__":

    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("  VALIDATION PHASE 2D — Walk-Forward + Monte Carlo")
    print("=" * 60)

    # Chargement des données
    cache_file = "./data/processed/price_matrix.csv"
    if not os.path.exists(cache_file):
        print("ERREUR : Cache non trouve. Lance run_backtest.py d'abord.")
        sys.exit(1)

    print(f"\nChargement des donnees...")
    price_matrix = pd.read_csv(cache_file, index_col="date", parse_dates=True)
    print(f"Donnees : {price_matrix.shape}")

    # Asset types
    from config import FUTURES_UNIVERSE
    asset_types = {
        s: "future" if s in FUTURES_UNIVERSE else "stock"
        for s in price_matrix.columns
    }

    # Validation complète
    validator = StrategyValidator(
        price_matrix    = price_matrix,
        asset_types     = asset_types,
        initial_capital = 100_000,
    )

    result = validator.validate(
        is_end_date    = "2021-12-31",
        oos_start_date = "2022-01-01",
        oos_end_date   = "2024-12-31",
        n_mc_sims      = 1000,
        save_results   = True,
    )

    print(f"\nValidation terminee !")
    print(f"Verdict : {'GO GHOST TEST' if result.passed else 'NO-GO - Retravailler'}")