# ============================================================
# performance.py — Métriques de Performance
# ============================================================
# RÔLE DE CE FICHIER :
# Analyser les résultats du backtest avec toutes les métriques
# utilisées en institution pour évaluer une stratégie.
#
# MÉTRIQUES IMPLÉMENTÉES :
#   Rendement    : CAGR, rendement total, rendement annuel
#   Risque       : Volatilité, Max Drawdown, VaR, CVaR
#   Ratio        : Sharpe, Sortino, Calmar, Omega
#   Drawdown     : Max DD, durée, recovery time
#   Comparaison  : Alpha, Beta, Information Ratio vs benchmark
#
# DÉPENDANCES :
#   pip install pandas numpy scipy
# ============================================================

import os
import sys
import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import RISK_FREE_RATE


class PerformanceAnalyzer:
    """
    Analyse complète des performances d'une stratégie de trading.

    Prend en entrée les résultats du backtest et produit
    un rapport de performance institutionnel complet.
    """

    def __init__(
        self,
        returns           : pd.Series,
        benchmark_returns : pd.Series = None,
        initial_capital   : float = 100_000,
        risk_free_rate    : float = RISK_FREE_RATE,
    ):
        """
        Initialise l'analyseur de performance.

        Args:
            returns           : Series des rendements journaliers nets
            benchmark_returns : Series des rendements du benchmark
            initial_capital   : capital de départ
            risk_free_rate    : taux sans risque annualisé
        """
        self.returns        = returns.dropna()
        self.benchmark      = benchmark_returns.dropna() if benchmark_returns is not None else None
        self.initial_capital = initial_capital
        self.rf_rate        = risk_free_rate

        # Taux sans risque JOURNALIER
        # (1 + rf_annuel)^(1/252) - 1 ≈ rf_annuel / 252
        self.rf_daily = (1 + risk_free_rate) ** (1/252) - 1

        # Valeur du portefeuille reconstituée depuis les rendements
        self.portfolio_values = initial_capital * (
            1 + (np.exp(self.returns) - 1)
        ).cumprod()

        logger.info(
            f"PerformanceAnalyzer initialisé | "
            f"{len(self.returns)} jours | "
            f"{self.returns.index[0].date()} → "
            f"{self.returns.index[-1].date()}"
        )

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 1 : Métriques de rendement
    # ─────────────────────────────────────────────────────────
    def compute_return_metrics(self) -> dict:
        """
        Calcule les métriques de rendement de la stratégie.

        CAGR — Compound Annual Growth Rate :
            CAGR = (V_final / V_initial)^(1/n_années) - 1
            C'est le rendement annuel moyen composé.
            C'est LA métrique de rendement standard en finance.

        RENDEMENT TOTAL :
            R_total = V_final / V_initial - 1

        RENDEMENT ANNUALISÉ (méthode alternative) :
            R_annuel = rendement_moyen_journalier × 252
            ≈ CAGR pour des rendements modérés

        Returns:
            dict des métriques de rendement
        """
        # Nombre d'années dans le backtest
        n_days  = len(self.returns)
        n_years = n_days / 252

        # Valeurs finales et initiales
        final_value   = self.portfolio_values.iloc[-1]
        initial_value = self.initial_capital

        # CAGR
        # (V_final / V_initial)^(1/n_années) - 1
        cagr = (final_value / initial_value) ** (1 / n_years) - 1

        # Rendement total
        total_return = final_value / initial_value - 1

        # Rendement annualisé depuis les returns journaliers
        # mean() × 252 = rendement annualisé arithmétique
        annual_return_arith = self.returns.mean() * 252

        # Rendement mensuel moyen
        # On regroupe par mois et on somme les rendements journaliers
        monthly_returns = self.returns.resample("ME").sum()
        avg_monthly     = monthly_returns.mean()

        # Meilleure et pire année
        annual_returns = self.returns.resample("YE").sum()
        best_year  = annual_returns.max()
        worst_year = annual_returns.min()

        # % de mois positifs
        pct_positive_months = (monthly_returns > 0).mean()

        return {
            "cagr"                  : cagr,
            "total_return"          : total_return,
            "annual_return_arith"   : annual_return_arith,
            "avg_monthly_return"    : avg_monthly,
            "best_year"             : best_year,
            "worst_year"            : worst_year,
            "pct_positive_months"   : pct_positive_months,
            "n_years"               : n_years,
            "final_value"           : final_value,
        }

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 2 : Métriques de risque
    # ─────────────────────────────────────────────────────────
    def compute_risk_metrics(self) -> dict:
        """
        Calcule les métriques de risque de la stratégie.

        VOLATILITÉ ANNUALISÉE :
            σ = std(rendements_journaliers) × √252

        VAR (Value at Risk) :
            VaR_95% = quantile 5% des rendements journaliers
            "Il y a 5% de chances de perdre plus que VaR_95% en 1 jour"

        CVAR (Conditional VaR / Expected Shortfall) :
            CVaR_95% = moyenne des rendements en dessous du VaR_95%
            "En cas de dépassement du VaR, la perte moyenne sera CVaR"
            Métrique plus conservatrice et plus informative que le VaR.
            Utilisée dans Bâle III / Solvabilité II.

        SKEWNESS :
            Mesure l'asymétrie de la distribution des rendements.
            Skewness > 0 : queue droite épaisse (bonnes surprises)
            Skewness < 0 : queue gauche épaisse (mauvaises surprises)
            Le momentum a souvent une skewness négative (momentum crash).

        KURTOSIS :
            Mesure l'épaisseur des queues de distribution.
            Kurtosis normale = 3 (distribution gaussienne)
            Kurtosis > 3 : queues épaisses → risque de pertes extrêmes
            Les marchés financiers ont toujours kurtosis > 3 ("fat tails").

        Returns:
            dict des métriques de risque
        """
        # Volatilité annualisée
        annual_vol = self.returns.std() * np.sqrt(252)

        # Downside deviation : écart-type des rendements NÉGATIFS seulement
        # Utilisé pour le calcul du Sortino Ratio
        # On ne pénalise pas la volatilité à la hausse
        negative_returns  = self.returns[self.returns < self.rf_daily]
        downside_deviation = negative_returns.std() * np.sqrt(252)

        # VaR historique (méthode non-paramétrique)
        # quantile(0.05) = valeur en dessous de laquelle se trouvent 5% des obs
        var_95 = self.returns.quantile(0.05)
        var_99 = self.returns.quantile(0.01)

        # CVaR (Expected Shortfall)
        # Moyenne de tous les rendements en dessous du VaR
        cvar_95 = self.returns[self.returns <= var_95].mean()
        cvar_99 = self.returns[self.returns <= var_99].mean()

        # Skewness et Kurtosis
        # scipy.stats donne la kurtosis excess (kurtosis - 3)
        # kurtosis excess = 0 pour une gaussienne
        skewness = stats.skew(self.returns)
        kurtosis = stats.kurtosis(self.returns)  # kurtosis excess

        # Autocorrélation des rendements (lag 1)
        # Si positive → momentum dans les rendements
        # Si négative → mean reversion dans les rendements
        autocorr = self.returns.autocorr(lag=1)

        return {
            "annual_volatility"  : annual_vol,
            "downside_deviation" : downside_deviation,
            "var_95"             : var_95,
            "var_99"             : var_99,
            "cvar_95"            : cvar_95,
            "cvar_99"            : cvar_99,
            "skewness"           : skewness,
            "kurtosis_excess"    : kurtosis,
            "autocorrelation"    : autocorr,
        }

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 3 : Analyse du drawdown
    # ─────────────────────────────────────────────────────────
    def compute_drawdown_analysis(self) -> dict:
        """
        Analyse complète des drawdowns.

        MAX DRAWDOWN (MDD) :
            MDD = min over t of (V(t) - peak(t)) / peak(t)
            C'est LA métrique de risque pour les investisseurs.
            Un MDD de -20% signifie qu'à un moment, le portefeuille
            a perdu 20% depuis son dernier sommet.

        DURÉE DU DRAWDOWN :
            Nombre de jours consécutifs en drawdown.
            Un drawdown long est psychologiquement difficile
            même si sa profondeur est modérée.

        RECOVERY TIME :
            Temps pour revenir au niveau du peak précédent
            après un drawdown.
            Un bon ratio = drawdown court avec recovery rapide.

        Returns:
            dict de l'analyse drawdown
        """
        # Série des drawdowns
        peak     = self.portfolio_values.expanding().max()
        drawdown = (self.portfolio_values - peak) / peak

        # Maximum Drawdown
        max_drawdown = drawdown.min()
        max_dd_date  = drawdown.idxmin()

        # Date du dernier peak avant le max drawdown
        peak_before_max_dd = self.portfolio_values[:max_dd_date].idxmax()

        # Durée du drawdown maximum (en jours)
        max_dd_duration = (max_dd_date - peak_before_max_dd).days

        # Recovery : quand est-ce qu'on revient au peak ?
        peak_value_at_dd = self.portfolio_values[peak_before_max_dd]
        recovery_after_max_dd = self.portfolio_values[max_dd_date:][
            self.portfolio_values[max_dd_date:] >= peak_value_at_dd
        ]

        if len(recovery_after_max_dd) > 0:
            recovery_date    = recovery_after_max_dd.index[0]
            recovery_days    = (recovery_date - max_dd_date).days
        else:
            recovery_date = None
            recovery_days = None  # Pas encore recovered à la fin du backtest

        # Drawdown actuel (à la fin du backtest)
        current_drawdown = drawdown.iloc[-1]

        # Durée totale underwater (% du temps en drawdown)
        pct_underwater = (drawdown < -0.001).mean()

        # Top 5 des drawdowns
        # On identifie les périodes distinctes de drawdown
        in_drawdown = drawdown < -0.001
        drawdown_periods = []
        start = None

        for date, val in in_drawdown.items():
            if val and start is None:
                start = date
            elif not val and start is not None:
                period_dd = drawdown[start:date].min()
                drawdown_periods.append({
                    "start"    : start,
                    "end"      : date,
                    "max_dd"   : period_dd,
                    "duration" : (date - start).days
                })
                start = None

        # Si on est encore en drawdown à la fin
        if start is not None:
            period_dd = drawdown[start:].min()
            drawdown_periods.append({
                "start"    : start,
                "end"      : drawdown.index[-1],
                "max_dd"   : period_dd,
                "duration" : (drawdown.index[-1] - start).days
            })

        # Top 5 drawdowns par profondeur
        dd_df = pd.DataFrame(drawdown_periods)
        if not dd_df.empty:
            top5_drawdowns = dd_df.nsmallest(5, "max_dd")
        else:
            top5_drawdowns = pd.DataFrame()

        return {
            "max_drawdown"         : max_drawdown,
            "max_dd_date"          : max_dd_date,
            "max_dd_duration_days" : max_dd_duration,
            "recovery_days"        : recovery_days,
            "recovery_date"        : recovery_date,
            "current_drawdown"     : current_drawdown,
            "pct_time_underwater"  : pct_underwater,
            "drawdown_series"      : drawdown,
            "top5_drawdowns"       : top5_drawdowns,
        }

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 4 : Ratios de performance
    # ─────────────────────────────────────────────────────────
    def compute_ratios(
        self,
        return_metrics   : dict,
        risk_metrics     : dict,
        drawdown_metrics : dict
    ) -> dict:
        """
        Calcule les ratios de performance clés.

        SHARPE RATIO :
            Sharpe = (CAGR - Rf) / Volatilité
            Mesure le rendement par unité de risque TOTAL.
            > 1.0 : bon | > 1.5 : très bon | > 2.0 : excellent
            Standard universel en finance.

        SORTINO RATIO :
            Sortino = (CAGR - Rf) / Downside_Deviation
            Comme Sharpe mais pénalise seulement la volatilité
            à la BAISSE (downside). Plus pertinent car les
            investisseurs ne se plaignent pas de la vol à la hausse.
            > 1.5 : bon | > 2.0 : très bon

        CALMAR RATIO :
            Calmar = CAGR / |Max_Drawdown|
            Mesure le rendement par unité de drawdown maximum.
            Très utilisé pour les stratégies CTA/momentum.
            > 0.5 : acceptable | > 1.0 : bon | > 2.0 : excellent

        OMEGA RATIO :
            Omega = E[max(r - threshold, 0)] / E[max(threshold - r, 0)]
            = Probabilité pondérée de gains / Probabilité pondérée de pertes
            > 1.0 : stratégie créatrice de valeur
            Moins sensible aux hypothèses de distribution que Sharpe.

        Returns:
            dict des ratios
        """
        cagr     = return_metrics["cagr"]
        vol      = risk_metrics["annual_volatility"]
        down_dev = risk_metrics["downside_deviation"]
        max_dd   = abs(drawdown_metrics["max_drawdown"])

        # Sharpe Ratio
        # (CAGR - Rf) / Volatilité annualisée
        sharpe = (cagr - self.rf_rate) / vol if vol > 0 else 0

        # Sortino Ratio
        # (CAGR - Rf) / Downside Deviation
        sortino = (cagr - self.rf_rate) / down_dev if down_dev > 0 else 0

        # Calmar Ratio
        # CAGR / |Max Drawdown|
        calmar = cagr / max_dd if max_dd > 0 else 0

        # Omega Ratio (seuil = taux sans risque journalier)
        threshold = self.rf_daily
        gains  = self.returns[self.returns > threshold] - threshold
        losses = threshold - self.returns[self.returns <= threshold]

        omega = gains.sum() / losses.sum() if losses.sum() > 0 else float("inf")

        # MAR Ratio (= Calmar mais sur la période totale)
        # Rendement total / |Max Drawdown|
        mar = return_metrics["total_return"] / max_dd if max_dd > 0 else 0

        return {
            "sharpe"  : sharpe,
            "sortino" : sortino,
            "calmar"  : calmar,
            "omega"   : omega,
            "mar"     : mar,
        }

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 5 : Comparaison vs Benchmark
    # ─────────────────────────────────────────────────────────
    def compute_benchmark_metrics(self) -> dict:
        """
        Calcule les métriques de comparaison vs le benchmark.

        ALPHA (Jensen's Alpha) :
            α = r_portfolio - [Rf + β × (r_benchmark - Rf)]
            Rendement excédentaire après ajustement pour le risque de marché.
            α > 0 : la stratégie génère de la valeur ajoutée
            C'est LE critère pour évaluer un gérant actif.

        BETA :
            β = Cov(r_portfolio, r_benchmark) / Var(r_benchmark)
            Sensibilité au marché.
            β = 0 : market neutral (notre cible pour long/short)
            β = 1 : bouge comme le marché
            β > 1 : amplifie les mouvements du marché

        INFORMATION RATIO :
            IR = Alpha / Tracking_Error
            Mesure la qualité du alpha généré par unité de
            risque actif pris vs le benchmark.
            > 0.5 : bon | > 1.0 : excellent
            Standard pour évaluer les gérants actifs.

        TRACKING ERROR :
            TE = std(r_portfolio - r_benchmark) × √252
            Mesure à quel point la stratégie s'écarte du benchmark.

        Returns:
            dict des métriques vs benchmark, ou dict vide si pas de benchmark
        """
        if self.benchmark is None:
            return {}

        # Alignement des index
        common = self.returns.index.intersection(self.benchmark.index)
        r_port = self.returns[common]
        r_bm   = self.benchmark[common]

        if len(r_port) < 20:
            return {}

        # Beta par régression linéaire
        # β = Cov(rp, rbm) / Var(rbm)
        covariance = np.cov(r_port, r_bm)
        beta  = covariance[0, 1] / covariance[1, 1]

        # Alpha annualisé (Jensen)
        # α = r_port_annuel - [Rf + β × (r_bm_annuel - Rf)]
        r_port_annual = r_port.mean() * 252
        r_bm_annual   = r_bm.mean() * 252
        alpha = r_port_annual - (self.rf_rate + beta * (r_bm_annual - self.rf_rate))

        # Tracking Error annualisé
        active_returns  = r_port - r_bm
        tracking_error  = active_returns.std() * np.sqrt(252)

        # Information Ratio
        active_return_annual = active_returns.mean() * 252
        info_ratio = active_return_annual / tracking_error if tracking_error > 0 else 0

        # Corrélation avec le benchmark
        correlation = r_port.corr(r_bm)

        # Up/Down capture ratio
        # Up capture   : performance relative en marché haussier
        # Down capture : performance relative en marché baissier
        up_market   = r_bm > 0
        down_market = r_bm < 0

        up_capture   = (r_port[up_market].mean() / r_bm[up_market].mean()
                        if r_bm[up_market].mean() != 0 else 0)
        down_capture = (r_port[down_market].mean() / r_bm[down_market].mean()
                        if r_bm[down_market].mean() != 0 else 0)

        return {
            "alpha"          : alpha,
            "beta"           : beta,
            "information_ratio" : info_ratio,
            "tracking_error" : tracking_error,
            "correlation_bm" : correlation,
            "up_capture"     : up_capture,
            "down_capture"   : down_capture,
        }

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 6 : Rapport complet
    # ─────────────────────────────────────────────────────────
    def full_report(self) -> dict:
        """
        Génère le rapport de performance complet.

        Calcule toutes les métriques et les présente
        dans un format standardisé institutionnel.

        Returns:
            dict complet de toutes les métriques
        """
        logger.info("📊 Calcul du rapport de performance complet...")

        returns_m  = self.compute_return_metrics()
        risk_m     = self.compute_risk_metrics()
        dd_m       = self.compute_drawdown_analysis()
        ratios_m   = self.compute_ratios(returns_m, risk_m, dd_m)
        bm_m       = self.compute_benchmark_metrics()

        report = {
            "returns"   : returns_m,
            "risk"      : risk_m,
            "drawdown"  : dd_m,
            "ratios"    : ratios_m,
            "benchmark" : bm_m,
        }

        self._print_report(report)
        return report

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 7 : Affichage du rapport
    # ─────────────────────────────────────────────────────────
    def _print_report(self, report: dict):
        """Affiche le rapport de performance formaté."""

        r = report["returns"]
        k = report["risk"]
        d = report["drawdown"]
        q = report["ratios"]
        b = report["benchmark"]

        print("\n" + "═"*60)
        print("  📊 RAPPORT DE PERFORMANCE — STRATÉGIE MOMENTUM")
        print("═"*60)

        print("\n  ── RENDEMENTS ──────────────────────────────────────")
        print(f"  CAGR                  : {r['cagr']:>10.2%}")
        print(f"  Rendement total       : {r['total_return']:>10.2%}")
        print(f"  Meilleure année       : {r['best_year']:>10.2%}")
        print(f"  Pire année            : {r['worst_year']:>10.2%}")
        print(f"  % mois positifs       : {r['pct_positive_months']:>10.1%}")
        print(f"  Capital final         : {r['final_value']:>10,.0f} $")

        print("\n  ── RISQUE ──────────────────────────────────────────")
        print(f"  Volatilité annuelle   : {k['annual_volatility']:>10.2%}")
        print(f"  Downside deviation    : {k['downside_deviation']:>10.2%}")
        print(f"  VaR 95% (1j)          : {k['var_95']:>10.2%}")
        print(f"  CVaR 95% (1j)         : {k['cvar_95']:>10.2%}")
        print(f"  Skewness              : {k['skewness']:>10.3f}")
        print(f"  Kurtosis excess       : {k['kurtosis_excess']:>10.3f}")

        print("\n  ── DRAWDOWN ────────────────────────────────────────")
        print(f"  Max Drawdown          : {d['max_drawdown']:>10.2%}")
        print(f"  Date du Max DD        : {str(d['max_dd_date'].date()):>10}")
        print(f"  Durée du Max DD       : {d['max_dd_duration_days']:>10} jours")
        recovery = d['recovery_days']
        print(f"  Recovery time         : "
              f"{'N/A (non recovered)' if recovery is None else f'{recovery:>6} jours':>10}")
        print(f"  % temps underwater    : {d['pct_time_underwater']:>10.1%}")
        print(f"  Drawdown actuel       : {d['current_drawdown']:>10.2%}")

        print("\n  ── RATIOS ──────────────────────────────────────────")
        print(f"  Sharpe Ratio          : {q['sharpe']:>10.3f}")
        print(f"  Sortino Ratio         : {q['sortino']:>10.3f}")
        print(f"  Calmar Ratio          : {q['calmar']:>10.3f}")
        print(f"  Omega Ratio           : {q['omega']:>10.3f}")

        if b:
            print("\n  ── VS BENCHMARK ────────────────────────────────────")
            print(f"  Alpha annualisé       : {b['alpha']:>10.2%}")
            print(f"  Beta                  : {b['beta']:>10.3f}")
            print(f"  Information Ratio     : {b['information_ratio']:>10.3f}")
            print(f"  Tracking Error        : {b['tracking_error']:>10.2%}")
            print(f"  Corrélation benchmark : {b['correlation_bm']:>10.3f}")
            print(f"  Up Capture            : {b['up_capture']:>10.2%}")
            print(f"  Down Capture          : {b['down_capture']:>10.2%}")

        if not report["drawdown"]["top5_drawdowns"].empty:
            print("\n  ── TOP 5 DRAWDOWNS ─────────────────────────────────")
            top5 = report["drawdown"]["top5_drawdowns"]
            for _, row in top5.iterrows():
                print(
                    f"  {str(row['start'].date())} → {str(row['end'].date())} : "
                    f"{row['max_dd']:.2%} ({row['duration']} jours)"
                )

        print("\n" + "═"*60)


# ============================================================
# SCRIPT PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  TEST COMPLET — Backtest + Performance")
    print("=" * 60)

    from strategies.momentum.backtest_vectorized import VectorizedBacktest
    from strategies.momentum.momentum_signal import MomentumSignalGenerator

    # ── Données simulées ───────────────────────────────────────
    np.random.seed(42)
    n_days  = 756
    assets  = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
               "GC",   "CL",   "JPM",   "XOM",  "NVDA"]

    asset_types = {
        "AAPL": "stock", "MSFT": "stock", "GOOGL": "stock",
        "AMZN": "stock", "META": "stock", "JPM":   "stock",
        "XOM":  "stock", "NVDA": "stock",
        "GC": "future",  "CL":  "future",
    }

    daily_returns = np.random.normal(
        0.08/252, 0.20/np.sqrt(252), (n_days, len(assets))
    )
    prices_array = 100 * np.exp(np.cumsum(daily_returns, axis=0))
    dates        = pd.bdate_range(start="2021-01-01", periods=n_days)
    price_matrix = pd.DataFrame(prices_array, index=dates, columns=assets)

    # ── Pipeline signal ────────────────────────────────────────
    print("\n📊 Calcul des signaux...")
    generator    = MomentumSignalGenerator(price_matrix)
    sig_results  = generator.run_full_pipeline()

    # ── Backtest vectorisé ─────────────────────────────────────
    print("\n🚀 Backtest en cours...")
    backtest = VectorizedBacktest(
        price_matrix    = price_matrix,
        initial_capital = 100_000,
        start_date      = "2021-01-01",
        end_date        = "2023-12-31",
        asset_types     = asset_types,
    )
    bt_results = backtest.run(sig_results, apply_risk_scaling=True)

    # ── Analyse de performance ─────────────────────────────────
    print("\n📈 Analyse de performance...")
    analyzer = PerformanceAnalyzer(
        returns           = bt_results["returns"],
        benchmark_returns = bt_results["benchmark_returns"],
        initial_capital   = 100_000,
    )

    report = analyzer.full_report()