# ============================================================
# run_backtest.py — Script Maître | Backtest sur Données Réelles
# ============================================================
# RÔLE DE CE FICHIER :
# Orchestrer l'ensemble du pipeline de backtest en une seule
# commande. Connecte tous les modules dans le bon ordre :
#   1. Données IBKR (ou cache local)
#   2. Signal Momentum
#   3. Backtest Vectorisé
#   4. Analyse de Performance
#   5. Sauvegarde des résultats
#
# USAGE :
#   # Avec IBKR TWS ouvert (Paper Trading port 7497)
#   python -m run_backtest
#
#   # Depuis le cache local (sans TWS)
#   python -m run_backtest --use-cache
#
#   # Avec paramètres personnalisés
#   python -m run_backtest --start 2018-01-01 --end 2024-01-01
#
#   # IBKR : matrice = actions uniquement (sans futures)
#   python -m run_backtest --stocks-only
#
#   # Données seules (reconstruit price_matrix.csv)
#   python -m data.ibkr_data --stocks-only
#   python -m data.ibkr_data --stocks-only --no-cache
#
# PRÉREQUIS :
#   pip install ib_insync pandas numpy scipy
#   TWS ouvert sur port 7497 (Paper Trading)
# ============================================================

import os
import sys
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Ajout de la racine au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    INITIAL_CAPITAL,
    BACKTEST_START,
    BACKTEST_END,
    STOCK_UNIVERSE,
    FUTURES_UNIVERSE,
    DATA_PATH,
    PROCESSED_DATA_PATH,
)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION DU LOGGING
# ─────────────────────────────────────────────────────────────
# On crée un logger qui écrit à la fois dans le terminal ET
# dans un fichier log horodaté pour pouvoir auditer les runs.

log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)

log_filename = log_dir / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),                      # Terminal
        logging.FileHandler(log_filename, mode="w"),  # Fichier
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# CLASSE PRINCIPALE : BacktestRunner
# ============================================================

class BacktestRunner:
    """
    Orchestrateur du pipeline complet de backtest.

    Gère l'ensemble du workflow de manière robuste avec
    gestion des erreurs, logging et sauvegarde des résultats.
    """

    def __init__(
        self,
        start_date    : str   = BACKTEST_START,
        end_date      : str   = BACKTEST_END,
        initial_capital: float = INITIAL_CAPITAL,
        use_cache     : bool  = False,
        save_results  : bool  = True,
        stocks_only   : bool  = False,
    ):
        """
        Initialise le BacktestRunner.

        Args:
            start_date      : début du backtest "YYYY-MM-DD"
            end_date        : fin du backtest "YYYY-MM-DD"
            initial_capital : capital de départ en USD
            use_cache       : utiliser les données en cache local
                              (pas besoin de TWS ouvert)
            save_results    : sauvegarder les résultats en CSV
            stocks_only     : si False et téléchargement IBKR : inclure les futures ;
                              si True : uniquement STOCK_UNIVERSE dans la matrice
        """
        self.start_date     = start_date
        self.end_date       = end_date
        self.initial_capital = initial_capital
        self.use_cache      = use_cache
        self.stocks_only    = stocks_only
        self._should_save   = save_results

        # Dossier de résultats horodaté
        # Chaque run crée son propre dossier → historique complet
        self.results_dir = Path(
            f"./results/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Ces attributs seront remplis au fur et à mesure
        self.price_matrix   = None
        self.signal_results = None
        self.bt_results     = None
        self.perf_report    = None

        # Types d'actifs — important pour le calcul des positions futures
        self.asset_types = {}

        logger.info("=" * 60)
        logger.info("  🚀 BACKTEST RUNNER — Stratégie Momentum")
        logger.info("=" * 60)
        logger.info(f"  Période    : {start_date} → {end_date}")
        logger.info(f"  Capital    : {initial_capital:,.0f}$")
        logger.info(f"  Mode cache : {'OUI' if use_cache else 'NON (IBKR live)'}")
        logger.info("=" * 60)

    # ─────────────────────────────────────────────────────────
    # ÉTAPE 1 : Chargement des données
    # ─────────────────────────────────────────────────────────
    def load_data(self) -> pd.DataFrame:
        """
        Charge les données de prix historiques.

        DEUX MODES :
          1. Cache local  : lit les CSV déjà téléchargés
          2. IBKR live    : télécharge depuis TWS

        FALLBACK INTELLIGENT :
        Si le cache existe et est récent (< 24h), on l'utilise
        même si use_cache=False — évite des téléchargements
        inutiles qui consomment les pacing limits IBKR.

        Returns:
            DataFrame price_matrix (dates × actifs)
        """
        logger.info("\n📥 ÉTAPE 1 — Chargement des données")
        logger.info("─" * 40)

        # Vérifier si un cache récent existe
        cache_file = Path(PROCESSED_DATA_PATH) / "price_matrix.csv"

        if cache_file.exists():
            # Vérifier la fraîcheur du cache
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            logger.info(f"  Cache trouvé : {cache_file} ({age_hours:.1f}h)")

            if self.use_cache or age_hours < 24:
                logger.info("  ✅ Chargement depuis le cache local...")
                price_matrix = pd.read_csv(
                    cache_file,
                    index_col="date",
                    parse_dates=True
                )
                logger.info(
                    f"  Cache chargé : {price_matrix.shape[0]} jours × "
                    f"{price_matrix.shape[1]} actifs"
                )
                self._set_asset_types(price_matrix.columns.tolist())
                self.price_matrix = price_matrix
                return price_matrix

        # Pas de cache valide → téléchargement IBKR
        logger.info("  Connexion à IBKR TWS...")

        try:
            from data.ibkr_data import IBKRDataFetcher
            fetcher = IBKRDataFetcher()

            if not fetcher.connect():
                raise ConnectionError(
                    "❌ Impossible de se connecter à TWS. "
                    "Vérifie que TWS est ouvert sur le port 7497."
                )

            try:
                # Téléchargement actions
                logger.info(f"  Téléchargement de {len(STOCK_UNIVERSE)} actions...")
                stocks_data = fetcher.fetch_stocks_data(
                    symbols=STOCK_UNIVERSE,
                    use_cache=True,
                )

                if self.stocks_only:
                    futures_data = {}
                    logger.info("  Mode stocks-only : pas de téléchargement futures.")
                    all_data = dict(stocks_data)
                else:
                    logger.info(f"  Téléchargement de {len(FUTURES_UNIVERSE)} futures...")
                    futures_data = fetcher.fetch_futures_data(
                        symbols=FUTURES_UNIVERSE,
                        use_cache=True,
                    )
                    all_data = {**stocks_data, **futures_data}

                price_matrix = fetcher.build_price_matrix(all_data)

                if price_matrix.empty:
                    raise ValueError("❌ Matrice des prix vide après téléchargement")

            finally:
                fetcher.disconnect()

        except ImportError:
            raise ImportError(
                "❌ Module ibkr_data non trouvé. "
                "Vérifie que data/ibkr_data.py existe."
            )

        self._set_asset_types(price_matrix.columns.tolist())
        self.price_matrix = price_matrix

        logger.info(
            f"  ✅ Données chargées : {price_matrix.shape[0]} jours × "
            f"{price_matrix.shape[1]} actifs | "
            f"{price_matrix.index[0].date()} → "
            f"{price_matrix.index[-1].date()}"
        )
        return price_matrix

    def _set_asset_types(self, symbols: list):
        """
        Détermine le type de chaque actif (stock ou future).
        Les futures sont ceux qui correspondent aux symboles
        de FUTURES_UNIVERSE.
        """
        for symbol in symbols:
            if symbol in FUTURES_UNIVERSE:
                self.asset_types[symbol] = "future"
            else:
                self.asset_types[symbol] = "stock"

        n_stocks  = sum(1 for v in self.asset_types.values() if v == "stock")
        n_futures = sum(1 for v in self.asset_types.values() if v == "future")
        logger.info(f"  Univers : {n_stocks} actions + {n_futures} futures")

    # ─────────────────────────────────────────────────────────
    # ÉTAPE 2 : Calcul des signaux
    # ─────────────────────────────────────────────────────────
    def compute_signals(self) -> dict:
        """
        Calcule les signaux momentum sur les données réelles.

        Lance la pipeline complète :
          Log Returns → Momentum Brut → Score Composite
          → Vol EWMA → Z-Score → Signal CS + TS → Signal Final

        Returns:
            dict des résultats de la pipeline signal
        """
        logger.info("\n📊 ÉTAPE 2 — Calcul des signaux momentum")
        logger.info("─" * 40)

        if self.price_matrix is None:
            raise ValueError("❌ Données non chargées. Lancer load_data() d'abord.")

        from strategies.momentum.momentum_signal import MomentumSignalGenerator

        # Vérification du minimum de données requises
        min_days = 252 + 21  # 1 an + skip period
        if len(self.price_matrix) < min_days:
            raise ValueError(
                f"❌ Pas assez de données : {len(self.price_matrix)} jours "
                f"(minimum requis : {min_days})"
            )

        start_time = time.time()

        generator = MomentumSignalGenerator(self.price_matrix)
        self.signal_results = generator.run_full_pipeline(
            cs_weight=0.5,
            ts_weight=0.5
        )

        elapsed = time.time() - start_time

        logger.info(
            f"  ✅ Signaux calculés en {elapsed:.2f}s | "
            f"{len(self.signal_results['signal_final'])} dates de signal"
        )

        # Statistiques rapides du signal
        sig = self.signal_results["signal_final"]
        logger.info(
            f"  Signal stats | "
            f"Long: {(sig > 0.1).mean().mean():.1%} | "
            f"Short: {(sig < -0.1).mean().mean():.1%} | "
            f"Neutre: {(sig.abs() <= 0.1).mean().mean():.1%}"
        )

        return self.signal_results

    # ─────────────────────────────────────────────────────────
    # ÉTAPE 3 : Backtest vectorisé
    # ─────────────────────────────────────────────────────────
    def run_backtest(self, apply_risk_scaling: bool = True) -> dict:
        """
        Lance le backtest vectorisé sur la période définie.

        Args:
            apply_risk_scaling : appliquer le vol targeting dynamique

        Returns:
            dict des résultats du backtest
        """
        logger.info("\n⚙️  ÉTAPE 3 — Backtest vectorisé")
        logger.info("─" * 40)

        if self.signal_results is None:
            raise ValueError("❌ Signaux non calculés. Lancer compute_signals() d'abord.")

        from strategies.momentum.backtest_vectorized import VectorizedBacktest

        start_time = time.time()

        backtest = VectorizedBacktest(
            price_matrix    = self.price_matrix,
            initial_capital = self.initial_capital,
            start_date      = self.start_date,
            end_date        = self.end_date,
            asset_types     = self.asset_types,
        )

        self.bt_results = backtest.run(
            signal_results     = self.signal_results,
            apply_risk_scaling = apply_risk_scaling,
        )

        elapsed = time.time() - start_time

        logger.info(f"  ✅ Backtest terminé en {elapsed:.2f}s")
        logger.info(
            f"  Turnover mensuel moyen : "
            f"{self.bt_results['monthly_turnover']:.1%}"
        )

        return self.bt_results

    # ─────────────────────────────────────────────────────────
    # ÉTAPE 4 : Analyse de performance
    # ─────────────────────────────────────────────────────────
    def analyze_performance(self) -> dict:
        """
        Lance l'analyse de performance complète.

        Returns:
            dict du rapport de performance
        """
        logger.info("\n📈 ÉTAPE 4 — Analyse de performance")
        logger.info("─" * 40)

        if self.bt_results is None:
            raise ValueError("❌ Backtest non lancé. Lancer run_backtest() d'abord.")

        from metrics.performance import PerformanceAnalyzer

        analyzer = PerformanceAnalyzer(
            returns           = self.bt_results["returns"],
            benchmark_returns = self.bt_results["benchmark_returns"],
            initial_capital   = self.initial_capital,
        )

        self.perf_report = analyzer.full_report()
        return self.perf_report

    # ─────────────────────────────────────────────────────────
    # ÉTAPE 5 : Sauvegarde des résultats
    # ─────────────────────────────────────────────────────────
    def save_results(self):
        """
        Sauvegarde tous les résultats du backtest.

        FICHIERS SAUVEGARDÉS :
          results/backtest_YYYYMMDD_HHMMSS/
          ├── returns.csv          ← rendements journaliers
          ├── portfolio_values.csv ← valeur du portefeuille
          ├── weights.csv          ← poids historiques
          ├── performance.csv      ← métriques de performance
          └── summary.txt          ← résumé textuel

        Ces fichiers sont utiles pour :
          - Comparer plusieurs runs avec différents paramètres
          - Créer des visualisations dans Excel/Python
          - Audit et documentation de la stratégie
        """
        if not self._should_save or self.bt_results is None:
            return

        logger.info("\n💾 ÉTAPE 5 — Sauvegarde des résultats")
        logger.info("─" * 40)

        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 1. Rendements journaliers
        returns_df = pd.DataFrame({
            "strategy_returns"  : self.bt_results["returns"],
            "gross_returns"     : self.bt_results["gross_returns"],
            "costs"             : self.bt_results["costs"],
            "benchmark_returns" : self.bt_results["benchmark_returns"],
        })
        returns_df.to_csv(self.results_dir / "returns.csv")
        logger.info("  ✅ returns.csv sauvegardé")

        # 2. Valeur du portefeuille
        values_df = pd.DataFrame({
            "portfolio_value"  : self.bt_results["portfolio_values"],
            "benchmark_value"  : self.bt_results["benchmark_values"],
        })
        values_df.to_csv(self.results_dir / "portfolio_values.csv")
        logger.info("  ✅ portfolio_values.csv sauvegardé")

        # 3. Poids historiques
        self.bt_results["weights"].to_csv(
            self.results_dir / "weights.csv"
        )
        logger.info("  ✅ weights.csv sauvegardé")

        # 4. Métriques de performance en CSV
        if self.perf_report:
            perf_flat = {}
            for section, metrics in self.perf_report.items():
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            perf_flat[f"{section}_{k}"] = v

            perf_series = pd.Series(perf_flat)
            perf_series.to_csv(self.results_dir / "performance.csv", header=["value"])
            logger.info("  ✅ performance.csv sauvegardé")

        # 5. Résumé textuel
        summary = self._build_summary()
        with open(self.results_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write(summary)
        logger.info("  ✅ summary.txt sauvegardé")

        logger.info(f"  📁 Résultats dans : {self.results_dir}")

    def _build_summary(self) -> str:
        """Construit un résumé textuel du backtest."""
        lines = [
            "=" * 60,
            "  RÉSUMÉ DU BACKTEST — STRATÉGIE MOMENTUM",
            "=" * 60,
            f"  Date du run   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Période       : {self.start_date} → {self.end_date}",
            f"  Capital initial: {self.initial_capital:,.0f}$",
            f"  Univers       : {len(self.asset_types)} actifs",
            "",
        ]

        if self.perf_report:
            r = self.perf_report.get("returns", {})
            k = self.perf_report.get("risk", {})
            q = self.perf_report.get("ratios", {})
            d = self.perf_report.get("drawdown", {})

            lines += [
                "  RENDEMENTS",
                f"    CAGR               : {r.get('cagr', 0):.2%}",
                f"    Rendement total    : {r.get('total_return', 0):.2%}",
                f"    Capital final      : {r.get('final_value', 0):,.0f}$",
                "",
                "  RISQUE",
                f"    Volatilité annuelle: {k.get('annual_volatility', 0):.2%}",
                f"    Max Drawdown       : {d.get('max_drawdown', 0):.2%}",
                f"    Sharpe Ratio       : {q.get('sharpe', 0):.3f}",
                f"    Sortino Ratio      : {q.get('sortino', 0):.3f}",
                f"    Calmar Ratio       : {q.get('calmar', 0):.3f}",
                "",
                f"  Turnover mensuel   : {self.bt_results['monthly_turnover']:.1%}",
            ]

        lines.append("=" * 60)
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────
    # MÉTHODE PRINCIPALE : run_all()
    # ─────────────────────────────────────────────────────────
    def run_all(self, apply_risk_scaling: bool = True) -> dict:
        """
        Lance le pipeline complet en une seule commande.

        PIPELINE :
          1. load_data()          → données IBKR ou cache
          2. compute_signals()    → signaux momentum
          3. run_backtest()       → simulation historique
          4. analyze_performance()→ métriques complètes
          5. save_results()       → sauvegarde CSV

        Args:
            apply_risk_scaling : appliquer le vol targeting

        Returns:
            dict avec tous les résultats
        """
        total_start = time.time()

        try:
            # Étape 1 — Données
            self.load_data()

            # Étape 2 — Signaux
            self.compute_signals()

            # Étape 3 — Backtest
            self.run_backtest(apply_risk_scaling=apply_risk_scaling)

            # Étape 4 — Performance
            self.analyze_performance()

            # Étape 5 — Sauvegarde
            if self._should_save:
                self.save_results()


        except Exception as e:
            logger.error(f"\n❌ ERREUR FATALE : {e}")
            logger.error("Pipeline interrompue.")
            raise

        total_elapsed = time.time() - total_start

        logger.info(f"\n✅ Pipeline complète en {total_elapsed:.1f}s")
        logger.info(f"📁 Log sauvegardé dans : {log_filename}")

        return {
            "price_matrix"   : self.price_matrix,
            "signal_results" : self.signal_results,
            "bt_results"     : self.bt_results,
            "perf_report"    : self.perf_report,
        }


# ============================================================
# PARSING DES ARGUMENTS EN LIGNE DE COMMANDE
# ============================================================
# argparse permet de passer des paramètres directement
# depuis le terminal sans modifier le code.
#
# Exemples :
#   python -m run_backtest
#   python -m run_backtest --use-cache
#   python -m run_backtest --start 2018-01-01 --end 2024-01-01
#   python -m run_backtest --capital 500000 --no-risk-scaling

def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Backtest de la stratégie Momentum sur données IBKR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python -m run_backtest                          # Run complet avec IBKR
  python -m run_backtest --use-cache              # Utiliser cache local
  python -m run_backtest --start 2018-01-01       # Depuis 2018
  python -m run_backtest --capital 500000         # Capital de 500k$
  python -m run_backtest --no-risk-scaling        # Sans vol targeting
  python -m run_backtest --stocks-only            # IBKR sans futures
        """
    )

    parser.add_argument(
        "--start",
        type=str,
        default=BACKTEST_START,
        help=f"Date de début du backtest (défaut: {BACKTEST_START})"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=BACKTEST_END,
        help=f"Date de fin du backtest (défaut: {BACKTEST_END})"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=INITIAL_CAPITAL,
        help=f"Capital initial en USD (défaut: {INITIAL_CAPITAL:,.0f})"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=False,
        help="Utiliser les données en cache local (pas besoin de TWS)"
    )
    parser.add_argument(
        "--no-risk-scaling",
        action="store_true",
        default=False,
        help="Désactiver le vol targeting dynamique"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Ne pas sauvegarder les résultats"
    )
    parser.add_argument(
        "--stocks-only",
        action="store_true",
        default=False,
        help="IBKR : matrice = actions uniquement (STOCK_UNIVERSE), sans futures",
    )

    return parser.parse_args()


# ============================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================

if __name__ == "__main__":

    # Parsing des arguments
    args = parse_args()

    print("\n" + "=" * 60)
    print("  🚀 BACKTEST — STRATÉGIE MOMENTUM | IBKR")
    print("=" * 60)
    print(f"  Période  : {args.start} → {args.end}")
    print(f"  Capital  : {args.capital:,.0f}$")
    print(f"  Cache    : {'OUI' if args.use_cache else 'NON (IBKR live)'}")
    print(f"  Actions  : {'uniquement' if args.stocks_only else '+ futures si IBKR'}")
    print(f"  Vol tgt  : {'NON' if args.no_risk_scaling else 'OUI'}")
    print("=" * 60 + "\n")

    # Création et lancement du runner
    runner = BacktestRunner(
        start_date     = args.start,
        end_date       = args.end,
        initial_capital = args.capital,
        use_cache      = args.use_cache,
        save_results   = not args.no_save,
        stocks_only    = args.stocks_only,
    )

    # Lancement du pipeline complet
    final_results = runner.run_all(
        apply_risk_scaling = not args.no_risk_scaling
    )

    print("\n" + "=" * 60)
    print("  ✅ BACKTEST TERMINÉ")
    print(f"  📁 Résultats dans : {runner.results_dir}")
    print(f"  📋 Log dans       : {log_filename}")
    print("=" * 60)


# Mode 1 — Télécharge les données IBKR + backtest complet
# (TWS doit être ouvert sur port 7497)
# python -m run_backtest

# Mode 2 — Utilise le cache local (TWS pas nécessaire)
# Si tu as déjà téléchargé les données une fois
# python -m run_backtest --use-cache

# Mode 3 — Paramètres personnalisés
# python -m run_backtest --start 2018-01-01 --end 2024-01-01 --capital 500000

# Mode 4 — Sans vol targeting (pour comparer)
# python -m run_backtest --use-cache --no-risk-scaling