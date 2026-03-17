# ============================================================
# optimizer.py — Optimisation Walk-Forward du Signal Momentum
# ============================================================
import os
import sys
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    INITIAL_CAPITAL,
    TRANSACTION_COST_BPS,
    SLIPPAGE_BPS,
    TARGET_VOLATILITY,
    MAX_LEVERAGE,
    MAX_POSITION_SIZE,
    RISK_FREE_RATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================
# DATACLASS — Paramètres du signal
# ============================================================

@dataclass
class SignalParams:
    """Paramètres complets du signal momentum à optimiser."""

    momentum_windows : list = field(default_factory=lambda: [21, 63, 126, 252])
    momentum_weights : dict = field(default_factory=lambda: {
        21: 0.10, 63: 0.20, 126: 0.30, 252: 0.40
    })
    skip_days        : int   = 21
    cs_weight        : float = 0.5
    ts_weight        : float = 0.5
    long_quantile    : float = 0.80
    short_quantile   : float = 0.20

    def __post_init__(self):
        assert abs(self.cs_weight + self.ts_weight - 1.0) < 1e-6
        assert abs(sum(self.momentum_weights.values()) - 1.0) < 1e-6
        assert self.long_quantile > 0.5

    def to_dict(self) -> dict:
        return {
            "windows"      : str(self.momentum_windows),
            "skip_days"    : self.skip_days,
            "cs_weight"    : self.cs_weight,
            "long_quantile": self.long_quantile,
        }


# ============================================================
# DATACLASS — Résultat d'une évaluation
# ============================================================

@dataclass
class EvaluationResult:
    """Résultat d'un backtest avec un jeu de paramètres."""
    params       : SignalParams
    sharpe       : float = 0.0
    sortino      : float = 0.0
    calmar       : float = 0.0
    cagr         : float = 0.0
    max_drawdown : float = 0.0
    volatility   : float = 0.0
    score        : float = 0.0
    train_start  : str   = ""
    train_end    : str   = ""
    test_start   : str   = ""
    test_end     : str   = ""
    is_test      : bool  = False


# ============================================================
# CLASSE 1 : UniverseFilter
# ============================================================

class UniverseFilter:
    """Filtre l'univers d'actifs pour améliorer la qualité du signal."""

    def __init__(self, price_matrix: pd.DataFrame):
        self.prices = price_matrix.copy()

    def filter_by_data_quality(
        self,
        max_missing_pct  : float = 0.05,
        min_history_days : int   = 252
    ) -> list:
        """Filtre les actifs avec trop de données manquantes."""
        valid_symbols = []
        for symbol in self.prices.columns:
            series = self.prices[symbol].dropna()
            if len(series) < min_history_days:
                continue
            missing_pct = self.prices[symbol].isna().sum() / len(self.prices)
            if missing_pct > max_missing_pct:
                continue
            valid_symbols.append(symbol)
        logger.info(f"  Filtre qualite : {len(valid_symbols)}/{len(self.prices.columns)} actifs")
        return valid_symbols

    def filter_by_volatility(
        self,
        min_annual_vol : float = 0.05,
        max_annual_vol : float = 0.80,
        window         : int   = 252
    ) -> list:
        """Filtre les actifs trop peu ou trop volatils."""
        log_returns   = np.log(self.prices / self.prices.shift(1))
        valid_symbols = []
        for symbol in self.prices.columns:
            recent = log_returns[symbol].dropna().tail(window)
            if len(recent) < 60:
                continue
            vol = recent.std() * np.sqrt(252)
            if min_annual_vol <= vol <= max_annual_vol:
                valid_symbols.append(symbol)
        logger.info(f"  Filtre volatilite [{min_annual_vol:.0%}, {max_annual_vol:.0%}] : {len(valid_symbols)}/{len(self.prices.columns)} actifs")
        return valid_symbols

    def filter_by_correlation(
        self,
        max_correlation : float = 0.85,
        window          : int   = 252
    ) -> list:
        """Retire les actifs trop correles."""
        log_returns     = np.log(self.prices / self.prices.shift(1))
        corr_matrix     = log_returns.tail(window).dropna(how="all").corr()
        symbols_to_remove = set()

        for i, sym_i in enumerate(self.prices.columns):
            if sym_i in symbols_to_remove:
                continue
            for j, sym_j in enumerate(self.prices.columns):
                if i >= j or sym_j in symbols_to_remove:
                    continue
                if sym_i not in corr_matrix.index or sym_j not in corr_matrix.columns:
                    continue
                if corr_matrix.loc[sym_i, sym_j] > max_correlation:
                    hist_i = self.prices[sym_i].dropna().__len__()
                    hist_j = self.prices[sym_j].dropna().__len__()
                    symbols_to_remove.add(sym_j if hist_i >= hist_j else sym_i)

        valid_symbols = [s for s in self.prices.columns if s not in symbols_to_remove]
        logger.info(f"  Filtre correlation (max {max_correlation:.0%}) : {len(valid_symbols)}/{len(self.prices.columns)} actifs")
        return valid_symbols

    def apply_all_filters(
        self,
        max_missing_pct : float = 0.05,
        min_annual_vol  : float = 0.05,
        max_annual_vol  : float = 0.80,
        max_correlation : float = 0.85,
    ) -> list:
        """Applique tous les filtres sequentiellement."""
        logger.info("\n  Filtrage de l'univers d'actifs...")
        quality_symbols = self.filter_by_data_quality(max_missing_pct)
        self.prices     = self.prices[quality_symbols]
        vol_symbols     = self.filter_by_volatility(min_annual_vol, max_annual_vol)
        self.prices     = self.prices[vol_symbols]
        final_symbols   = self.filter_by_correlation(max_correlation)
        logger.info(f"\n  Univers final : {len(final_symbols)} actifs")
        return final_symbols


# ============================================================
# CLASSE 2 : WalkForwardOptimizer
# ============================================================

class WalkForwardOptimizer:
    """Optimiseur walk-forward pour la strategie momentum."""

    def __init__(
        self,
        price_matrix    : pd.DataFrame,
        asset_types     : dict  = None,
        initial_capital : float = INITIAL_CAPITAL,
    ):
        self.prices          = price_matrix.copy()
        self.asset_types     = asset_types or {}
        self.initial_capital = initial_capital
        self.best_params     = None

        logger.info(
            f"WalkForwardOptimizer initialise | "
            f"{len(self.prices)} jours x {self.prices.shape[1]} actifs"
        )

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 1 : Génération des splits
    # ─────────────────────────────────────────────────────────
    def generate_splits(
        self,
        train_years : int = 4,
        test_years  : int = 2,   # CORRECTION : 2 ans au lieu de 1
        step_years  : int = 1,
    ) -> list:
        """
        Génère les fenêtres train/test pour le walk-forward.

        CORRECTION IMPORTANTE :
        Les périodes TEST doivent être d'au moins 2 ans (504 jours)
        car le signal momentum a besoin de 252 jours de warmup.
        Avec 1 an de test, après le warmup il reste ~0 jours valides.
        Avec 2 ans de test, après le warmup il reste ~252 jours valides.
        """
        splits     = []
        train_days = int(train_years * 252)
        test_days  = int(test_years  * 252)
        step_days  = int(step_years  * 252)

        cursor = 0
        while True:
            train_end_idx  = cursor + train_days
            test_start_idx = train_end_idx
            test_end_idx   = test_start_idx + test_days

            if test_end_idx > len(self.prices):
                break

            split = {
                "train_start": self.prices.index[cursor].strftime("%Y-%m-%d"),
                "train_end"  : self.prices.index[train_end_idx - 1].strftime("%Y-%m-%d"),
                "test_start" : self.prices.index[test_start_idx].strftime("%Y-%m-%d"),
                "test_end"   : self.prices.index[min(test_end_idx-1, len(self.prices)-1)].strftime("%Y-%m-%d"),
            }
            splits.append(split)
            cursor += step_days

        logger.info(
            f"  {len(splits)} splits walk-forward | "
            f"Train: {train_years}Y | Test: {test_years}Y | Step: {step_years}Y"
        )
        for i, s in enumerate(splits):
            logger.info(
                f"    Split {i+1} : Train [{s['train_start']} -> {s['train_end']}] "
                f"| Test [{s['test_start']} -> {s['test_end']}]"
            )
        return splits

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 2 : Évaluation d'un jeu de paramètres
    # ─────────────────────────────────────────────────────────
    def _evaluate_params(
        self,
        params     : SignalParams,
        start_date : str,
        end_date   : str,
    ) -> EvaluationResult:
        """
        Évalue un jeu de paramètres sur une période donnée.

        CORRECTION IMPORTANTE :
        On passe TOUTES les données disponibles au MomentumSignalGenerator
        (pas seulement la période de test) pour qu'il ait assez de
        données pour le warmup du signal (252 jours).
        Ensuite on filtre les résultats sur la période souhaitée.
        """
        try:
            from strategies.momentum.momentum_signal import MomentumSignalGenerator
            from strategies.momentum.backtest_vectorized import VectorizedBacktest

            # ── CORRECTION CLÉ ────────────────────────────────
            # On utilise TOUTES les données jusqu'à end_date pour le signal
            # Le signal generator a besoin de l'historique complet pour
            # calculer les fenêtres momentum (252 jours minimum)
            prices_full = self.prices.loc[:end_date]

            # Vérification minimum de données
            min_required = max(params.momentum_windows) + params.skip_days + 50
            if len(prices_full) < min_required:
                return EvaluationResult(params=params)

            # Patch temporaire du config
            import config as cfg
            orig = {
                "windows": cfg.MOMENTUM_WINDOWS,
                "weights": cfg.MOMENTUM_WEIGHTS,
                "skip"   : cfg.SKIP_DAYS,
                "lq"     : cfg.LONG_QUANTILE,
                "sq"     : cfg.SHORT_QUANTILE,
            }

            cfg.MOMENTUM_WINDOWS = params.momentum_windows
            cfg.MOMENTUM_WEIGHTS = params.momentum_weights
            cfg.SKIP_DAYS        = params.skip_days
            cfg.LONG_QUANTILE    = params.long_quantile
            cfg.SHORT_QUANTILE   = 1 - params.long_quantile

            try:
                logging.disable(logging.CRITICAL)

                # Signal sur toutes les données disponibles
                generator   = MomentumSignalGenerator(prices_full)
                sig_results = generator.run_full_pipeline(
                    cs_weight=params.cs_weight,
                    ts_weight=params.ts_weight
                )

                # Backtest sur la période souhaitée SEULEMENT
                bt = VectorizedBacktest(
                    price_matrix    = prices_full,
                    initial_capital = self.initial_capital,
                    start_date      = start_date,
                    end_date        = end_date,
                    asset_types     = self.asset_types,
                )
                bt_results = bt.run(
                    signal_results     = sig_results,
                    apply_risk_scaling = True,
                )

                logging.disable(logging.NOTSET)

            finally:
                cfg.MOMENTUM_WINDOWS = orig["windows"]
                cfg.MOMENTUM_WEIGHTS = orig["weights"]
                cfg.SKIP_DAYS        = orig["skip"]
                cfg.LONG_QUANTILE    = orig["lq"]
                cfg.SHORT_QUANTILE   = orig["sq"]

            # Calcul des métriques
            returns = bt_results["returns"].dropna()

            # Filtre sur la période exacte start_date → end_date
            returns = returns.loc[start_date:end_date]

            if len(returns) < 50:
                return EvaluationResult(params=params)

            n_years     = len(returns) / 252
            port_values = self.initial_capital * (1 + (np.exp(returns) - 1)).cumprod()
            cagr        = (port_values.iloc[-1] / self.initial_capital) ** (1/n_years) - 1
            annual_vol  = returns.std() * np.sqrt(252)

            peak     = port_values.expanding().max()
            drawdown = (port_values - peak) / peak
            max_dd   = drawdown.min()

            rf_daily    = (1 + RISK_FREE_RATE) ** (1/252) - 1
            excess_ret  = returns.mean() - rf_daily
            sharpe      = excess_ret / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

            neg_returns = returns[returns < rf_daily]
            sortino     = (
                excess_ret / neg_returns.std() * np.sqrt(252)
                if len(neg_returns) > 10 and neg_returns.std() > 0 else 0
            )

            calmar = cagr / abs(max_dd) if abs(max_dd) > 0.001 else 0

            score = (
                0.40 * max(float(sharpe), -2) +
                0.30 * max(float(calmar), -2) +
                0.30 * max(float(sortino), -2)
            )

            return EvaluationResult(
                params       = params,
                sharpe       = float(sharpe),
                sortino      = float(sortino),
                calmar       = float(calmar),
                cagr         = float(cagr),
                max_drawdown = float(max_dd),
                volatility   = float(annual_vol),
                score        = float(score),
            )

        except Exception as e:
            logging.disable(logging.NOTSET)
            logger.debug(f"  Erreur _evaluate_params : {type(e).__name__}: {e}")
            return EvaluationResult(params=params)

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 3 : Score composite
    # ─────────────────────────────────────────────────────────
    def _score_composite(
        self,
        score_sharpe  : float,
        score_calmar  : float,
        score_sortino : float,
        consistency   : float = 1.0,
    ) -> float:
        """Calcule le score composite pondéré avec penalite de consistance."""
        base_score          = 0.40 * score_sharpe + 0.30 * score_calmar + 0.30 * score_sortino
        consistency_penalty = 1.0 - 0.6 * (1.0 - consistency)
        return base_score * consistency_penalty

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 4 : Grid Search
    # ─────────────────────────────────────────────────────────
    def grid_search(
        self,
        splits     : list,
        param_grid : dict = None,
    ) -> pd.DataFrame:
        """Teste toutes les combinaisons de paramètres."""
        if param_grid is None:
            param_grid = {
                "window_configs": [
                    {"windows": [21, 63, 126, 252], "weights": {21: 0.10, 63: 0.20, 126: 0.30, 252: 0.40}},
                    {"windows": [21, 63, 126, 252], "weights": {21: 0.30, 63: 0.30, 126: 0.25, 252: 0.15}},
                    {"windows": [21, 63, 126, 252], "weights": {21: 0.05, 63: 0.15, 126: 0.30, 252: 0.50}},
                    {"windows": [63, 126, 252],     "weights": {63: 0.30, 126: 0.35, 252: 0.35}},
                    {"windows": [21, 63, 126, 252], "weights": {21: 0.25, 63: 0.25, 126: 0.25, 252: 0.25}},
                ],
                "cs_weights"    : [0.0, 0.25, 0.50, 0.75, 1.0],
                "skip_days"     : [0, 10, 21],
                "long_quantiles": [0.70, 0.80, 0.90],
            }

        n_combos = (
            len(param_grid["window_configs"]) *
            len(param_grid["cs_weights"]) *
            len(param_grid["skip_days"]) *
            len(param_grid["long_quantiles"])
        )
        logger.info(f"\n  Grid Search | {n_combos} combinaisons x {len(splits)} splits")

        results_by_params = {}
        total_evals = n_combos * len(splits)
        eval_count  = 0

        for wc in param_grid["window_configs"]:
            for cs_w in param_grid["cs_weights"]:
                for skip in param_grid["skip_days"]:
                    for lq in param_grid["long_quantiles"]:

                        params    = SignalParams(
                            momentum_windows=wc["windows"], momentum_weights=wc["weights"],
                            skip_days=skip, cs_weight=cs_w, ts_weight=1.0-cs_w,
                            long_quantile=lq, short_quantile=1-lq,
                        )
                        param_key = f"w{wc['windows']}_cs{cs_w}_sk{skip}_lq{lq}"

                        if param_key not in results_by_params:
                            results_by_params[param_key] = {"params": params, "results": []}

                        for split in splits:
                            eval_count += 1
                            if eval_count % 50 == 0:
                                logger.info(f"  Progression : {eval_count}/{total_evals} ({eval_count/total_evals:.0%})")

                            r_train = self._evaluate_params(params, split["train_start"], split["train_end"])
                            r_train.is_test = False

                            r_test  = self._evaluate_params(params, split["test_start"], split["test_end"])
                            r_test.is_test  = True

                            results_by_params[param_key]["results"].append({"train": r_train, "test": r_test})

        # Calcul des scores agrégés
        summary_rows = []
        for param_key, data in results_by_params.items():
            params  = data["params"]
            results = data["results"]
            if not results:
                continue

            test_sharpes  = [r["test"].sharpe  for r in results if r["test"].sharpe  != 0]
            test_calmars  = [r["test"].calmar  for r in results if r["test"].calmar  != 0]
            test_sortinos = [r["test"].sortino for r in results if r["test"].sortino != 0]
            test_cagrs    = [r["test"].cagr    for r in results]

            if not test_sharpes:
                continue

            avg_sharpe   = np.mean(test_sharpes)
            avg_calmar   = np.mean(test_calmars)  if test_calmars  else 0
            avg_sortino  = np.mean(test_sortinos) if test_sortinos else 0
            avg_cagr     = np.mean(test_cagrs)
            consistency  = sum(1 for s in test_sharpes if s > 0) / len(test_sharpes)
            final_score  = self._score_composite(avg_sharpe, avg_calmar, avg_sortino, consistency)

            train_sharpes    = [r["train"].sharpe for r in results if r["train"].sharpe != 0]
            avg_train_sharpe = np.mean(train_sharpes) if train_sharpes else 0
            tt_ratio         = avg_train_sharpe / avg_sharpe if avg_sharpe != 0 else float("inf")

            summary_rows.append({
                "params"          : params,
                "windows"         : str(params.momentum_windows),
                "cs_weight"       : params.cs_weight,
                "skip_days"       : params.skip_days,
                "long_quantile"   : params.long_quantile,
                "test_sharpe_avg" : round(avg_sharpe, 4),
                "test_calmar_avg" : round(avg_calmar, 4),
                "test_sortino_avg": round(avg_sortino, 4),
                "test_cagr_avg"   : round(avg_cagr, 4),
                "train_sharpe_avg": round(avg_train_sharpe, 4),
                "consistency"     : round(consistency, 4),
                "train_test_ratio": round(tt_ratio, 4),
                "final_score"     : round(final_score, 4),
            })

        if not summary_rows:
            logger.warning("  Aucun resultat valide dans le grid search")
            return pd.DataFrame()

        params_list = [r.pop("params") for r in summary_rows]
        results_df  = pd.DataFrame(summary_rows)
        results_df["params"] = params_list
        results_df  = results_df.sort_values("final_score", ascending=False).reset_index(drop=True)

        logger.info(f"\n  Grid Search termine | {len(results_df)} combinaisons evaluees")
        return results_df

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 5 : Optimisation Bayésienne
    # ─────────────────────────────────────────────────────────
    def bayesian_search(
        self,
        splits   : list,
        n_trials : int = 100,
    ) -> pd.DataFrame:
        """Optimisation Bayésienne avec Optuna."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("  Optuna non installe. pip install optuna")
            return pd.DataFrame()

        logger.info(f"\n  Optimisation Bayesienne | {n_trials} essais...")

        results_list = []

        WINDOW_CONFIGS = {
            "standard"      : {"windows": [21, 63, 126, 252], "weights": {21: 0.10, 63: 0.20, 126: 0.30, 252: 0.40}},
            "short_emphasis": {"windows": [21, 63, 126, 252], "weights": {21: 0.30, 63: 0.30, 126: 0.25, 252: 0.15}},
            "long_emphasis" : {"windows": [21, 63, 126, 252], "weights": {21: 0.05, 63: 0.15, 126: 0.30, 252: 0.50}},
            "no_short"      : {"windows": [63, 126, 252],     "weights": {63: 0.30, 126: 0.35, 252: 0.35}},
            "equal"         : {"windows": [21, 63, 126, 252], "weights": {21: 0.25, 63: 0.25, 126: 0.25, 252: 0.25}},
        }

        def objective(trial):
            wc_name       = trial.suggest_categorical("window_config",  list(WINDOW_CONFIGS.keys()))
            cs_weight     = trial.suggest_categorical("cs_weight",      [0.0, 0.25, 0.5, 0.75, 1.0])
            skip_days     = trial.suggest_categorical("skip_days",      [0, 10, 21])
            long_quantile = trial.suggest_categorical("long_quantile",  [0.70, 0.80, 0.90])

            wc     = WINDOW_CONFIGS[wc_name]
            params = SignalParams(
                momentum_windows = wc["windows"],
                momentum_weights = wc["weights"],
                skip_days        = skip_days,
                cs_weight        = cs_weight,
                ts_weight        = 1.0 - cs_weight,
                long_quantile    = long_quantile,
                short_quantile   = 1 - long_quantile,
            )

            test_scores = []
            for split in splits:
                result = self._evaluate_params(params, split["test_start"], split["test_end"])
                if result.score != 0:
                    test_scores.append(result.score)

            if not test_scores:
                return -999.0

            avg_score   = float(np.mean(test_scores))
            consistency = sum(1 for s in test_scores if s > 0) / len(test_scores)
            final_score = avg_score * (1 - 0.6 * (1 - consistency))

            results_list.append({
                "params"       : params,
                "windows"      : str(wc["windows"]),
                "cs_weight"    : cs_weight,
                "skip_days"    : skip_days,
                "long_quantile": long_quantile,
                "final_score"  : final_score,
                "consistency"  : consistency,
            })

            logger.info(f"  Trial {trial.number:3d} | score: {final_score:.4f} | {wc_name} | cs={cs_weight} | skip={skip_days} | lq={long_quantile}")
            return final_score

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        if not results_list:
            logger.warning("  Aucun resultat Bayesien valide")
            return pd.DataFrame()

        params_objects = [row["params"] for row in results_list]
        scalar_rows    = [{k: v for k, v in row.items() if k != "params"} for row in results_list]
        results_df     = pd.DataFrame(scalar_rows)
        results_df.insert(0, "params", params_objects)
        results_df     = results_df.sort_values("final_score", ascending=False).reset_index(drop=True)

        logger.info(f"\n  Meilleur score : {study.best_value:.4f} | Params : {study.best_params}")
        return results_df

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 6 : Extraction des meilleurs paramètres
    # ─────────────────────────────────────────────────────────
    def get_best_params(
        self,
        results_df          : pd.DataFrame,
        max_train_test_ratio: float = 2.0,
    ) -> SignalParams:
        """Extrait les meilleurs paramètres en évitant l'overfitting."""
        if results_df.empty:
            logger.warning("  Aucun resultat, retour des params par defaut")
            return SignalParams()

        if "train_test_ratio" in results_df.columns:
            filtered = results_df[results_df["train_test_ratio"] <= max_train_test_ratio]
            if filtered.empty:
                filtered = results_df
        else:
            filtered = results_df

        best_row    = filtered.iloc[0]
        best_params = best_row["params"]

        logger.info("\n  === MEILLEURS PARAMETRES ===")
        logger.info(f"  Score final   : {best_row['final_score']:.4f}")
        logger.info(f"  Fenetres      : {best_params.momentum_windows}")
        logger.info(f"  CS Weight     : {best_params.cs_weight}")
        logger.info(f"  Skip Days     : {best_params.skip_days}")
        logger.info(f"  Long Quantile : {best_params.long_quantile}")

        if "test_sharpe_avg" in best_row.index:
            logger.info(f"  Sharpe test   : {best_row['test_sharpe_avg']:.4f}")
            logger.info(f"  Consistency   : {best_row['consistency']:.1%}")

        self.best_params = best_params
        return best_params

    # ─────────────────────────────────────────────────────────
    # MÉTHODE 7 : Pipeline complète
    # ─────────────────────────────────────────────────────────
    def optimize(
        self,
        method              : str   = "bayesian",
        train_years         : int   = 4,
        test_years          : int   = 2,
        step_years          : int   = 1,
        n_trials            : int   = 100,
        max_train_test_ratio: float = 2.0,
        save_results        : bool  = True,
    ) -> SignalParams:
        """Lance le pipeline d'optimisation complet."""
        logger.info("\n" + "="*55)
        logger.info("  OPTIMISATION WALK-FORWARD")
        logger.info("="*55)

        splits = self.generate_splits(train_years, test_years, step_years)
        if not splits:
            logger.error("  Pas assez de donnees")
            return SignalParams()

        if method == "grid":
            results_df = self.grid_search(splits)
        elif method == "bayesian":
            results_df = self.bayesian_search(splits, n_trials)
        else:
            raise ValueError(f"Methode inconnue : {method}")

        if results_df.empty:
            return SignalParams()

        if save_results:
            os.makedirs("./results/optimization", exist_ok=True)
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"./results/optimization/{method}_{ts}.csv"
            results_df.drop(columns=["params"], errors="ignore").to_csv(path, index=False)
            logger.info(f"  Resultats sauvegardes : {path}")

        display_cols = [c for c in ["windows", "cs_weight", "skip_days", "long_quantile",
                                     "test_sharpe_avg", "consistency", "final_score"]
                        if c in results_df.columns]
        logger.info("\n  TOP 10 PARAMETRES (out-of-sample) :")
        print(results_df[display_cols].head(10).to_string(index=False))

        return self.get_best_params(results_df, max_train_test_ratio)


# ============================================================
# SCRIPT PRINCIPAL
# ============================================================

if __name__ == "__main__":

    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("  TEST — WalkForwardOptimizer")
    print("=" * 60)

    cache_file = "./data/processed/price_matrix.csv"

    if os.path.exists(cache_file):
        print(f"\nChargement depuis {cache_file}...")
        price_matrix = pd.read_csv(cache_file, index_col="date", parse_dates=True)
        print(f"Donnees chargees : {price_matrix.shape}")
    else:
        print("\nGeneration de donnees simulees...")
        np.random.seed(42)
        n_days = 2520
        assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "GC", "CL", "JPM", "XOM", "NVDA"]
        dates  = pd.bdate_range(start="2015-01-01", periods=n_days)
        rets   = np.random.normal(0.08/252, 0.20/np.sqrt(252), (n_days, len(assets)))
        prices = 100 * np.exp(np.cumsum(rets, axis=0))
        price_matrix = pd.DataFrame(prices, index=dates, columns=assets)

    # Filtrage
    print("\nFiltrage de l'univers...")
    uf            = UniverseFilter(price_matrix)
    valid_symbols = uf.apply_all_filters(
        max_missing_pct=0.05, min_annual_vol=0.05,
        max_annual_vol=0.80,  max_correlation=0.90,
    )
    price_matrix_filtered = price_matrix[valid_symbols]
    print(f"Univers filtre : {len(valid_symbols)} actifs")

    from config import FUTURES_UNIVERSE
    asset_types = {s: "future" if s in FUTURES_UNIVERSE else "stock" for s in valid_symbols}

    # Optimisation
    print("\nLancement de l'optimisation...")
    optimizer = WalkForwardOptimizer(
        price_matrix    = price_matrix_filtered,
        asset_types     = asset_types,
        initial_capital = 100_000,
    )

    best_params = optimizer.optimize(
        method      = "bayesian",
        train_years = 4,
        test_years  = 2,   # 2 ans pour avoir assez de données après le warmup
        step_years  = 1,
        n_trials    = 100,
    )

    print("\n" + "=" * 60)
    print("  MEILLEURS PARAMETRES TROUVES")
    print("=" * 60)
    print(f"  Fenetres   : {best_params.momentum_windows}")
    print(f"  Poids      : {best_params.momentum_weights}")
    print(f"  CS Weight  : {best_params.cs_weight}")
    print(f"  Skip Days  : {best_params.skip_days}")
    print(f"  Long Quant : {best_params.long_quantile}")
    print("=" * 60)
    print("\nMaintenant mets a jour config.py avec ces parametres !")