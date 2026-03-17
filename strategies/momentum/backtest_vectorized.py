# ============================================================
# backtest_vectorized.py — Moteur de Backtest Vectorisé
# ============================================================
import os
import sys
import logging
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    INITIAL_CAPITAL, TRANSACTION_COST_BPS, SLIPPAGE_BPS,
    BACKTEST_START, BACKTEST_END, REBALANCING_FREQUENCY,
    TARGET_VOLATILITY, MAX_LEVERAGE, MAX_POSITION_SIZE,
    LONG_QUANTILE, SHORT_QUANTILE, RISK_FREE_RATE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


class VectorizedBacktest:
    def __init__(self, price_matrix, initial_capital=INITIAL_CAPITAL, start_date=BACKTEST_START, end_date=BACKTEST_END, asset_types=None):
        self.prices = price_matrix.loc[start_date:end_date].copy()
        if self.prices.empty:
            raise ValueError(f"❌ Aucune donnée entre {start_date} et {end_date}")
        self.initial_capital = initial_capital
        self.asset_types = asset_types or {s: "stock" for s in price_matrix.columns}
        self.results = None
        logger.info(f"VectorizedBacktest initialisé | {len(self.prices)} jours | {self.prices.shape[1]} actifs | {self.prices.index[0].date()} → {self.prices.index[-1].date()}")

    def _get_rebalancing_dates(self):
        freq_map = {"daily": "B", "weekly": "W-FRI", "monthly": "ME"}
        freq = freq_map.get(REBALANCING_FREQUENCY, "ME")
        rebal_dates = self.prices.resample(freq).last().index
        rebal_dates = rebal_dates[rebal_dates.isin(self.prices.index)]
        logger.info(f"📅 {len(rebal_dates)} dates de rebalancement ({REBALANCING_FREQUENCY})")
        return rebal_dates

    def _compute_target_weights_matrix(self, signal_final, ewma_vol):
        common_idx = signal_final.index.intersection(ewma_vol.index)
        signals = signal_final.reindex(common_idx)
        vols = ewma_vol.reindex(common_idx).clip(lower=0.01)
        raw_weights = signals * (TARGET_VOLATILITY / vols)
        weights = raw_weights.clip(lower=-MAX_POSITION_SIZE, upper=MAX_POSITION_SIZE)
        gross_exp = weights.abs().sum(axis=1)
        scaling = np.minimum(1.0, MAX_LEVERAGE / gross_exp.replace(0, 1))
        weights = weights.mul(scaling, axis=0)
        logger.info(f"✅ Matrice des poids calculée : {weights.shape} | Levier moyen: {weights.abs().sum(axis=1).mean():.2f}x | Position max: {weights.abs().max().max():.3f}")
        return weights

    def _propagate_weights(self, target_weights, rebal_dates):
        all_dates = self.prices.index
        weights_all = pd.DataFrame(np.nan, index=all_dates, columns=target_weights.columns)
        rebal_in_range = rebal_dates[rebal_dates.isin(all_dates)]
        common_rebal = rebal_in_range[rebal_in_range.isin(target_weights.index)]
        weights_all.loc[common_rebal] = target_weights.reindex(common_rebal).values
        weights_all = weights_all.ffill().fillna(0)
        weights_all = weights_all.shift(1).fillna(0)
        logger.info(f"✅ Poids propagés sur {len(weights_all)} jours | Jours avec position: {(weights_all.abs().sum(axis=1) > 0).sum()}")
        return weights_all

    def _apply_dynamic_risk_scaling(self, weights_all):
        log_returns = np.log(self.prices / self.prices.shift(1)).fillna(0)
        port_returns = (weights_all * log_returns).sum(axis=1)
        realized_vol = port_returns.rolling(20).std().multiply(np.sqrt(252)).fillna(TARGET_VOLATILITY).clip(lower=0.01)
        scaling = (TARGET_VOLATILITY / realized_vol).clip(0.25, 1.50)
        scaling = scaling.shift(1).fillna(1.0)
        scaled_weights = weights_all.mul(scaling, axis=0)
        logger.info(f"✅ Risk scaling appliqué | Scaling moyen: {scaling.mean():.2f}x | min: {scaling.min():.2f}x | max: {scaling.max():.2f}x")
        return scaled_weights

    def _compute_pnl(self, weights_all, rebal_dates):
        """
        Calcule P&L, coûts et valeur du portefeuille.

        FIX BUG TURNOVER :
        Le .where(rebal_mask) échouait à cause d'un mismatch de type
        entre les index (DatetimeTZAware vs Naive).
        Solution : get_indexer() convertit les dates en positions
        entières, puis assignation par .iloc — 100% fiable.

        FORMULE RENDEMENT :
            r(t) = Σ_i [ w_i(t) × r_i(t) ]

        FORMULE COÛTS :
            Coût(t) = Σ_i |Δw_i(t)| × (TC + Slip) / 10000
            Seulement aux dates de rebalancement.
        """
        log_returns = np.log(self.prices / self.prices.shift(1)).fillna(0)
        gross_returns = (weights_all * log_returns).sum(axis=1)

        # Positions entières — évite tout mismatch de type de dates
        # Les poids changent à J+1 (lag d'1 jour)
        # On décale les rebal_dates d'1 jour business day
        rebal_dates_shifted = rebal_dates.shift(1, freq="B")
        rebal_positions = self.prices.index.get_indexer(
            rebal_dates_shifted, method="nearest"
        )
        rebal_positions = rebal_positions[rebal_positions >= 0]

        # Turnover journalier
        prev_weights = weights_all.shift(1).fillna(0)
        turnover_all = (weights_all - prev_weights).abs().sum(axis=1)

        # Assignation directe par position entière
        total_cost_fraction = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10_000
        daily_costs = pd.Series(0.0, index=self.prices.index)
        for pos in rebal_positions:
            if pos < len(turnover_all):
                daily_costs.iloc[pos] = turnover_all.iloc[pos] * total_cost_fraction

        monthly_turnover = turnover_all.iloc[rebal_positions].mean() if len(rebal_positions) > 0 else 0.0

        net_returns = gross_returns - daily_costs
        simple_net_returns = np.exp(net_returns) - 1
        portfolio_values = self.initial_capital * (1 + simple_net_returns).cumprod()
        total_cost_dollars = daily_costs.sum() * self.initial_capital

        logger.info(
            f"✅ P&L calculé | "
            f"Rendement: {(portfolio_values.iloc[-1]/self.initial_capital - 1):.1%} | "
            f"Turnover mensuel: {monthly_turnover:.1%} | "
            f"Coûts totaux: {total_cost_dollars:,.0f}$"
        )
        return {
            "returns": net_returns, "gross_returns": gross_returns,
            "costs": daily_costs, "portfolio_values": portfolio_values,
            "weights": weights_all, "monthly_turnover": monthly_turnover,
        }

    def run(self, signal_results, apply_risk_scaling=True, cs_weight=0.5, ts_weight=0.5):
        """Lance le backtest complet."""
        logger.info("\n" + "="*55)
        logger.info("  🚀 LANCEMENT DU BACKTEST VECTORISÉ")
        logger.info("="*55)

        signal_final = signal_results["signal_final"].reindex(signal_results["signal_final"].index.intersection(self.prices.index))
        ewma_vol = signal_results["ewma_vol"].reindex(signal_results["ewma_vol"].index.intersection(self.prices.index))

        if signal_final.empty:
            raise ValueError("❌ Aucun signal valide dans la période")

        logger.info(f"  Signaux disponibles : {len(signal_final)} dates")

        logger.info("  Calcul des poids cibles...")
        rebal_dates = self._get_rebalancing_dates()
        target_weights = self._compute_target_weights_matrix(signal_final, ewma_vol)

        logger.info("  Propagation des poids (lag 1j)...")
        weights_all = self._propagate_weights(target_weights, rebal_dates)

        if apply_risk_scaling:
            logger.info("  Application du risk scaling dynamique...")
            weights_all = self._apply_dynamic_risk_scaling(weights_all)

        logger.info("  Calcul du P&L et des coûts...")
        pnl = self._compute_pnl(weights_all, rebal_dates)

        ew = pd.DataFrame(1.0 / self.prices.shape[1], index=self.prices.index, columns=self.prices.columns).shift(1).fillna(0)
        log_ret_bm = np.log(self.prices / self.prices.shift(1)).fillna(0)
        benchmark_returns = (ew * log_ret_bm).sum(axis=1)
        benchmark_values = self.initial_capital * (1 + (np.exp(benchmark_returns) - 1)).cumprod()

        final_value = pnl["portfolio_values"].iloc[-1]
        total_return = final_value / self.initial_capital - 1

        self.results = {
            "returns": pnl["returns"], "gross_returns": pnl["gross_returns"],
            "costs": pnl["costs"], "portfolio_values": pnl["portfolio_values"],
            "weights": pnl["weights"], "target_weights": target_weights,
            "benchmark_returns": benchmark_returns, "benchmark_values": benchmark_values,
            "rebal_dates": rebal_dates, "monthly_turnover": pnl["monthly_turnover"],
            "initial_capital": self.initial_capital, "n_assets": self.prices.shape[1], "n_days": len(self.prices),
        }

        logger.info("\n" + "─"*55)
        logger.info(f"  ✅ BACKTEST TERMINÉ")
        logger.info(f"  Période  : {self.prices.index[0].date()} → {self.prices.index[-1].date()}")
        logger.info(f"  Capital  : {self.initial_capital:,.0f}$ → {final_value:,.0f}$")
        logger.info(f"  Rendement: {total_return:.1%}")
        logger.info("─"*55)
        return self.results


if __name__ == "__main__":
    print("=" * 60)
    print("  TEST — VectorizedBacktest")
    print("=" * 60)

    from strategies.momentum.momentum_signal import MomentumSignalGenerator

    np.random.seed(42)
    n_days = 756
    assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "GC", "CL", "JPM", "XOM", "NVDA"]
    asset_types = {"AAPL": "stock", "MSFT": "stock", "GOOGL": "stock", "AMZN": "stock", "META": "stock", "JPM": "stock", "XOM": "stock", "NVDA": "stock", "GC": "future", "CL": "future"}

    daily_returns = np.random.normal(0.08/252, 0.20/np.sqrt(252), (n_days, len(assets)))
    prices_array = 100 * np.exp(np.cumsum(daily_returns, axis=0))
    dates = pd.bdate_range(start="2021-01-01", periods=n_days)
    price_matrix = pd.DataFrame(prices_array, index=dates, columns=assets)

    print("\n📊 Calcul des signaux momentum...")
    generator = MomentumSignalGenerator(price_matrix)
    results = generator.run_full_pipeline()

    print("\n🚀 Lancement du backtest...")
    backtest = VectorizedBacktest(price_matrix=price_matrix, initial_capital=100_000, start_date="2021-01-01", end_date="2023-12-31", asset_types=asset_types)
    bt_results = backtest.run(signal_results=results, apply_risk_scaling=True)

    print("\n📈 Aperçu des rendements journaliers :")
    print(bt_results["returns"].tail(10).round(5))
    print("\n💼 Aperçu des valeurs du portefeuille :")
    print(bt_results["portfolio_values"].tail(10).round(2))
    print(f"\n✅ Backtest prêt — lance metrics.performance pour le rapport complet")