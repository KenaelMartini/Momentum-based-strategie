from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .risk_types import RegimeFeatures


class RegimeFeatureCalculator:
    def __init__(
        self,
        vix_series: Optional[pd.Series] = None,
        min_history: int = 200,
        min_assets: int = 5,
    ):
        self.vix_series = vix_series.sort_index() if vix_series is not None else None
        self.min_history = min_history
        self.min_assets = min_assets

    def compute(self, prices: pd.DataFrame) -> Optional[RegimeFeatures]:
        if prices is None:
            return None

        px = prices.dropna(axis=1, how="all").copy()
        if len(px) < self.min_history or px.shape[1] < self.min_assets:
            return None

        rets = np.log(px / px.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        last_date = px.index[-1]
        market_ret = rets.mean(axis=1)
        market_index = (1.0 + market_ret).cumprod()

        trend_score = self._trend_score(px)
        vol_score, vol_ratio = self._vol_score(market_ret)
        corr_score, avg_corr = self._corr_score(rets)
        breadth_score, breadth = self._breadth_score(px)
        dispersion_score, dispersion = self._dispersion_score(rets)

        vix_score = 0.5
        vix_level = float("nan")
        vix_5d_change = float("nan")
        if self.vix_series is not None:
            vix_score, vix_level, vix_5d_change = self._vix_score(last_date)

        market_63d_return = 0.0
        if len(market_index) >= 64:
            market_63d_return = float(market_index.iloc[-1] / market_index.iloc[-64] - 1.0)

        return RegimeFeatures(
            trend_score=float(trend_score),
            vol_score=float(vol_score),
            corr_score=float(corr_score),
            breadth_score=float(breadth_score),
            dispersion_score=float(dispersion_score),
            vix_score=float(vix_score),
            vol_ratio_20_63=float(vol_ratio),
            avg_corr_42d=float(avg_corr),
            breadth_200d=float(breadth),
            dispersion_20d=float(dispersion),
            vix_level=float(vix_level),
            vix_5d_change=float(vix_5d_change),
            market_63d_return=float(market_63d_return),
        )

    def _trend_score(self, px: pd.DataFrame) -> float:
        ma_50 = px.iloc[-50:].mean()
        ma_200 = px.iloc[-200:].mean()
        spot = px.iloc[-1]

        above_200 = (spot > ma_200).mean()
        above_50 = (spot > ma_50).mean()
        score = 0.65 * above_200 + 0.35 * above_50
        return float(np.clip(score, 0.0, 1.0))

    def _vol_score(self, market_ret: pd.Series) -> tuple[float, float]:
        vol_20 = market_ret.iloc[-20:].std()
        vol_63 = market_ret.iloc[-63:].std()
        ratio = float(vol_20 / (vol_63 + 1e-12))

        if ratio <= 0.90:
            score = 1.00
        elif ratio <= 1.10:
            score = 0.85
        elif ratio <= 1.25:
            score = 0.65
        elif ratio <= 1.50:
            score = 0.40
        elif ratio <= 1.80:
            score = 0.20
        else:
            score = 0.05

        return float(score), ratio

    def _corr_score(self, rets: pd.DataFrame) -> tuple[float, float]:
        corr = rets.iloc[-42:].corr()
        n_assets = len(corr)
        if n_assets <= 1:
            return 0.5, 0.0

        avg_corr = (corr.to_numpy().sum() - n_assets) / (n_assets * (n_assets - 1) + 1e-12)

        if avg_corr <= 0.20:
            score = 1.00
        elif avg_corr <= 0.30:
            score = 0.80
        elif avg_corr <= 0.40:
            score = 0.55
        elif avg_corr <= 0.50:
            score = 0.30
        else:
            score = 0.10

        return float(score), float(avg_corr)

    def _breadth_score(self, px: pd.DataFrame) -> tuple[float, float]:
        ma_200 = px.iloc[-200:].mean()
        breadth = (px.iloc[-1] > ma_200).mean()

        if breadth >= 0.70:
            score = 1.00
        elif breadth >= 0.55:
            score = 0.80
        elif breadth >= 0.45:
            score = 0.55
        elif breadth >= 0.35:
            score = 0.30
        else:
            score = 0.10

        return float(score), float(breadth)

    def _dispersion_score(self, rets: pd.DataFrame) -> tuple[float, float]:
        dispersion = float(rets.iloc[-20:].std(axis=1).mean())

        if dispersion >= 0.030:
            score = 1.00
        elif dispersion >= 0.022:
            score = 0.80
        elif dispersion >= 0.015:
            score = 0.55
        elif dispersion >= 0.010:
            score = 0.30
        else:
            score = 0.10

        return float(score), dispersion

    def _vix_score(self, date: pd.Timestamp) -> tuple[float, float, float]:
        vix = self.vix_series[self.vix_series.index <= date]
        if len(vix) < 6:
            return 0.5, float("nan"), float("nan")

        level = float(vix.iloc[-1])
        change_5d = float(vix.iloc[-1] / vix.iloc[-6] - 1.0)

        if level < 16 and change_5d < 0.05:
            score = 1.00
        elif level < 20 and change_5d < 0.10:
            score = 0.80
        elif level < 25 and change_5d < 0.20:
            score = 0.55
        elif level < 32:
            score = 0.30
        else:
            score = 0.10

        return float(score), level, change_5d
