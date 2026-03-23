# ============================================================
# DataHandler — flux de prix jour par jour (pas de look-ahead)
# ============================================================
from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .events import MarketEvent

logger = logging.getLogger(__name__)


class DataHandler:
    """
    Distribue les données UNE DATE À LA FOIS.
    get_history(date, window) filtre strictement <= date
    → impossible de voir le futur par construction.
    """

    def __init__(self, data_path, start_date, end_date, min_history=252):
        self.data_path = Path(data_path)
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.min_history = min_history
        self.prices_full = None
        self.returns_full = None
        self.current_idx = 0
        self.dates = None
        self._load_data()

    def _load_data(self):
        logger.info(f"Chargement des données : {self.data_path}")
        df = pd.read_csv(self.data_path, index_col=0, parse_dates=True).sort_index()
        mask = (df.index >= self.start_date - timedelta(days=365)) & (df.index <= self.end_date)
        self.prices_full = df[mask].copy()
        self.returns_full = np.log(self.prices_full / self.prices_full.shift(1))
        self.dates = self.prices_full.index[self.prices_full.index >= self.start_date]
        self.current_idx = 0
        logger.info(f"  {len(self.prices_full)} jours chargés | {len(self.prices_full.columns)} actifs")
        logger.info(f"  Période : {self.dates[0].date()} → {self.dates[-1].date()}")

    def get_next_bar(self) -> Optional[MarketEvent]:
        if self.current_idx >= len(self.dates):
            return None
        date = self.dates[self.current_idx]
        prices = self.prices_full.loc[date]
        self.current_idx += 1
        return MarketEvent(date=date, prices=prices)

    def get_history(self, date: pd.Timestamp, window: int) -> pd.DataFrame:
        """Retourne les prix jusqu'à date — jamais au-delà."""
        history = self.prices_full[self.prices_full.index <= date]
        return history.iloc[-window:] if len(history) >= window else history

    @property
    def has_data(self) -> bool:
        return self.current_idx < len(self.dates)
