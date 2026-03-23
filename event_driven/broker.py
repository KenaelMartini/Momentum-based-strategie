# ============================================================
# SimulatedBroker — slippage, commission, exécution T+1
# ============================================================
from __future__ import annotations

import numpy as np
import pandas as pd

from config import SLIPPAGE_BPS, TRANSACTION_COST_BPS

from .events import FillEvent, OrderEvent

COMMISSION_RATE = TRANSACTION_COST_BPS / 10_000
SLIPPAGE_RATE = SLIPPAGE_BPS / 10_000


class SimulatedBroker:
    """
    Exécution réaliste :
    - Slippage : prix défavorable selon direction de l'ordre
    - Commission : % de la valeur du trade
    - Exécution à T+1 (ordres soumis à T, exécutés au bar suivant)
    """

    def __init__(self, commission_rate=COMMISSION_RATE, slippage_rate=SLIPPAGE_RATE):
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.pending_orders = []

    def submit_order(self, order: OrderEvent):
        self.pending_orders.append(order)

    def execute_pending(self, current_prices: pd.Series) -> list:
        fills = []
        for order in self.pending_orders:
            t = order.ticker
            qty = order.quantity
            if t not in current_prices.index:
                continue
            p = float(current_prices[t])
            if np.isnan(p) or p <= 0:
                continue
            direction = 1 if qty > 0 else -1
            fill_price = p * (1 + direction * self.slippage_rate)
            commission = abs(qty) * fill_price * self.commission_rate
            fills.append(
                FillEvent(
                    date=order.date,
                    ticker=t,
                    quantity=qty,
                    fill_price=fill_price,
                    commission=commission,
                )
            )
        self.pending_orders = []
        return fills
