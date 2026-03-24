import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from event_driven import EventDrivenEngine


class _FakeRiskSnapshot:
    def __init__(self):
        self.regime = types.SimpleNamespace(name="NORMAL")
        self.regime_score = 0.75
        self.positions_to_close = []
        self.trading_suspended = False
        self.risk_scaling = 1.0
        self.confidence = 0.8
        self.dd_max_stop = False
        self.suspension_reason = ""
        self.suspended_days = 0
        self.prolonged_underwater_active = False
        self.current_drawdown = 0.0


class _FakeRiskManager:
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital

    def update(
        self,
        date,
        prices,
        portfolio_value,
        current_positions,
        entry_prices,
        prev_prices,
    ):
        return _FakeRiskSnapshot()

    def apply_post_rebalance_recut_check(self, date, portfolio_value, current_positions):
        return None


class _FakeSignalGeneratorV2:
    def __init__(self, data_handler, risk_manager, rebalance_threshold=None, **kwargs):
        self.data_handler = data_handler
        self.risk_manager = risk_manager
        self.last_diagnostics = {"rebal_threshold": 0.03, "signal_reason": "test"}

    def update_ewma_vol(self, prices, prev_prices):
        return None

    def compute_weights(self, date, risk_snapshot, market_regime_state=""):
        history = self.data_handler.get_history(date, 1)
        assets = list(history.columns)
        if len(assets) < 2:
            return {}
        return {assets[0]: 0.2, assets[1]: -0.2}


def _install_fake_event_driven_risk(monkeypatch):
    fake_module = types.ModuleType("event_driven_risk")
    fake_module.EventDrivenRiskManager = _FakeRiskManager
    fake_module.MomentumSignalGeneratorV2 = _FakeSignalGeneratorV2
    monkeypatch.setitem(sys.modules, "event_driven_risk", fake_module)


def _write_price_matrix(tmp_path: Path) -> Path:
    dates = pd.bdate_range("2024-01-01", periods=45)
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0004, 0.01, size=(len(dates), 3))
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    df = pd.DataFrame(prices, index=dates, columns=["AAA", "BBB", "CCC"])
    path = tmp_path / "price_matrix.csv"
    df.to_csv(path)
    return path


def test_event_driven_engine_run_and_save_results(tmp_path, monkeypatch):
    _install_fake_event_driven_risk(monkeypatch)
    monkeypatch.chdir(tmp_path)
    data_path = _write_price_matrix(tmp_path)

    engine = EventDrivenEngine(
        data_path=str(data_path),
        start_date="2024-01-01",
        end_date="2024-03-31",
        output_dir=str(tmp_path / "out"),
    )
    metrics = engine.run()
    files = engine.save_results()

    assert metrics
    assert {"cagr", "sharpe", "max_dd", "calmar", "final_value", "n_trades", "avg_turnover"} <= set(
        metrics.keys()
    )
    assert "stats" in files
    assert Path(files["stats"]).exists()


def test_event_driven_engine_calls_per_day_callback(tmp_path, monkeypatch):
    _install_fake_event_driven_risk(monkeypatch)
    monkeypatch.chdir(tmp_path)
    data_path = _write_price_matrix(tmp_path)
    calls = []

    def _cb(stats, engine):
        calls.append((stats.date, engine.n_trades))

    engine = EventDrivenEngine(
        data_path=str(data_path),
        start_date="2024-01-01",
        end_date="2024-03-31",
        output_dir=str(tmp_path / "out"),
        per_day_callback=_cb,
    )
    engine.run()

    assert len(calls) == len(engine.portfolio.history)
    assert len(calls) > 0
