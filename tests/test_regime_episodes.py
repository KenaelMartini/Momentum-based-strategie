"""Épisodes drawdown / croissance par segment de régime."""

import pandas as pd

from risk.regime_engine import _count_contiguous_runs_strictly_longer_than, regime_streak_episode_counts


def test_count_runs_strictly_longer_than():
    s = pd.Series([True] * 11 + [False] + [True] * 5)
    assert _count_contiguous_runs_strictly_longer_than(s, 10) == 1
    # 11 True puis 5 True : un seul run > 5 (les 5 ne sont pas > 5)
    assert _count_contiguous_runs_strictly_longer_than(s, 5) == 1
    s2 = pd.Series([True] * 10)
    assert _count_contiguous_runs_strictly_longer_than(s2, 10) == 0


def test_regime_streak_episodes_resets_on_regime_change():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=15, freq="D"),
            "market_regime_state": ["A"] * 6 + ["B"] * 9,
            "drawdown": [0.0, -0.01, -0.02, -0.03, -0.04, -0.05] + [-0.01] * 9,
            "daily_return": [0.01] * 15,
        }
    )
    dd, gr = regime_streak_episode_counts(
        df,
        dd_episode_gt_days=3,
        growth_episode_gt_days=2,
    )
    # A: 5 jours <0 consécutifs → longueur 5 > 3 → 1 épisode
    assert dd.get("A", 0) == 1
    # B: 9 jours <0 → un run > 3 → 1 épisode
    assert dd.get("B", 0) == 1
    # croissance: chaque segment a daily_return>0 partout → longueur 6 et 9, toutes > 2
    assert gr.get("A", 0) == 1
    assert gr.get("B", 0) == 1
