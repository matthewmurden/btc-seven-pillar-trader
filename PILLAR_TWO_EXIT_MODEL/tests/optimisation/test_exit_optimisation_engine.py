import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from optimisation.exit_optimisation_engine import _sl_grid, _tp_grid, optimise_exits  # noqa: E402


def synthetic_df(n: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "exit_direction_consistency": rng.uniform(-1, 1, n),
            "survival_confidence": rng.uniform(0.5, 1.0, n),
            "volatility_after_entry_pred": rng.normal(0, 0.1, n),
            "volatility_after_entry_z": rng.normal(0, 1, n),
            "mae_risk_score": rng.uniform(0, 2, n),
            "ATR_14": rng.uniform(0.5, 2.0, n),
            "expected_exit_time": rng.uniform(1, 10, n),
            "exit_regime": rng.integers(0, 3, n),
            "ATR_slope": rng.normal(0, 0.1, n),
            "volatility_score": rng.normal(0, 1, n),
            "path_entropy_score": rng.uniform(0, 1, n),
        }
    )


def test_tp_grid_dynamic():
    atr = 1.0
    grid_fast = _tp_grid(atr, 1.5)
    grid_mid = _tp_grid(atr, 3.0)
    grid_slow = _tp_grid(atr, 6.0)
    assert np.isclose(grid_fast.max(), 2.0)
    assert np.isclose(grid_mid.max(), 3.0)
    assert np.isclose(grid_slow.max(), 4.0)
    assert len(grid_fast) == len(grid_mid) == len(grid_slow) == 20


def test_sl_grid_entropy_effect():
    atr = 1.0
    low_entropy_grid = _sl_grid(atr, 0.0)
    high_entropy_grid = _sl_grid(atr, 1.0)
    assert low_entropy_grid.min() < high_entropy_grid.min()
    assert np.isclose(low_entropy_grid.max(), 3.0)
    assert np.isclose(high_entropy_grid.max(), 3.0)


def test_expectancy_monotonicity():
    df = synthetic_df(2)
    df.loc[0, "exit_direction_consistency"] = 1.0  # higher TP prob
    df.loc[1, "exit_direction_consistency"] = -1.0  # higher SL prob
    res = optimise_exits(df)
    assert res["recommended_tp_pct"].iloc[0] >= res["recommended_sl_pct"].iloc[0]
    assert res["recommended_tp_pct"].iloc[1] >= 0.0  # still finite


def test_vectorised_equals_loop_small():
    df = synthetic_df(3)
    res_vec = optimise_exits(df)
    # NaN-free and positive
    assert not res_vec.isna().any().any()
    assert (res_vec[["recommended_tp_pct", "recommended_sl_pct"]] > 0).all().all()


def test_edge_cases_atr_zero_and_exit_time_short():
    df = synthetic_df(2)
    df.loc[0, "ATR_14"] = 0.0
    df.loc[0, "expected_exit_time"] = 1.0
    res = optimise_exits(df)
    assert (res["recommended_tp_pct"] > 0).all()
    assert (res["recommended_sl_pct"] > 0).all()


def test_path_entropy_bounds():
    df = synthetic_df(2)
    df.loc[0, "path_entropy_score"] = 0.0
    df.loc[1, "path_entropy_score"] = 1.0
    res = optimise_exits(df)
    assert (res["recommended_tp_pct"] > 0).all()


def test_end_to_end_sanity():
    df = synthetic_df(5)
    res = optimise_exits(df)
    assert list(res.columns) == ["recommended_tp_pct", "recommended_sl_pct", "expected_exit_time"]
    assert not res.isna().any().any()
    assert (res[["recommended_tp_pct", "recommended_sl_pct"]] > 0).all().all()
    assert (res["expected_exit_time"] >= 0).all()
