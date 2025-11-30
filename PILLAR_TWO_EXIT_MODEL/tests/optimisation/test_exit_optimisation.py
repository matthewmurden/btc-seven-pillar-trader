import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from optimisation.exit_optimisation_engine import _build_grids, optimise_exits  # noqa: E402


def synthetic_df(n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    return pd.DataFrame(
        {
            "tp_first_proba": rng.uniform(0.1, 0.9, n),
            "sl_first_proba": rng.uniform(0.1, 0.9, n),
            "survival_confidence": rng.uniform(0.5, 1.0, n),
            "volatility_after_entry_pred": rng.normal(0, 0.1, n),
            "ATR_14": rng.uniform(0.5, 2.0, n),
            "expected_exit_time": rng.uniform(1, 200, n),
        }
    )


def test_grid_sizes():
    tp, sl = _build_grids()
    assert len(tp) == 25
    assert len(sl) == 25
    assert np.isclose(tp[0], 0.5)
    assert np.isclose(tp[-1], 4.0)
    assert np.isclose(sl[0], 0.5)
    assert np.isclose(sl[-1], 3.0)


def test_expectancy_logic_basic():
    df = pd.DataFrame(
        {
            "tp_first_proba": [0.8],
            "sl_first_proba": [0.2],
            "survival_confidence": [1.0],
            "volatility_after_entry_pred": [0.0],
            "ATR_14": [1.0],
            "expected_exit_time": [50.0],
        }
    )
    res = optimise_exits(df)
    assert "recommended_tp_pct" in res
    assert "recommended_sl_pct" in res
    assert res["recommended_tp_pct"].iloc[0] >= res["recommended_sl_pct"].iloc[0]


def test_penalties_applied():
    df = pd.DataFrame(
        {
            "tp_first_proba": [0.5, 0.5],
            "sl_first_proba": [0.5, 0.5],
            "survival_confidence": [1.0, 0.1],
            "volatility_after_entry_pred": [0.0, -1.0],
            "ATR_14": [1.0, 1.0],
            "expected_exit_time": [10.0, 10.0],
        }
    )
    res = optimise_exits(df)
    assert res["recommended_tp_pct"].notna().all()
    assert res["recommended_sl_pct"].notna().all()
    # Lower survival and negative volatility should still produce finite outputs
    assert np.isfinite(res.to_numpy()).all()


def test_missing_atr_handled():
    df = synthetic_df(3)
    df.loc[0, "ATR_14"] = 0.0
    res = optimise_exits(df)
    assert res["recommended_tp_pct"].notna().all()
    assert res["recommended_sl_pct"].notna().all()


def test_no_nans_output():
    df = synthetic_df(10)
    res = optimise_exits(df)
    assert not res.isna().any().any()
    assert np.isfinite(res.to_numpy()).all()


def test_max_expectancy_selection():
    # Construct two rows where one clearly dominates
    df = pd.DataFrame(
        {
            "tp_first_proba": [0.9, 0.1],
            "sl_first_proba": [0.1, 0.9],
            "survival_confidence": [1.0, 1.0],
            "volatility_after_entry_pred": [0.0, 0.0],
            "ATR_14": [1.0, 1.0],
            "expected_exit_time": [10.0, 10.0],
        }
    )
    res = optimise_exits(df)
    assert res["recommended_tp_pct"].iloc[0] >= res["recommended_sl_pct"].iloc[0]
    assert res["recommended_tp_pct"].iloc[1] >= res["recommended_sl_pct"].iloc[1]


def test_tiny_dataset():
    df = synthetic_df(1)
    res = optimise_exits(df)
    assert len(res) == 1
    assert res["recommended_tp_pct"].notna().all()
    assert res["recommended_sl_pct"].notna().all()
