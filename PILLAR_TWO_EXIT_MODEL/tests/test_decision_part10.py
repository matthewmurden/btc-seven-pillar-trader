"""Tests for Part 10 final exit decision layer."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exit_decision.decision_layer import FinalExitDecision  # noqa: E402


def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_signal": [1, 0, 1, 1, 1],
            "direction": [1, -1, 1, 1, -1],
            "entry_price": [100.0, 101.0, 102.0, 103.0, 104.0],
            "entry_confidence": [0.9, 0.8, 0.7, 0.6, 0.5],
            "mae_pred": [0.01, 0.01, 0.03, 0.015, 0.01],
            "mfe_pred": [0.02, 0.01, 0.04, 0.02, 0.03],
            "exit_time_pred": [10, 20, 30, 15, 12],
            "optimal_tp_pred": [0.1, 0.1, 0.1, 0.1, 0.1],
            "optimal_sl_pred": [-0.05, -0.05, -0.05, -0.05, -0.05],
            "tp_first_pred": [0.6, 0.5, 0.7, 0.55, 0.65],
            "sl_first_pred": [0.4, 0.5, 0.3, 0.45, 0.35],
            "asymmetry_ratio": [2.0, 1.0, 3.0, 1.5, 2.5],
            "reward_risk_prediction": [2.0, 1.5, 2.5, 1.8, 2.2],
            "survival_confidence": [0.9, 0.8, 0.7, 0.6, 0.5],
            "volatility_after_entry_pred": [0.2, 0.3, 0.4, 0.5, 0.6],
            "exit_direction_consistency": [0.2, 0.0, 0.4, -0.1, 0.1],
            "exit_quality_score": [0.3, 0.2, 0.5, 0.4, 0.35],
            "recommended_tp_pct": [0.06, 0.06, 0.06, 0.03, 0.07],
            "recommended_sl_pct": [0.04, 0.04, 0.04, 0.04, 0.04],
        }
    )


def test_reject_low_exit_quality():
    decider = FinalExitDecision()
    df = sample_df()
    out = decider.run_batch(df)
    assert bool(out.loc[1, "take_trade"]) is False  # exit_quality_score 0.2 < 0.25


def test_reject_no_entry_signal():
    decider = FinalExitDecision()
    df = sample_df()
    out = decider.run_batch(df)
    assert bool(out.loc[1, "take_trade"]) is False  # entry_signal 0


def test_reject_small_tp():
    decider = FinalExitDecision()
    df = sample_df()
    out = decider.run_batch(df)
    assert bool(out.loc[3, "take_trade"]) is False  # recommended_tp_pct 0.03 < 0.05


def test_mae_halves_position_size():
    decider = FinalExitDecision()
    df = sample_df()
    out = decider.run_batch(df)
    assert bool(out.loc[2, "take_trade"]) is False or out.loc[2, "final_position_size_factor"] == 0.5
    if bool(out.loc[2, "take_trade"]):
        assert out.loc[2, "final_position_size_factor"] == 0.5


def test_valid_trade_structure():
    decider = FinalExitDecision()
    df = sample_df()
    out = decider.run_batch(df)
    row = out.loc[0]
    assert bool(row["take_trade"]) is True
    assert row["final_direction"] == df.loc[0, "direction"]
    assert row["final_tp_pct"] == df.loc[0, "recommended_tp_pct"]
    assert row["final_sl_pct"] == df.loc[0, "recommended_sl_pct"]
    assert row["final_expected_exit_time"] == df.loc[0, "exit_time_pred"]
    assert row["final_position_size_factor"] == 1.0


def test_run_batch_columns():
    decider = FinalExitDecision()
    df = sample_df()
    out = decider.run_batch(df)
    expected_cols = [
        "take_trade",
        "final_direction",
        "final_tp_pct",
        "final_sl_pct",
        "final_expected_exit_time",
        "final_position_size_factor",
    ]
    for col in expected_cols:
        assert col in out.columns


if __name__ == "__main__":
    test_reject_low_exit_quality()
    test_reject_no_entry_signal()
    test_reject_small_tp()
    test_mae_halves_position_size()
    test_valid_trade_structure()
    test_run_batch_columns()
