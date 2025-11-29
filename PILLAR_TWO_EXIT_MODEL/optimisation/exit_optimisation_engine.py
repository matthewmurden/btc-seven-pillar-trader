"""Exit optimisation engine (Part 6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "exit_optimisation.parquet"

REQUIRED_COLS = [
    "tp_first_pred",
    "sl_first_pred",
    "survival_confidence",
    "volatility_score",
    "regime_state",
    "exit_time_pred",
]


def optimise_exit(df: pd.DataFrame) -> pd.DataFrame:
    """Optimise TP/SL per row using expectancy grid with penalties."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for optimisation: {missing}")

    out = df.copy()
    atr = out["atr"] if "atr" in out.columns else out.get("feature_volatility_atr")
    if atr is None:
        # derive ATR from OHLC if available
        if {"high", "low", "close"}.issubset(out.columns):
            prev_close = out["close"].shift(1)
            tr = pd.concat(
                [
                    out["high"] - out["low"],
                    (out["high"] - prev_close).abs(),
                    (out["low"] - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr = tr.rolling(14, min_periods=5).mean()
        else:
            raise ValueError("Missing ATR column (atr or feature_volatility_atr)")
    atr = atr.ffill().bfill()

    n = len(out)
    tp_probs = out["tp_first_pred"].astype(float).values
    sl_probs = out["sl_first_pred"].astype(float).values
    survival = out["survival_confidence"].astype(float).values
    vol_score = out["volatility_score"].astype(float).values
    regime = out["regime_state"].astype(int).values
    exit_time = out["exit_time_pred"].astype(float).values

    tp_grid = np.linspace(0.5, 4.0, 20)
    sl_grid = np.linspace(0.5, 3.0, 20)

    tp_candidates = atr.values[:, None] * tp_grid[None, :]
    sl_candidates = atr.values[:, None] * sl_grid[None, :]

    expectancy = tp_probs[:, None] * tp_candidates - sl_probs[:, None] * sl_candidates
    expectancy *= survival[:, None]

    vol_mask = vol_score[:, None] > 0.7
    expectancy = expectancy - vol_mask * (vol_score[:, None] * sl_candidates * 0.5)

    # regime adjustments
    regime_adjust = np.zeros_like(expectancy)
    regime_adjust[regime == 1] += tp_candidates[regime == 1] * 0.4  # stronger trend bonus
    regime_adjust[regime == 2] -= (
        tp_candidates[regime == 2] * 0.3 + sl_candidates[regime == 2] * 0.05
    )  # breakout cautious
    regime_adjust[regime == 0] -= tp_candidates[regime == 0] * 0.5  # strong range conservatism
    expectancy += regime_adjust

    best_idx = np.argmax(expectancy, axis=1)
    recommended_tp = tp_candidates[np.arange(n), best_idx]
    recommended_sl = sl_candidates[np.arange(n), best_idx]

    out["recommended_tp_pct"] = recommended_tp
    out["recommended_sl_pct"] = recommended_sl
    out["expected_exit_time"] = exit_time

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out[
        ["recommended_tp_pct", "recommended_sl_pct", "expected_exit_time"]
    ].to_parquet(OUTPUT_PATH)
    return out


if __name__ == "__main__":
    optimise_exit(pd.DataFrame())
