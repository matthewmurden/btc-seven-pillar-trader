import pandas as pd
import numpy as np


def _future_ret(close: pd.Series, horizon: int) -> pd.Series:
    """Future percentage return."""
    return (close.shift(-horizon) - close) / close.replace(0, np.nan)


def validate_label_distribution(out: pd.DataFrame) -> None:
    """Print basic distribution diagnostics for labels."""
    try:
        print("\n[Label Builder] Distribution diagnostics:")
        for col in ["next_1m_up", "next_5m_up", "next_15m_trend", "breakout_target", "reversion_target"]:
            if col in out.columns:
                pct_pos = out[col].mean() * 100
                print(f"  {col}: {pct_pos:.2f}% positives")

        ret_15 = _future_ret(out["close"], 15)
        print(f"  future_ret_15m min/mean/max: {ret_15.min():.4f}/{ret_15.mean():.4f}/{ret_15.max():.4f}")

        sample = out[
            [
                "close",
                "next_1m_up",
                "next_5m_up",
                "next_15m_trend",
                "breakout_target",
                "reversion_target",
            ]
        ].head(20)
        print("\n[Label Builder] Preview (first 20 rows):")
        print(sample)
    except Exception as exc:
        print(f"[Label Builder] Diagnostics failed: {exc}")


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean, deterministic label builder using only price-based logic.

    Outputs:
        next_1m_up
        next_5m_up
        next_15m_trend
        breakout_target
        reversion_target
    """

    out = df.copy()

    # -------------------------
    # Safety: required columns
    # -------------------------
    if "close" not in out:
        raise ValueError("label_builder: df must contain 'close'")
    if "high" not in out:
        print("[Warning] 'high' missing — using close as fallback.")
        out["high"] = out["close"]
    if "low" not in out:
        print("[Warning] 'low' missing — using close as fallback.")
        out["low"] = out["close"]

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)

    # ================================================================
    # 1 — DIRECTIONAL LABELS
    # ================================================================
    out["next_1m_up"] = (close.shift(-1) > close).astype(int)
    out["next_5m_up"] = (close.shift(-5) > close).astype(int)
    out["next_15m_trend"] = (_future_ret(close, 15) > 0).astype(int)

    # ================================================================
    # 2 — BREAKOUT LABEL (pure price)
    #    Up or down breakout within next 5 bars beyond a small threshold
    # ================================================================
    future_high_5 = high.shift(-1).rolling(5, min_periods=1).max()
    future_low_5 = low.shift(-1).rolling(5, min_periods=1).min()
    up_break = (future_high_5 - high) / high.replace(0, np.nan) > 0.004  # >0.4%
    down_break = (future_low_5 - low) / low.replace(0, np.nan) < -0.004  # < -0.4%
    out["breakout_target"] = (up_break | down_break).astype(int)

    # ================================================================
    # 3 — REVERSION LABEL (snapback within 5–10 bars)
    #    Detect a spike then return near the origin after ~10 bars
    # ================================================================
    future_max_5 = high.shift(-1).rolling(5, min_periods=1).max()
    future_min_5 = low.shift(-1).rolling(5, min_periods=1).min()
    future_close_10 = close.shift(-10)

    up_spike = (future_max_5 - close) / close.replace(0, np.nan) > 0.003  # >0.3%
    down_spike = (future_min_5 - close) / close.replace(0, np.nan) < -0.003  # <-0.3%

    mean_revert = future_close_10.notna() & (np.abs((future_close_10 - close) / close.replace(0, np.nan)) < 0.002)
    out["reversion_target"] = ((up_spike | down_spike) & mean_revert).astype(int)

    # Range warnings
    br_mean = out["breakout_target"].mean()
    rev_mean = out["reversion_target"].mean()
    if br_mean < 0.03 or br_mean > 0.12:
        print("[Warning] breakout probability out of expected range")
    if rev_mean < 0.10 or rev_mean > 0.25:
        print("[Warning] reversion probability out of expected range")

    # ================================================================
    # Final cleanup and diagnostics
    # ================================================================
    out = out.ffill().bfill()
    validate_label_distribution(out)
    return out
