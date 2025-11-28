# src/engines/trend/backtest_trend_engine.py
from pathlib import Path
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.engines.trend.trend_engine import compute_trend_engine_score
from src.engines.trend.trend_engine_config import TrendConfig


def main():
    print("\n------------------------------------------------")
    print("Trend Engine Backtest Results")
    print("------------------------------------------------")

    # Load data
    data_path = Path("data/processed/features_5m.parquet")
    df = pd.read_parquet(data_path)

    # Compute trend score
    cfg = TrendConfig()
    df = compute_trend_engine_score(df, cfg)
    trend = df["trend_score"]

    # Compute next candle move
    df["future_close"] = df["close"].shift(-1)
    df["next_up"] = (df["future_close"] > df["close"]).astype(int)
    df["next_down"] = (df["future_close"] < df["close"]).astype(int)

    # Remove last NaN row
    df = df.dropna(subset=["future_close"])

    # 1. Positive trend_score → next candle UP
    pos_prob = df[df["trend_score"] > 0]["next_up"].mean()

    # 2. Negative trend_score → next candle DOWN
    neg_prob = df[df["trend_score"] < 0]["next_down"].mean()

    print(f"Positive trend → next candle UP:   {pos_prob:.3f}")
    print(f"Negative trend → next candle DOWN: {neg_prob:.3f}")
    print("------------------------------------------------")

    # 3. Strong signals (top/bottom 10%)
    top_threshold = trend.quantile(0.9)
    bottom_threshold = trend.quantile(0.1)

    strong_up = df[df["trend_score"] >= top_threshold]["next_up"].mean()
    strong_down = df[df["trend_score"] <= bottom_threshold]["next_down"].mean()

    print(f"Strong positive (top 10%) → next up:  {strong_up:.3f}")
    print(f"Strong negative (bottom 10%) → next down: {strong_down:.3f}")
    print("------------------------------------------------")

    # 4. Regime-based results
    if "regime" in df.columns:
        print("Positive trend by regime:")
        print(df[df["trend_score"] > 0].groupby("regime")["next_up"].mean())
        print("------------------------------------------------")

        print("Strong trend by regime:")
        print(df[df["trend_score"] >= top_threshold].groupby("regime")["next_up"].mean())
        print("------------------------------------------------")
    else:
        print("No 'regime' column detected — skipping regime analysis.")

    # 5. Decile analysis
    df["decile"] = pd.qcut(df["trend_score"], 10, labels=False)

    decile_stats = df.groupby("decile")["next_up"].mean()
    print("Next-up probability by trend_score decile:")
    print(decile_stats)
    print("------------------------------------------------")


if __name__ == "__main__":
    main()