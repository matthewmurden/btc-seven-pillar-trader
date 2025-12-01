"""
Ad-hoc analyzer for deterministic engine outputs.
Run directly: python tests/analyze_deterministic_engines.py
Prints descriptive stats and distributions for deterministic_regime_engines.parquet.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "deterministic_regime_engines.parquet"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing deterministic engine output: {DATA_PATH}")
    return pd.read_parquet(DATA_PATH)


def print_header(df: pd.DataFrame) -> None:
    print("\n=== File Info ===")
    print(f"Path: {DATA_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print("\nHead:\n", df.head())


def describe_numeric(df: pd.DataFrame) -> None:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return
    desc = df[num_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    print("\n=== Numeric Summary (with percentiles) ===")
    print(desc)


def print_distributions(df: pd.DataFrame) -> None:
    print("\n=== Distributions ===")
    for col in [
        "trend_flag",
        "trend_direction",
        "chop_flag",
        "volatility_regime",
        "liquidity_regime",
        "mean_reversion_flag",
    ]:
        if col in df.columns:
            counts = df[col].value_counts(normalize=True).sort_index()
            print(f"{col}: {counts.to_dict()}")


def print_cross_tabs(df: pd.DataFrame) -> None:
    print("\n=== Cross Tabs ===")
    if {"volatility_regime", "liquidity_regime"}.issubset(df.columns):
        vt = pd.crosstab(df["volatility_regime"], df["liquidity_regime"], normalize="all").round(4)
        print("Volatility vs Liquidity (fraction of total):\n", vt)
    if {"trend_flag", "chop_flag"}.issubset(df.columns):
        tc = pd.crosstab(df["trend_flag"], df["chop_flag"], normalize="all").round(4)
        print("\nTrend vs Chop (fraction of total):\n", tc)


def print_score_snaps(df: pd.DataFrame) -> None:
    print("\n=== Score Snapshots ===")
    for col in [
        "trend_strength_score",
        "chop_strength_score",
        "volatility_score",
        "liquidity_score",
        "mean_reversion_score",
    ]:
        if col in df.columns:
            s = df[col]
            print(
                f"{col}: mean={s.mean():.4f} std={s.std():.4f} "
                f"min={s.min():.4f} p25={s.quantile(0.25):.4f} "
                f"p50={s.quantile(0.50):.4f} p75={s.quantile(0.75):.4f} max={s.max():.4f}"
            )


def main():
    with pd.option_context("display.max_columns", None, "display.width", 140):
        df = load_data()
        print_header(df)
        describe_numeric(df)
        print_distributions(df)
        print_cross_tabs(df)
        print_score_snaps(df)


if __name__ == "__main__":
    main()
