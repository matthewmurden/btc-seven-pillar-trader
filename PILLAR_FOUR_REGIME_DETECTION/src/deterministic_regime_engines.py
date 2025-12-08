import logging
from pathlib import Path

import pandas as pd

from engines.trend_engine import compute_trend_engine
from engines.chop_engine import compute_chop_engine
from engines.volatility_engine import compute_volatility_engine
from engines.liquidity_engine import compute_liquidity_engine
from engines.mean_reversion_engine import compute_mean_reversion_engine

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "part_one" / "raw_regime_features.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "part_two" / "deterministic_regime_engines.parquet"


def load_raw_features() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        # regenerate part_one features from normalised inputs
        from build_raw_regime_features import main as build_part_one  # type: ignore

        logging.info("raw_regime_features missing; regenerating from normalised_features")
        build_part_one()
    df = pd.read_parquet(INPUT_PATH)
    if "open_time_ms" in df.columns:
        df = df.sort_values("open_time_ms").reset_index(drop=True)
    return df


def build_engines(df: pd.DataFrame) -> pd.DataFrame:
    engines = [
        compute_trend_engine(df),
        compute_chop_engine(df),
        compute_volatility_engine(df),
        compute_liquidity_engine(df),
        compute_mean_reversion_engine(df),
    ]
    out = pd.concat(engines, axis=1)
    out = out.fillna(0)
    return out


def save_output(df: pd.DataFrame):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved deterministic engines → %s", OUTPUT_PATH)


def print_summary(df: pd.DataFrame):
    print("\n=== Deterministic Regime Engines Summary ===")
    print(f"Rows: {len(df):,}")
    print(df.head())
    for col in df.columns:
        col_min, col_max = df[col].min(), df[col].max()
        print(f"{col}: min={col_min}, max={col_max}")
    print("===========================================\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    raw = load_raw_features()
    engines_df = build_engines(raw)
    save_output(engines_df)
    print_summary(engines_df)


if __name__ == "__main__":
    main()
