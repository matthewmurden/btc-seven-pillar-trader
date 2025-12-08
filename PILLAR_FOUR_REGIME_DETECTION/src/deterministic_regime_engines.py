import logging
from pathlib import Path

import pandas as pd

from engines.trend_engine import compute_trend_engine
from engines.chop_engine import compute_chop_engine
from engines.volatility_engine import compute_volatility_engine
from engines.liquidity_engine import compute_liquidity_engine
from engines.mean_reversion_engine import compute_mean_reversion_engine


# ---------------------------------------------------------------------
# Paths – consistent with build_raw_regime_features.py
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "part_one" / "raw_regime_features.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "part_two" / "deterministic_regime_engines.parquet"


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def load_raw_features() -> pd.DataFrame:
    """
    Load raw_regime_features.parquet produced by Pillar 4 Part 1.
    If it is missing, rebuild it from normalised_features using
    build_raw_regime_features.main().
    """
    if not INPUT_PATH.exists():
        from build_raw_regime_features import main as build_part_one  # type: ignore

        logging.info("raw_regime_features missing; regenerating from normalised_features")
        build_part_one()

    df = pd.read_parquet(INPUT_PATH)

    # Ensure deterministic ordering by time
    if "open_time_ms" in df.columns:
        df = df.sort_values("open_time_ms").reset_index(drop=True)

    return df


def build_engines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all deterministic regime engines on the raw regime feature matrix.

    Input:  raw_regime_features (full feature matrix from Part 1)
    Output: DataFrame containing:
      - open_time_ms
      - trend_engine outputs
      - chop_engine outputs
      - volatility_engine outputs
      - liquidity_engine outputs
      - mean_reversion_engine outputs
    """
    # Compute each engine on the same index
    trend_df = compute_trend_engine(df)
    chop_df = compute_chop_engine(df)
    vol_df = compute_volatility_engine(df)
    liq_df = compute_liquidity_engine(df)
    mr_df = compute_mean_reversion_engine(df)

    # Sanity: all engines should have the same index as the raw features
    for name, eng in [
        ("trend_engine", trend_df),
        ("chop_engine", chop_df),
        ("volatility_engine", vol_df),
        ("liquidity_engine", liq_df),
        ("mean_reversion_engine", mr_df),
    ]:
        if not eng.index.equals(df.index):
            logging.warning(
                "%s output index does not match raw features index; aligning by position.",
                name,
            )
            eng = eng.reindex(df.index)

    # Concatenate all engine outputs column-wise
    engines_only = pd.concat([trend_df, chop_df, vol_df, liq_df, mr_df], axis=1)

    # Attach open_time_ms for alignment/joining with other pillars
    if "open_time_ms" in df.columns:
        out = pd.concat([df[["open_time_ms"]], engines_only], axis=1)
    else:
        logging.warning("open_time_ms missing from raw features; output will not include timestamps.")
        out = engines_only.copy()

    # Clean up any stray NaNs (should be rare due to engine defaults)
    out = out.fillna(0)

    return out


def save_output(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    logging.info("Saved deterministic engines → %s", OUTPUT_PATH)


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== Deterministic Regime Engines Summary ===")
    print(f"Rows: {len(df):,}")
    print(df.head())
    for col in df.columns:
        col_min, col_max = df[col].min(), df[col].max()
        print(f"{col}: min={col_min}, max={col_max}")
    print("===========================================\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    raw = load_raw_features()
    engines_df = build_engines(raw)
    save_output(engines_df)
    print_summary(engines_df)


if __name__ == "__main__":
    main()