"""Data loading utilities for Pillar Two exit modelling raw inputs."""

from pathlib import Path
from typing import List

import pandas as pd

PILLAR_TWO_ROOT = Path(__file__).resolve().parent.parent
ENTRY_OUTPUT_CANDIDATES = [
    PILLAR_TWO_ROOT.parent
    / "PILLAR_ONE_ENTRY_MODEL"
    / "data"
    / "processed"
    / "entry_features.parquet",
    PILLAR_TWO_ROOT.parent
    / "PILLAR_ONE_ENTRY_MODEL"
    / "data"
    / "processed"
    / "features_with_entry_probability.parquet",
]
RAW_EXIT_OUTPUT_PATH = PILLAR_TWO_ROOT / "data" / "raw" / "exit_raw.parquet"

RAW_EXIT_COLUMNS: List[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "delta",
    "wick_top",
    "wick_bottom",
    "volatility",
    "agg_trades",
    "micro_price_change",
    "micro_spread",
]

ALIAS_COLUMNS = {
    "wick_top": "upper_wick",
    "wick_bottom": "lower_wick",
}


def load_entry_outputs(path: Path | None = None) -> pd.DataFrame:
    """
    Load the final exported dataframe from Pillar One.

    Parameters
    ----------
    path : Path | None
        Optional override for the Pillar One entry features parquet file.

    Returns
    -------
    pd.DataFrame
        The loaded entry dataframe.
    """
    if path is not None:
        candidates = [path]
    else:
        candidates = ENTRY_OUTPUT_CANDIDATES

    for candidate in candidates:
        if candidate.exists():
            return pd.read_parquet(candidate)

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Pillar One entry outputs not found at any of the expected paths:\n{searched}"
    )


def load_raw_exit_inputs(entry_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the raw columns required for exit modelling.

    Parameters
    ----------
    entry_df : pd.DataFrame
        DataFrame loaded from Pillar One outputs.

    Returns
    -------
    pd.DataFrame
        DataFrame with only the required raw exit columns.
    """
    df = entry_df.copy()

    # Apply column aliases for naming differences between pillars.
    for target, alias in ALIAS_COLUMNS.items():
        if target not in df.columns and alias in df.columns:
            df[target] = df[alias]

    missing = [col for col in RAW_EXIT_COLUMNS if col not in df.columns]

    # Derive aggregated trades if not present but constituent counts exist.
    if "agg_trades" in missing and {"taker_buy_qty", "taker_sell_qty"}.issubset(
        df.columns
    ):
        df["agg_trades"] = df["taker_buy_qty"] + df["taker_sell_qty"]
        missing = [col for col in RAW_EXIT_COLUMNS if col not in df.columns]

    # Provide placeholders for microstructure metrics if absent.
    for micro_col in ("micro_price_change", "micro_spread"):
        if micro_col in missing:
            df[micro_col] = pd.NA
    missing = [col for col in RAW_EXIT_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns for raw exit inputs: {', '.join(missing)}"
        )

    return df[RAW_EXIT_COLUMNS].copy()


def save_raw_exit_inputs(raw_df: pd.DataFrame, path: Path = RAW_EXIT_OUTPUT_PATH) -> Path:
    """
    Save the raw exit inputs dataframe to the configured parquet path.

    Parameters
    ----------
    raw_df : pd.DataFrame
        DataFrame containing the raw exit inputs.
    path : Path
        Destination path for saving the parquet file.

    Returns
    -------
    Path
        The path where the file was saved.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_parquet(path)
    return path
