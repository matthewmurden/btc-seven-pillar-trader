"""Load entry signals from Pillar One for the test period."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_entry_signals_for_test(test_len: int) -> pd.DataFrame:
    """
    Load entry outputs from Pillar One and align to test length.
    This uses a simple positional alignment for the backtest.
    """
    candidates = [
        PROJECT_ROOT.parent / "PILLAR_ONE_ENTRY_MODEL" / "data" / "processed" / "pillar1_outputs.parquet",
        PROJECT_ROOT.parent / "PILLAR_ONE_ENTRY_MODEL" / "data" / "processed" / "features_with_entry_probability.parquet",
        PROJECT_ROOT.parent / "PILLAR_ONE_ENTRY_MODEL" / "data" / "processed" / "features_with_final_signal.parquet",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_parquet(path)
            break
    else:
        raise FileNotFoundError("No Pillar One entry outputs found.")

    cols = []
    for c in ["timestamp", "close", "direction", "entry_probability", "entry_confidence", "signal", "entry_signal"]:
        if c in df.columns:
            cols.append(c)
    entry_df = df[cols].reset_index(drop=True)
    if "entry_signal" not in entry_df.columns and "signal" in entry_df.columns:
        entry_df["entry_signal"] = (entry_df["signal"] != 0).astype(int)
    entry_df.rename(columns={"close": "entry_price"}, inplace=True)

    return entry_df.iloc[-test_len:].reset_index(drop=True)
