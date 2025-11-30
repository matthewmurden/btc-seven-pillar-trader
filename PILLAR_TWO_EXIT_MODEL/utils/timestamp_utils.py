import pandas as pd


def normalise_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timestamp is numeric then convert to datetime (seconds)."""
    if "timestamp" not in df.columns:
        raise ValueError("Missing timestamp column")
    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df
