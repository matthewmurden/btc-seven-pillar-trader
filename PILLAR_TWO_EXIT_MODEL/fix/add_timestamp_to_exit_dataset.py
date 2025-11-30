import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/exit_raw.parquet")
EXIT_PATH = Path("data/processed/features_part3_with_labels.parquet")
OUT_PATH = Path("data/processed/features_part3_with_labels_timestamp.parquet")

print("Loading raw exit file...")
raw = pd.read_parquet(RAW_PATH)
print("Raw shape:", raw.shape)

print("Loading processed exit dataset...")
exit_df = pd.read_parquet(EXIT_PATH)
print("Exit DF shape:", exit_df.shape)

timestamp_series = None
source = "raw"
if "timestamp" in raw.columns:
    timestamp_series = raw["timestamp"]
else:
    candidates = [
        Path("../PILLAR_ONE_ENTRY_MODEL/data/processed/pillar1_export.parquet"),
        Path("../PILLAR_ONE_ENTRY_MODEL/data/processed/pillar1_outputs.parquet"),
    ]
    for cand in candidates:
        if cand.exists():
            print(f"Raw missing timestamp. Using Pillar 1 timestamps from {cand}")
            p1 = pd.read_parquet(cand)
            if "timestamp" in p1.columns:
                timestamp_series = p1["timestamp"]
                source = cand.name
                break
    if timestamp_series is None:
        print("⚠️ No timestamp in raw or Pillar 1 exports. Generating synthetic timestamps.")
        timestamp_series = pd.date_range("2000-01-01", periods=len(exit_df), freq="T")
        source = "synthetic"

timestamp_series = pd.to_datetime(timestamp_series)

min_len = min(len(timestamp_series), len(exit_df))
if len(timestamp_series) != len(exit_df):
    print(f"⚠️ WARNING: Length mismatch: ts_source={len(timestamp_series)}, exit_df={len(exit_df)}")
    print(f"Truncating both to minimum length: {min_len}")

if hasattr(timestamp_series, "iloc"):
    timestamp_series = timestamp_series.iloc[:min_len].reset_index(drop=True)
else:
    timestamp_series = timestamp_series[:min_len]
exit_df = exit_df.iloc[:min_len].reset_index(drop=True)

exit_df.insert(0, "timestamp", timestamp_series.values)

print(f"Saving fixed dataset with timestamps from {source}:", OUT_PATH)
exit_df.to_parquet(OUT_PATH, index=False)
print("✓ Timestamp successfully added")
