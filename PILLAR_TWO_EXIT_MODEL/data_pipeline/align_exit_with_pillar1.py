import pandas as pd
from pathlib import Path
from utils.timestamp_utils import normalise_timestamp


def align_exit_with_pillar1():
    p_exit = Path("data/processed/features_part3_with_labels.parquet")
    p_p1 = Path("../PILLAR_ONE_ENTRY_MODEL/data/processed/pillar1_export.parquet")

    exit_df = normalise_timestamp(pd.read_parquet(p_exit))
    p1_df = normalise_timestamp(pd.read_parquet(p_p1)[["timestamp"]])

    # REAL alignment — timestamp intersection
    aligned = exit_df.merge(p1_df, on="timestamp", how="inner")
    aligned = aligned.sort_values("timestamp").reset_index(drop=True)

    out = Path("data/processed/features_exit_aligned.parquet")
    aligned.to_parquet(out)

    print("Exit aligned dataset saved:", out)
    print("Rows:", len(aligned))
    print("Columns:", len(aligned.columns))
    print("Timestamp range:", aligned.timestamp.min(), "→", aligned.timestamp.max())


if __name__ == "__main__":
    align_exit_with_pillar1()
