import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq

from src.data_loader.api.api_htf_loader import load_htf
from src.data_loader.bulk.bulk_trades_streamer import stream_bulk_trade_chunks
from src.data_loader.bulk.bulk_trades_aggregator import aggregate_trade_chunk
from src.feature_engine.feature_engine import compute_all_features


TMP_FILE = "data/processed/tmp_trades_5m.parquet"
FINAL_FILE = "data/processed/features_5m.parquet"


def build_training_dataset():

    # ----------------------------------------------------
    print("[1/4] Loading 5m candles...")
    # ----------------------------------------------------
    df_c = pd.read_csv("data/raw/candles_5m.csv")
    df_c = df_c.sort_values("timestamp").reset_index(drop=True)

    # ----------------------------------------------------
    print("[2/4] Streaming & aggregating bulk trades (ParquetWriter)...")
    # ----------------------------------------------------

    # If an old temp file exists, delete it
    if os.path.exists(TMP_FILE):
        os.remove(TMP_FILE)

    writer = None
    total_chunks = 0

    for i, chunk in enumerate(stream_bulk_trade_chunks("data/raw/bulk_trades")):
        print(f"   → Processing trade chunk {i+1}")

        aggregated = aggregate_trade_chunk(chunk)
        table = pa.Table.from_pandas(aggregated)

        if writer is None:
            writer = pq.ParquetWriter(TMP_FILE, table.schema)

        writer.write_table(table)
        total_chunks += 1

    if writer:
        writer.close()

    print(f"   → Completed {total_chunks} aggregated chunks.")

    # ----------------------------------------------------
    print("[2.5/4] Loading aggregated 5m trades...")
    # ----------------------------------------------------
    df_trades_5m = pd.read_parquet(TMP_FILE)
    df_trades_5m = df_trades_5m.sort_values("timestamp").drop_duplicates()

    # ----------------------------------------------------
    print("[3/4] Loading HTF candles (15m, 1h)...")
    # ----------------------------------------------------
    df_htf = load_htf("BTCUSDT")

    # ----------------------------------------------------
    print("[4/4] Computing features...")
    # ----------------------------------------------------
    df_features = compute_all_features(df_c, df_trades_5m, df_htf)

    df_features.to_parquet(FINAL_FILE, index=False)
    print("[✔] Saved training dataset →", FINAL_FILE)


if __name__ == "__main__":
    build_training_dataset()