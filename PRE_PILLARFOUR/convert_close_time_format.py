from pathlib import Path
import polars as pl


CANON_PATH = Path(__file__).resolve().parent / "processed" / "canonical_raw_with_utilities.parquet"


def main() -> None:
    if not CANON_PATH.exists():
        print(f"Missing file: {CANON_PATH}")
        return

    df = pl.read_parquet(CANON_PATH)
    if "close_time" not in df.columns:
        print("close_time column not found; nothing to do.")
        return

    df = df.with_columns(
        pl.col("close_time")
        .str.to_datetime(time_zone="UTC", strict=False)
        .dt.strftime("%Y-%m-%dT%H:%M:%S%.3fZ")
        .alias("close_time")
    )

    tmp_path = CANON_PATH.with_suffix(".parquet.tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(CANON_PATH)
    print(f"Updated close_time format and saved to {CANON_PATH}")


if __name__ == "__main__":
    main()
