"""
Time-travel invariance test for feature leakage.

For sampled timestamps t in a fold window, recompute features using only data up
to t and verify they match the original full-window features. Any deviation
means forward leakage.

Usage example:
python RUN_LAYER_OPTIMISING/scripts/leak_test.py \\
  --fold-cache RUN_LAYER_OPTIMISING/data/cache/fold_20250328 \\
  --window test \\
  --processed-snapshot /path/to/processed_fit_snapshot \\
  --samples 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# We reuse the pipeline pieces directly.
sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
import run_hparam_search as hps  # type: ignore


def _window_bounds(meta: Dict[str, str], window: str) -> Tuple[int, int]:
    if window not in {"val", "test"}:
        raise ValueError("window must be one of ['val','test']")
    if window == "val":
        start = meta["val_start_utc"]
        end = meta["test_start_utc"]
    else:
        start = meta["test_start_utc"]
        end = meta["test_end_utc"]
    start_ms = int(pd.to_datetime(start, utc=True).value // 1_000_000)
    end_ms = int(pd.to_datetime(end, utc=True).value // 1_000_000)
    return start_ms, end_ms


def _load_baseline(fold_cache: Path, window: str) -> Tuple[pd.DataFrame, Dict[str, str], Tuple[int, int]]:
    meta_path = fold_cache / "fold_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"fold_meta.json missing under {fold_cache}")
    meta = json.loads(meta_path.read_text())
    merged_path = fold_cache / f"merged_{window}.parquet"
    if not merged_path.exists():
        raise FileNotFoundError(f"{merged_path} missing")
    df = pd.read_parquet(merged_path).sort_values("open_time_ms").reset_index(drop=True)
    bounds = _window_bounds(meta, window)
    return df, meta, bounds


def _clear_raw_dir() -> None:
    shutil.rmtree(hps.RAW_DIR, ignore_errors=True)
    hps.RAW_DIR.mkdir(parents=True, exist_ok=True)


def _reset_processed_from_snapshot(snapshot: Path | None, gse_params: Path | None) -> None:
    """
    Reset processed dir to a clean copy of the provided snapshot (if given),
    and drop in GSE params if supplied. Always clears processed to avoid
    cross-sample contamination.
    """
    shutil.rmtree(hps.PROCESSED_DIR, ignore_errors=True)
    hps.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if snapshot is not None:
        if not snapshot.exists():
            raise FileNotFoundError(f"Processed snapshot missing during restore: {snapshot}")
        shutil.copytree(snapshot, hps.PROCESSED_DIR, dirs_exist_ok=True)
    if gse_params is not None:
        if not gse_params.exists():
            raise FileNotFoundError(f"GSE params path provided but missing: {gse_params}")
        dst = hps.PREF4_PROCESSED / "normalisation_params.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gse_params, dst)


def _build_features_window(start_ms: int, end_ms: int, snapshot: Path | None, gse_params: Path | None) -> pd.DataFrame:
    _clear_raw_dir()
    _reset_processed_from_snapshot(snapshot, gse_params)
    hps._patch_offline_paths()
    os.environ["PIPELINE_MODE"] = "test"
    os.environ["INFERENCE_ONLY"] = "1"
    hps.offline_fetch.fetch_raw_window(start_ms=start_ms, end_ms=end_ms)
    hps._run_post_raw_build_chain()
    hps._run_pipeline_infer()
    merged = hps._merge_p1_p2()
    merged = merged[(merged["open_time_ms"] >= start_ms) & (merged["open_time_ms"] < end_ms)].copy()
    return merged.sort_values("open_time_ms").reset_index(drop=True)


def _maybe_swap_processed(snapshot: Path | None) -> Path | None:
    """
    Optionally swap in a FIT snapshot (processed artifacts). Returns backup path
    to restore, or None if no swap was done.
    """
    if snapshot is None:
        return None
    if not snapshot.exists():
        raise FileNotFoundError(
            f"--processed-snapshot does not exist: {snapshot}. "
            "Pass a real FIT snapshot path, or omit the flag to use current processed artifacts."
        )
    processed = hps.PROCESSED_DIR
    backup = processed.parent / f"{processed.name}_leaktest_backup"
    if backup.exists():
        raise RuntimeError(f"Backup path already exists, refusing to overwrite: {backup}")
    if processed.exists():
        shutil.move(processed, backup)
    shutil.copytree(snapshot, processed)
    return backup


def _restore_processed(backup: Path | None) -> None:
    if backup is None:
        return
    processed = hps.PROCESSED_DIR
    shutil.rmtree(processed, ignore_errors=True)
    shutil.move(backup, processed)


def run_leak_test(
    fold_cache: Path,
    window: str,
    processed_snapshot: Path | None,
    gse_params: Path | None,
    samples: int,
    seed: int,
    tolerance: float,
    out_path: Path,
) -> None:
    df_full, meta, (start_ms, _) = _load_baseline(fold_cache, window)
    feature_cols = [c for c in df_full.columns if c != "open_time_ms"]

    rng = np.random.default_rng(seed)
    sample_times = df_full["open_time_ms"].to_numpy()
    if len(sample_times) == 0:
        raise RuntimeError("No rows in baseline features.")
    take = min(samples, len(sample_times))
    times = rng.choice(sample_times, size=take, replace=False)

    backup = _maybe_swap_processed(processed_snapshot)
    results: List[Dict[str, object]] = []
    try:
        for t in sorted(times):
            end_ms = int(t + 60_000)  # include the sampled bar
            recomputed = _build_features_window(
                start_ms=start_ms, end_ms=end_ms, snapshot=processed_snapshot, gse_params=gse_params
            )
            base_row = df_full[df_full["open_time_ms"] == t]
            new_row = recomputed[recomputed["open_time_ms"] == t]

            if base_row.empty or new_row.empty:
                results.append(
                    {"open_time_ms": int(t), "max_abs_diff": float("inf"), "status": "missing_row"}
                )
                continue

            base_row = base_row.iloc[0]
            new_row = new_row.iloc[0]
            diffs = []
            for col in feature_cols:
                b = base_row[col]
                n = new_row[col]
                diffs.append(abs(float(b) - float(n)))
            max_diff = float(max(diffs) if diffs else 0.0)
            status = "pass" if max_diff <= tolerance else "FAIL"
            results.append(
                {"open_time_ms": int(t), "max_abs_diff": max_diff, "status": status}
            )
    finally:
        _restore_processed(backup)

    out_df = pd.DataFrame(results).sort_values("open_time_ms")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    n_fail = (out_df["status"] == "FAIL").sum()
    logging.info("Done. %d samples, %d failures. Output: %s", len(out_df), n_fail, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-travel invariance leak test.")
    parser.add_argument("--fold-cache", type=Path, required=True, help="Fold cache dir (contains merged_*.parquet, fold_meta.json)")
    parser.add_argument("--window", choices=["val", "test"], default="test")
    parser.add_argument("--processed-snapshot", type=Path, help="Path to FIT processed snapshot to restore before testing")
    parser.add_argument("--gse-params", type=Path, help="Path to normalisation_params.json from FIT (Phase 0)")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--out", type=Path, help="Output CSV path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    fold_cache = args.fold_cache
    out_path = args.out or (fold_cache / f"leakage_invariance_{args.window}.csv")
    if args.processed_snapshot and not args.processed_snapshot.exists():
        raise FileNotFoundError(
            f"--processed-snapshot path not found: {args.processed_snapshot}. "
            "Provide the real FIT snapshot or omit the flag to use current processed artifacts."
        )

    run_leak_test(
        fold_cache=fold_cache,
        window=args.window,
        processed_snapshot=args.processed_snapshot,
        gse_params=args.gse_params,
        samples=args.samples,
        seed=args.seed,
        tolerance=args.tolerance,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
