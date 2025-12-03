import logging
import platform
import time
from pathlib import Path

import pandas as pd

from drift_detector import DriftDetector
from normalisation_engine import NormalisationEngine


ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "PILLAR_FOUR_REGIME_DETECTION" / "data" / "processed"
NORMALISATION_DIR = DATA_PROCESSED / "normalisation"
NORMALISATION_DIR.mkdir(parents=True, exist_ok=True)


def _load_phase0(logger: logging.Logger) -> pd.DataFrame:
    candidates = [
        DATA_PROCESSED / "phase0" / "canonical_raw_with_utilities.parquet",
        DATA_PROCESSED / "raw_layer" / "canonical_raw_with_utilities.parquet",
    ]
    for path in candidates:
        if path.exists():
            logger.info("Loading Phase 0 output from %s", path)
            return pd.read_parquet(path)
    raise FileNotFoundError("canonical_raw_with_utilities.parquet not found in expected locations.")


def _log_example_stats(df: pd.DataFrame, logger: logging.Logger) -> None:
    cols = [c for c in df.columns if c.endswith(("_zscore", "_mad", "_volnorm", "_minmax", "_log", "_regime_norm"))]
    if not cols:
        return
    sample_cols = cols[:5]
    desc = df[sample_cols].agg(["min", "max", "mean", "std"]).T
    logger.info("Example stats for normalised features:\n%s", desc)


def _log_missing(df: pd.DataFrame, logger: logging.Logger) -> None:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        logger.info("No missing values in output.")
    else:
        logger.info("Missing values (top 10):\n%s", missing.head(10))


def _log_df_overview(df: pd.DataFrame, label: str, logger: logging.Logger) -> None:
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    logger.info("%s shape=%s cols=%d approx_mem=%.2fMB", label, df.shape, len(df.columns), mem_mb)


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("phase0.run_norm")
    logger.info("Starting normalisation on %s (%s)", platform.system(), platform.machine())

    stage_start = time.time()
    raw_df = _load_phase0(logger)
    _log_df_overview(raw_df, "Raw Phase0 data", logger)

    engine = NormalisationEngine(logger=logger)

    logger.info(
        "Fitting normalisation engine on shape %s with progress every %.1fs",
        raw_df.shape,
        engine.config.progress_interval_seconds,
    )
    fit_start = time.time()
    norm_df = engine.fit_transform(raw_df)
    logger.info("Fit/transform completed in %.2fs", time.time() - fit_start)
    _log_df_overview(norm_df, "Normalised data", logger)
    logger.info("Total features normalised: %d", len(engine.feature_map))

    drift_report_path = NORMALISATION_DIR / "drift_report.txt"
    drift_start = time.time()
    DriftDetector(output_path=drift_report_path, logger=logger).run(
        raw_df, [entry["feature"] for entry in engine.feature_map]
    )
    logger.info("Drift detection finished in %.2fs", time.time() - drift_start)

    normalised_path = NORMALISATION_DIR / "normalised_features.parquet"
    params_path = NORMALISATION_DIR / "normalisation_params.json"

    norm_df.to_parquet(normalised_path, index=False)
    engine.save(params_path)

    _log_example_stats(norm_df, logger)
    _log_missing(norm_df, logger)

    if "vol_regime" in raw_df.columns:
        counts = raw_df["vol_regime"].value_counts(dropna=False)
        logger.info("Regime distribution:\n%s", counts)

    logger.info("Saved normalised features to %s", normalised_path)
    logger.info("Saved normalisation params to %s", params_path)
    logger.info("Saved drift report to %s", drift_report_path)
    logger.info("End-to-end normalisation run completed in %.2fs", time.time() - stage_start)


if __name__ == "__main__":
    run()
