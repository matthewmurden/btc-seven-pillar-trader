import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl


DEFAULT_INPUT = Path("processed/canonical_with_all_engines.parquet")
DEFAULT_OUTPUT = Path("processed/normalised_features.parquet")
DEFAULT_PARAMS = Path("processed/normalisation_params.json")
DEFAULT_REGIME_COL = "vol_regime"

# Rolling windows / hyperparams
Z_WINDOW = 1000
MAD_WINDOW = 1000
DRIFT_LONG = 2000
DRIFT_SHORT = 200
MM_Q_LOW = 0.01
MM_Q_HIGH = 0.99
CLIP_Z = 5.0
CLIP_MAD = 6.0
CLIP_VOL = 6.0
CLIP_DRIFT = 8.0
EPS = 1e-9

# Default volatility references per feature (can be overridden via CLI-provided JSON)
DEFAULT_VOL_MAP = {
    "delta": "atr_14",
    "micro_delta": "atr_14",
    "agg_delta": "atr_14",
}


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def ensure_open_time_ms(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure we have an integer millisecond timestamp column called `open_time_ms`.
    Does NOT change row ordering.
    """
    if "open_time_ms" in df.columns:
        ts_max = df["open_time_ms"].max()
        # If something came in as ns, downscale to ms.
        if ts_max is not None and ts_max > 1_000_000_000_000_000:
            return df.with_columns((pl.col("open_time_ms") // 1_000_000).alias("open_time_ms"))
        return df

    if "open_time" in df.columns:
        if df.schema["open_time"] == pl.Utf8:
            ts = pl.col("open_time").str.to_datetime(time_zone="UTC")
        else:
            ts = pl.col("open_time")
        return df.with_columns(ts.dt.timestamp(time_unit="ms").alias("open_time_ms"))

    raise ValueError("open_time_ms/open_time missing from input data")


def clean_expr(col: str) -> pl.Expr:
    """
    Base cleaning expression:
    - cast to float
    - replace NaN/inf with 0 so downstream math doesn't explode
    NOTE: we only use this for *derived* columns; raw cols are not overwritten.
    """
    expr = pl.col(col).cast(pl.Float64)
    return pl.when(expr.is_infinite() | expr.is_nan()).then(0.0).otherwise(expr)


def rolling_z(df: pl.DataFrame, col: str) -> pl.DataFrame:
    mean = clean_expr(col).rolling_mean(window_size=Z_WINDOW, min_periods=1)
    std = clean_expr(col).rolling_std(window_size=Z_WINDOW, min_periods=1)
    z_expr = ((clean_expr(col) - mean) / (std.clip(1e-6, None) + EPS)).clip(-CLIP_Z, CLIP_Z).cast(pl.Float32)
    return df.with_columns(z_expr.alias(f"{col}_z"))


def rolling_mad(df: pl.DataFrame, col: str) -> pl.DataFrame:
    med = clean_expr(col).rolling_median(window_size=MAD_WINDOW, min_periods=1)
    mad = (clean_expr(col) - med).abs().rolling_median(window_size=MAD_WINDOW, min_periods=1)
    scale = (mad * 1.4826).clip(1e-6, None)
    mad_expr = ((clean_expr(col) - med) / (scale + EPS)).clip(-CLIP_MAD, CLIP_MAD).cast(pl.Float32)
    return df.with_columns(mad_expr.alias(f"{col}_mad"))


def vol_scale(df: pl.DataFrame, col: str, vol_col: str) -> pl.DataFrame:
    if vol_col not in df.columns:
        logging.warning("Vol ref %s missing for %s; skipping _vol", vol_col, col)
        return df
    expr = (clean_expr(col) / (clean_expr(vol_col) + EPS)).clip(-CLIP_VOL, CLIP_VOL).cast(pl.Float32)
    return df.with_columns(expr.alias(f"{col}_vol"))


def minmax_scale(
    df: pl.DataFrame,
    col: str,
    params: Optional[Dict] = None,
    fit: bool = True,
) -> Tuple[pl.DataFrame, Dict]:
    """
    Limited min-max scaling using global quantiles.
    """
    out = f"{col}_mm"
    if fit:
        series = df[col]
        if series.null_count() == series.len():
            logging.warning(
                "minmax_scale: column %s is all null/None; skipping _mm and filling 0.0",
                col,
            )
            params = {"q_low": 0.0, "q_high": 0.0, "skipped": True}
            return df.with_columns(pl.lit(0.0).cast(pl.Float32).alias(out)), params
        series = series.drop_nulls()
        series = series.filter(~series.is_nan())
        series = series.filter(~series.is_infinite())
        if series.len() == 0:
            logging.warning(
                "minmax_scale: column %s has no finite values; skipping _mm and filling 0.0",
                col,
            )
            params = {"q_low": 0.0, "q_high": 0.0, "skipped": True}
            return df.with_columns(pl.lit(0.0).cast(pl.Float32).alias(out)), params
        q_low = series.quantile(MM_Q_LOW)
        q_high = series.quantile(MM_Q_HIGH)
        if q_low is None or q_high is None:
            logging.warning("minmax_scale: column %s quantile None; skipping _mm and filling 0.0", col)
            params = {"q_low": 0.0, "q_high": 0.0, "skipped": True}
            return df.with_columns(pl.lit(0.0).cast(pl.Float32).alias(out)), params
        q_low = float(q_low)
        q_high = float(q_high)
        if not math.isfinite(q_low) or not math.isfinite(q_high):
            logging.warning(
                "minmax_scale: column %s quantiles not finite (q_low=%s, q_high=%s); skipping _mm and filling 0.0",
                col,
                q_low,
                q_high,
            )
            params = {"q_low": 0.0, "q_high": 0.0, "skipped": True}
            return df.with_columns(pl.lit(0.0).cast(pl.Float32).alias(out)), params
        if q_high <= q_low:
            q_high = q_low + 1e-6
        params = {"q_low": q_low, "q_high": q_high}
    else:
        if not params or params.get("skipped"):
            logging.warning("minmax_scale: column %s marked skipped; filling _mm with 0.0", col)
            return df.with_columns(pl.lit(0.0).cast(pl.Float32).alias(out)), params or {"skipped": True}
        q_low, q_high = params["q_low"], params["q_high"]

    expr = (
        (clean_expr(col).clip(q_low, q_high) - q_low)
        / (q_high - q_low + EPS)
    ).clip(0, 1).cast(pl.Float32)
    return df.with_columns(expr.alias(out)), params


def log_scale(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """
    Signed log scaling: sign(x) * log(1 + |x|)
    """
    base = clean_expr(col)
    expr = (base.abs() + 1.0).log() * base.sign()
    return df.with_columns(expr.cast(pl.Float32).alias(f"{col}_log"))


def drift_metric(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """
    Simple drift metric:
    (short_mean - long_mean) / long_std, all shifted by 1 bar to avoid
    using the current bar in the statistic.
    """
    base = clean_expr(col)
    long_mean = base.rolling_mean(window_size=DRIFT_LONG, min_periods=1).shift(1)
    long_std = base.rolling_std(window_size=DRIFT_LONG, min_periods=1).shift(1)
    short_mean = base.rolling_mean(window_size=DRIFT_SHORT, min_periods=1).shift(1)
    drift_expr = ((short_mean - long_mean) / (long_std.clip(1e-6, None) + EPS)).clip(-CLIP_DRIFT, CLIP_DRIFT)
    return df.with_columns(drift_expr.cast(pl.Float32).alias(f"{col}_drift"))


def regime_scale(
    df: pl.DataFrame,
    col: str,
    regime_col: str,
    params: Optional[Dict] = None,
    fit: bool = True,
) -> Tuple[pl.DataFrame, Dict]:
    """
    Regime-conditioned scaling: per-regime mean/std (global, not rolling).
    """
    out = f"{col}_reg"
    if regime_col not in df.columns:
        logging.warning("Regime column %s missing; skipping _reg for %s", regime_col, col)
        return df, {}

    if fit:
        stats = (
            df.group_by(regime_col)
            .agg(
                pl.col(col).mean().alias("mean"),
                pl.col(col).std().alias("std"),
            )
            .to_dict(as_series=False)
        )
        reg_params: Dict = {}
        for rid, m, s in zip(stats[regime_col], stats["mean"], stats["std"]):
            if m is None:
                continue
            if s is None:
                s = 0.0
            reg_params[rid] = {
                "mean": float(m),
                "std": float(max(s, 1e-6)),
            }
    else:
        reg_params = params or {}

    if not reg_params:
        return df, {}

    map_df = pl.DataFrame(
        {
            regime_col: list(reg_params.keys()),
            "reg_mean": [v["mean"] for v in reg_params.values()],
            "reg_std": [v["std"] for v in reg_params.values()],
        }
    )
    df = df.join(map_df, on=regime_col, how="left")
    expr = ((clean_expr(col) - pl.col("reg_mean")) / (pl.col("reg_std") + EPS)).clip(-CLIP_Z, CLIP_Z)
    df = df.with_columns(expr.cast(pl.Float32).alias(out)).drop(["reg_mean", "reg_std"])
    return df, reg_params


def collect_feature_list(df: pl.DataFrame, exclude: List[str]) -> List[str]:
    """
    Collect float-like feature columns to normalise, excluding:
    - ids / timestamps
    - already-normalised columns (suffix-based)
    """
    floats = {pl.Float32, pl.Float64}
    cols: List[str] = []
    for name, dtype in zip(df.columns, df.dtypes):
        if name in exclude:
            continue
        if any(name.endswith(sfx) for sfx in ("_z", "_mad", "_vol", "_reg", "_mm", "_log", "_drift")):
            continue
        if dtype in floats:
            cols.append(name)
    return cols


def _clean_scaled_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Post-pass cleanup:
    - Only touch derived *_z/_mad/_vol/_reg/_mm/_log/_drift columns.
    - Replace null/NaN/inf with 0.0.
    Raw columns remain exactly as in the input.
    """
    suffixes = ("_z", "_mad", "_vol", "_reg", "_mm", "_log", "_drift")
    float_types = (pl.Float32, pl.Float64)
    scaled_cols = [
        name
        for name, dtype in zip(df.columns, df.dtypes)
        if dtype in float_types and any(name.endswith(sfx) for sfx in suffixes)
    ]
    if not scaled_cols:
        return df

    return df.with_columns(
        [
            pl.when(pl.col(c).is_null() | pl.col(c).is_nan() | pl.col(c).is_infinite())
            .then(0.0)
            .otherwise(pl.col(c))
            .alias(c)
            for c in scaled_cols
        ]
    )


class GlobalScalingEngine:
    def __init__(
        self,
        regime_col: str = DEFAULT_REGIME_COL,
        vol_map: Optional[Dict[str, str]] = None,
    ):
        self.regime_col = regime_col
        self.vol_map = vol_map or DEFAULT_VOL_MAP

    def _prepare(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Ensure open_time_ms exists and sort deterministically,
        while preserving original row order via an index.
        """
        df = ensure_open_time_ms(df)
        df = df.with_columns(pl.arange(0, df.height, eager=True).alias("_order_idx"))
        df = df.sort("open_time_ms")
        return df

    def _restore_order(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.sort("_order_idx").drop("_order_idx")

    def fit(self, df: pl.DataFrame, features: List[str]) -> Tuple[pl.DataFrame, Dict]:
        df_sorted = self._prepare(df)

        params: Dict = {
            "meta": {
                "version": "1.0",
                "timestamp": datetime.utcnow().isoformat(),
                "z_window": Z_WINDOW,
                "mad_window": MAD_WINDOW,
                "drift_windows": {"long": DRIFT_LONG, "short": DRIFT_SHORT},
                "mm_quantiles": [MM_Q_LOW, MM_Q_HIGH],
                "regime_col": self.regime_col,
            },
            "features": {},
        }

        for col in features:
            if col not in df_sorted.columns:
                logging.warning("Skipping %s (missing)", col)
                continue

            logging.info("Normalising %s", col)

            # rolling z
            df_sorted = rolling_z(df_sorted, col)

            # MAD
            df_sorted = rolling_mad(df_sorted, col)

            # volatility scaling
            vol_ref = self.vol_map.get(col)
            if vol_ref:
                df_sorted = vol_scale(df_sorted, col, vol_ref)

            # regime-conditioned scaling
            df_sorted, reg_params = regime_scale(df_sorted, col, self.regime_col, fit=True)

            # limited min–max
            df_sorted, mm_params = minmax_scale(df_sorted, col, fit=True)

            # log scaling
            df_sorted = log_scale(df_sorted, col)

            # drift metric
            df_sorted = drift_metric(df_sorted, col)

            params["features"][col] = {
                "vol_ref": vol_ref,
                "mm": mm_params,
                "regime": reg_params,
            }

        df_sorted = _clean_scaled_columns(df_sorted)
        return self._restore_order(df_sorted), params

    def apply(self, df: pl.DataFrame, features: List[str], params: Dict) -> pl.DataFrame:
        df_sorted = self._prepare(df)

        for col in features:
            if col not in df_sorted.columns:
                logging.warning("Skipping %s (missing)", col)
                continue

            logging.info("Applying normalisation to %s", col)

            # rolling z / MAD are "online" style; we recompute on this batch
            df_sorted = rolling_z(df_sorted, col)
            df_sorted = rolling_mad(df_sorted, col)

            # volatility scaling
            vol_ref = params["features"].get(col, {}).get("vol_ref") or self.vol_map.get(col)
            if vol_ref:
                df_sorted = vol_scale(df_sorted, col, vol_ref)

            # regime-conditioned scaling using stored params
            reg_params = params["features"].get(col, {}).get("regime")
            df_sorted, _ = regime_scale(
                df_sorted,
                col,
                self.regime_col,
                params=reg_params,
                fit=False,
            )

            # min–max scaling using stored quantiles
            mm_params = params["features"].get(col, {}).get("mm")
            if mm_params:
                df_sorted, _ = minmax_scale(
                    df_sorted,
                    col,
                    params=mm_params,
                    fit=False,
                )

            # log scaling (parameter-free)
            df_sorted = log_scale(df_sorted, col)

            # drift metric (online, same windows)
            df_sorted = drift_metric(df_sorted, col)

        df_sorted = _clean_scaled_columns(df_sorted)
        return self._restore_order(df_sorted)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Global Feature Normalisation Layer (Global Scaling Engine)",
    )
    parser.add_argument("--mode", choices=["fit", "apply"], default="fit")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--regime-col", type=str, default=DEFAULT_REGIME_COL)
    parser.add_argument(
        "--features",
        nargs="+",
        help="Columns to normalise; default = all float columns",
    )
    parser.add_argument(
        "--vol-map-json",
        type=str,
        help="JSON string mapping feature->vol_ref, e.g. '{\"delta\":\"atr_14\"}'",
    )
    args = parser.parse_args()

    logging.info("Loading data from %s", args.input)
    df = pl.read_parquet(args.input)

    vol_map = DEFAULT_VOL_MAP.copy()
    if args.vol_map_json:
        try:
            vol_map.update(json.loads(args.vol_map_json))
        except json.JSONDecodeError:
            logging.error("Invalid vol-map-json; using defaults.")

    engine = GlobalScalingEngine(regime_col=args.regime_col, vol_map=vol_map)

    df = ensure_open_time_ms(df)
    exclude = ["open_time_ms", "open_time", args.regime_col]
    feature_list = args.features if args.features else collect_feature_list(df, exclude)
    logging.info("Features to normalise: %d", len(feature_list))

    if args.mode == "fit":
        out_df, params = engine.fit(df, feature_list)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.params.parent.mkdir(parents=True, exist_ok=True)
        out_df.write_parquet(args.output)
        with open(args.params, "w") as f:
            json.dump(params, f, indent=2)
        logging.info("Saved normalised features to %s", args.output)
        logging.info("Saved params to %s", args.params)
    else:
        with open(args.params) as f:
            params = json.load(f)
        out_df = engine.apply(df, feature_list, params)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_df.write_parquet(args.output)
        logging.info("Applied normalisation and saved to %s", args.output)


if __name__ == "__main__":
    main()

"""
Example usage:
cd PRE_PILLARFOUR
python global_scaling_engine.py \
    --mode fit \
    --input processed/canonical_with_all_engines.parquet \
    --output processed/normalised_features.parquet \
    --params processed/normalisation_params.json
"""
