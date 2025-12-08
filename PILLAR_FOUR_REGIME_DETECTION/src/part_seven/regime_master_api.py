import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


TimestampLike = Union[int, float, datetime, np.datetime64, pd.Timestamp]


@dataclass
class RegimeMasterAPI:
    """
    Unified interface to query stable, fused, and ML regime information.

    Expected inputs (all length-aligned, same ordering):
      - stable_path: Part 5 → regime_labels_stable.parquet
      - fused_path:  Part 4 → regime_labels_fused.parquet
      - ml_path:     Part 3 → regime_ml_predictions.parquet
      - det_path:    Part 2 → deterministic_regime_engines.parquet (provides open_time_ms)

    After preload_all_regimes(), self.df contains a merged timeline with:
      - open_time_ms
      - stable regimes + smoothing context
      - fused regimes + fusion metadata
      - ML predictions + probabilities
    """

    stable_path: Path
    fused_path: Path
    ml_path: Path
    det_path: Path
    df: Optional[pd.DataFrame] = None

    # ---------------------------------------------------------------------
    # LOADING / MERGING
    # ---------------------------------------------------------------------
    def preload_all_regimes(self) -> None:
        """
        Load stable, fused, ML, and deterministic outputs; align on open_time_ms; store in self.df.
        """
        logging.info("Loading stable regimes from %s", self.stable_path)
        stable = pd.read_parquet(self.stable_path)

        logging.info("Loading fused regimes from %s", self.fused_path)
        fused = pd.read_parquet(self.fused_path)

        logging.info("Loading ML predictions from %s", self.ml_path)
        ml = pd.read_parquet(self.ml_path)

        logging.info("Loading deterministic engines from %s (for timestamps)", self.det_path)
        det = pd.read_parquet(self.det_path)

        # Basic length sanity
        n = len(det)
        if not (len(stable) == len(fused) == len(ml) == n):
            raise ValueError(
                f"Length mismatch between det ({len(det)}), "
                f"stable ({len(stable)}), fused ({len(fused)}), ml ({len(ml)})"
            )

        # Ensure deterministic has open_time_ms and sorted
        if "open_time_ms" not in det.columns:
            raise ValueError("deterministic_regime_engines parquet missing 'open_time_ms'")

        det = det.sort_values("open_time_ms").reset_index(drop=True)

        # Index-align the others to det
        stable = stable.reset_index(drop=True)
        fused = fused.reset_index(drop=True)
        ml = ml.reset_index(drop=True)

        # Start with canonical time axis
        df = det[["open_time_ms"]].copy()

        # Attach stable layer (Part 5)
        # regime_labels_stable.parquet: regime_label_stable, regime_label_prev, hold_length, smoothing_mode
        for col in ["regime_label_stable", "regime_label_prev", "hold_length", "smoothing_mode"]:
            if col not in stable.columns:
                raise ValueError(f"Stable parquet missing required column '{col}'")
        df = pd.concat(
            [
                df,
                stable[["regime_label_stable", "regime_label_prev", "hold_length", "smoothing_mode"]],
            ],
            axis=1,
        )

        # Attach fused layer (Part 4)
        # regime_labels_fused.parquet: regime_label_fused, ml_confidence, fusion_mode, regime_pred, deterministic_label
        for col in ["regime_label_fused", "ml_confidence", "fusion_mode", "regime_pred", "deterministic_label"]:
            if col not in fused.columns:
                raise ValueError(f"Fused parquet missing required column '{col}'")

        df = pd.concat(
            [
                df,
                fused[["regime_label_fused", "fusion_mode", "deterministic_label"]],
            ],
            axis=1,
        )

        # Attach ML predictions (Part 3)
        # regime_ml_predictions.parquet: regime_pred, regime_confidence, prob_0..prob_7
        if "regime_pred" not in ml.columns or "regime_confidence" not in ml.columns:
            raise ValueError("ML parquet missing 'regime_pred' or 'regime_confidence'")

        prob_cols = [c for c in ml.columns if c.startswith("prob_")]
        if not prob_cols:
            raise ValueError("ML parquet has no probability columns (prob_*)")

        df["regime_pred_ml"] = ml["regime_pred"].to_numpy()
        df["ml_confidence"] = ml["regime_confidence"].to_numpy()
        for c in prob_cols:
            df[c] = ml[c].to_numpy()

        self.df = df
        self.validate_regime_consistency()

    # ---------------------------------------------------------------------
    # INTERNAL: locate row by timestamp
    # ---------------------------------------------------------------------
    def _locate_row(self, t: TimestampLike) -> pd.Series:
        if self.df is None:
            raise RuntimeError("Regime data not preloaded. Call preload_all_regimes() first.")

        if isinstance(t, (datetime, np.datetime64, pd.Timestamp)):
            ts = int(pd.to_datetime(t).timestamp() * 1000)
        else:
            ts = int(t)

        idx = self.df["open_time_ms"].searchsorted(ts, side="right") - 1
        if idx < 0:
            raise IndexError("Timestamp precedes available regime data.")
        return self.df.iloc[idx]

    # ---------------------------------------------------------------------
    # PUBLIC QUERIES
    # ---------------------------------------------------------------------
    def get_regime_for_timestamp(self, t: TimestampLike) -> Dict[str, Union[int, float, str]]:
        """
        Core regime snapshot for a given timestamp.

        Returns
        -------
        {
          "regime_label_stable": str or numeric (final smoothed regime),
          "regime_label_fused":  str or numeric (pre-stability fused label),
          "regime_pred_ml":      int (0..7, raw ML class),
        }
        """
        row = self._locate_row(t)
        return {
            "regime_label_stable": row["regime_label_stable"],
            "regime_label_fused": row["regime_label_fused"],
            "regime_pred_ml": row["regime_pred_ml"],
        }

    def get_regime_context(self, t: TimestampLike) -> Dict[str, Union[int, float, str]]:
        """
        Return stability / smoothing context around the regime at time t.
        """
        row = self._locate_row(t)
        return {
            "regime_label_stable": row["regime_label_stable"],
            "regime_label_prev": row.get("regime_label_prev", np.nan),
            "hold_length": row.get("hold_length", np.nan),
            "smoothing_mode": row.get("smoothing_mode", ""),
        }

    def get_regime_probabilities(self, t: TimestampLike) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Return ML confidence and per-class probabilities at time t.
        """
        row = self._locate_row(t)
        prob_cols = [c for c in self.df.columns if c.startswith("prob_")]
        probs = {c: float(row[c]) for c in prob_cols}
        return {"ml_confidence": float(row["ml_confidence"]), "probabilities": probs}

    # ---------------------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------------------
    def validate_regime_consistency(self) -> None:
        """
        Structural validation of the merged regime timeline.
        """
        if self.df is None:
            raise RuntimeError("No data loaded for validation.")

        df = self.df

        required_cols = [
            "open_time_ms",
            "regime_label_stable",
            "regime_label_fused",
            "regime_pred_ml",
            "ml_confidence",
            "regime_label_prev",
            "hold_length",
            "smoothing_mode",
        ]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Time sanity
        if df["open_time_ms"].isna().any():
            raise ValueError("NaNs found in open_time_ms")
        if not df["open_time_ms"].is_monotonic_increasing:
            raise ValueError("open_time_ms not sorted")

        # Stable/fused labels: only require non-null, type can be string or numeric
        if df["regime_label_stable"].isna().any():
            raise ValueError("NaNs in regime_label_stable")
        if df["regime_label_fused"].isna().any():
            raise ValueError("NaNs in regime_label_fused")

        # ML sanity
        if df["ml_confidence"].isna().any():
            raise ValueError("NaNs in ml_confidence")

        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        if not prob_cols:
            raise ValueError("No probability columns found (prob_*)")

        probs_sum = df[prob_cols].sum(axis=1)
        if not np.allclose(probs_sum.values, 1.0, atol=1e-3):
            raise ValueError("Probability rows do not sum to ~1.0")

        # Check for NaNs in critical fields
        critical_cols = required_cols + prob_cols
        if df[critical_cols].isna().any().any():
            raise ValueError("NaNs found in critical fields")

        logging.info("RegimeMasterAPI: regime timeline passed consistency checks.")


def export_final_regime_timeline(api: RegimeMasterAPI) -> pd.DataFrame:
    """
    Export merged regime timeline with required columns.

    Does NOT write to disk; caller decides where/how to persist.
    """
    if api.df is None:
        raise RuntimeError("API not preloaded; call preload_all_regimes() first.")

    prob_cols = [c for c in api.df.columns if c.startswith("prob_")]
    required = [
        "open_time_ms",
        "regime_label_stable",
        "regime_label_fused",
        "regime_label_prev",
        "hold_length",
        "smoothing_mode",
        "regime_pred_ml",
        "ml_confidence",
    ] + prob_cols

    missing = [c for c in required if c not in api.df.columns]
    if missing:
        raise ValueError(f"Missing required columns for export: {missing}")

    final_df = api.df[required].copy()
    logging.info("Prepared final regime timeline with %d rows", len(final_df))
    return final_df

# from pathlib import Path
# from regime_master_api import RegimeMasterAPI, export_final_regime_timeline

# BASE = Path("PILLAR_FOUR_REGIME_DETECTION/data/processed")

# api = RegimeMasterAPI(
#     stable_path=BASE / "part_five" / "regime_labels_stable.parquet",
#     fused_path=BASE / "part_four" / "regime_labels_fused.parquet",
#     ml_path=BASE / "part_three" / "regime_ml_predictions.parquet",
#     det_path=BASE / "part_two" / "deterministic_regime_engines.parquet",
# )

# api.preload_all_regimes()

# # Example: get regime at some timestamp
# ctx = api.get_regime_for_timestamp(1667260800000)
# print(ctx)

# # Export final regime timeline
# final_df = export_final_regime_timeline(api)
# final_df.to_parquet(BASE / "regime_final.parquet", index=False)