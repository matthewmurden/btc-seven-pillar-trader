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
    """

    stable_path: Path
    fused_path: Path
    ml_path: Path
    df: Optional[pd.DataFrame] = None

    def preload_all_regimes(self) -> None:
        """
        Load stable, fused, and ML predictions; align on open_time_ms; store in self.df.
        """
        logging.info("Loading stable regimes from %s", self.stable_path)
        stable = pd.read_parquet(self.stable_path)
        logging.info("Loading fused regimes from %s", self.fused_path)
        fused = pd.read_parquet(self.fused_path)
        logging.info("Loading ML predictions from %s", self.ml_path)
        ml = pd.read_parquet(self.ml_path)

        # Ensure timestamps
        for frame_name, frame in [("stable", stable), ("fused", fused), ("ml", ml)]:
            if "open_time_ms" not in frame.columns:
                raise ValueError(f"{frame_name} parquet missing 'open_time_ms'")
            frame.sort_values("open_time_ms", inplace=True)
            frame.drop_duplicates(subset=["open_time_ms"], keep="last", inplace=True)
            frame.reset_index(drop=True, inplace=True)

        # Merge
        df = stable.merge(
            fused[["open_time_ms", "regime_label_fused"]],
            on="open_time_ms",
            how="left",
        )
        prob_cols = [c for c in ml.columns if c.startswith("prob_")]
        ml_cols = ["regime_pred", "regime_confidence"] + prob_cols
        df = df.merge(
            ml[["open_time_ms"] + ml_cols],
            on="open_time_ms",
            how="left",
        )

        df.rename(
            columns={
                "regime_confidence": "ml_confidence",
                "regime_pred": "regime_pred_ml",
            },
            inplace=True,
        )

        self.df = df
        self.validate_regime_consistency()

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

    def get_regime_for_timestamp(self, t: TimestampLike) -> Dict[str, Union[int, float, str]]:
        row = self._locate_row(t)
        return {
            "regime_label_stable": row["regime_label_stable"],
            "regime_label_fused": row["regime_label_fused"],
            "regime_pred_ml": row["regime_pred_ml"],
        }

    def get_regime_context(self, t: TimestampLike) -> Dict[str, Union[int, float, str]]:
        row = self._locate_row(t)
        return {
            "regime_label_stable": row["regime_label_stable"],
            "regime_label_prev": row.get("regime_label_prev", np.nan),
            "hold_length": row.get("hold_length", np.nan),
            "smoothing_mode": row.get("smoothing_mode", ""),
        }

    def get_regime_probabilities(self, t: TimestampLike) -> Dict[str, Union[float, Dict[str, float]]]:
        row = self._locate_row(t)
        prob_cols = [c for c in self.df.columns if c.startswith("prob_")]
        probs = {c: row[c] for c in prob_cols}
        return {"ml_confidence": row["ml_confidence"], "probabilities": probs}

    def validate_regime_consistency(self) -> None:
        if self.df is None:
            raise RuntimeError("No data loaded for validation.")

        df = self.df
        required_cols = [
            "open_time_ms",
            "regime_label_stable",
            "regime_label_fused",
            "regime_pred_ml",
            "ml_confidence",
        ]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        if df["open_time_ms"].isna().any():
            raise ValueError("NaNs found in open_time_ms")
        if not df["open_time_ms"].is_monotonic_increasing:
            raise ValueError("open_time_ms not sorted")

        if df["regime_label_stable"].isna().any():
            raise ValueError("NaNs in regime_label_stable")
        if not np.issubdtype(df["regime_label_stable"].dtype, np.integer):
            raise ValueError("regime_label_stable must be integer type")

        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        if prob_cols:
            probs_sum = df[prob_cols].sum(axis=1)
            if not np.allclose(probs_sum.values, 1.0, atol=1e-2):
                raise ValueError("Probability rows do not sum to ~1.0")
        else:
            raise ValueError("No probability columns found (prob_*)")

        critical_cols = required_cols + prob_cols + ["regime_label_prev", "hold_length", "smoothing_mode"]
        if df[critical_cols].isna().any().any():
            raise ValueError("NaNs found in critical fields")


def export_final_regime_timeline(api: RegimeMasterAPI, out_path: Path) -> pd.DataFrame:
    """
    Export merged regime timeline with required columns. Does not write to disk; caller decides.
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
    # Return dataframe; caller can save to disk
    return final_df
