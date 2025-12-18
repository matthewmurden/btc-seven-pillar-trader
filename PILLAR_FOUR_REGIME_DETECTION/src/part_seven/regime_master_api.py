from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeMasterAPI:
    """
    Lightweight RegimeMasterAPI for loading and querying regime timelines.
    This is used by live integration to export a unified regime_timeline.parquet.
    """

    def __init__(
        self,
        *,
        stable_path: Path,
        fused_path: Path,
        ml_path: Path,
        det_path: Path,
    ) -> None:
        self.stable_path = Path(stable_path)
        self.fused_path = Path(fused_path)
        self.ml_path = Path(ml_path)
        self.det_path = Path(det_path)

        self._stable: Optional[pd.DataFrame] = None
        self._fused: Optional[pd.DataFrame] = None
        self._ml: Optional[pd.DataFrame] = None
        self._det: Optional[pd.DataFrame] = None

    def preload_all_regimes(self) -> None:
        self._stable = _load_regime_df(self.stable_path)
        self._fused = _load_regime_df(self.fused_path)
        self._ml = _load_regime_df(self.ml_path)
        self._det = _load_regime_df(self.det_path)
        logger.info(
            "RegimeMasterAPI: loaded stable=%s fused=%s ml=%s det=%s",
            _shape(self._stable),
            _shape(self._fused),
            _shape(self._ml),
            _shape(self._det),
        )

    def get_regime_for_timestamp(self, ts_ms: int) -> Optional[str]:
        df = self._stable
        if df is None or df.empty:
            return None
        row = _row_for_ts(df, ts_ms)
        if row is None:
            return None
        return str(row.get("regime_label_stable", row.get("regime_final", "")))

    def get_regime_probabilities(self, ts_ms: int) -> Optional[Dict[str, Any]]:
        df = self._ml
        if df is None or df.empty:
            return None
        row = _row_for_ts(df, ts_ms)
        if row is None:
            return None
        return row.to_dict()

    def get_regime_context(self, ts_ms: int) -> Optional[Dict[str, Any]]:
        timeline = export_final_regime_timeline(self)
        if timeline.empty:
            return None
        row = _row_for_ts(timeline, ts_ms)
        if row is None:
            return None
        return row.to_dict()

    def validate_regime_consistency(self) -> bool:
        df = self._stable
        if df is None or df.empty:
            logger.warning("RegimeMasterAPI: no stable regimes loaded.")
            return False
        if "open_time_ms" in df.columns:
            if df["open_time_ms"].isna().any():
                logger.warning("RegimeMasterAPI: stable regimes missing open_time_ms values.")
                return False
        return True


def export_final_regime_timeline(api: RegimeMasterAPI) -> pd.DataFrame:
    """
    Export a unified regime timeline by joining stable + fused + ML + deterministic.
    Ensures a `regime_final` column (stable label) is present.
    """
    stable = api._stable if api._stable is not None else pd.DataFrame()
    fused = api._fused
    ml = api._ml
    det = api._det

    if stable.empty:
        return stable

    out = stable.copy()
    if "regime_label_stable" in out.columns and "regime_final" not in out.columns:
        out["regime_final"] = out["regime_label_stable"]

    out = _merge_regimes(out, fused)
    out = _merge_regimes(out, ml)
    out = _merge_regimes(out, det)

    if "open_time_ms" in out.columns:
        out = out.drop_duplicates("open_time_ms").sort_values("open_time_ms")

    return out


def _load_regime_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("Regime file missing: %s", path)
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    if "open_time_ms" not in df.columns and "open_time" in df.columns:
        df["open_time_ms"] = pd.to_datetime(df["open_time"], utc=True).astype("int64") // 1_000_000
    return df


def _merge_regimes(base: pd.DataFrame, extra: Optional[pd.DataFrame]) -> pd.DataFrame:
    if extra is None or extra.empty:
        return base
    if "open_time_ms" in base.columns and "open_time_ms" in extra.columns:
        merged = base.merge(extra, on="open_time_ms", how="left")
        return merged
    if len(base) == len(extra):
        base = base.reset_index(drop=True)
        extra = extra.reset_index(drop=True)
        dup_cols = [c for c in extra.columns if c in base.columns]
        extra = extra.drop(columns=dup_cols, errors="ignore")
        return pd.concat([base, extra], axis=1)
    return base


def _row_for_ts(df: pd.DataFrame, ts_ms: int) -> Optional[pd.Series]:
    if df.empty:
        return None
    if "open_time_ms" not in df.columns:
        return df.iloc[-1]
    arr = df["open_time_ms"].to_numpy()
    idx = int(np.searchsorted(arr, ts_ms, side="right") - 1)
    if idx < 0:
        return None
    return df.iloc[idx]


def _shape(df: Optional[pd.DataFrame]) -> str:
    if df is None:
        return "None"
    return f"{df.shape}"
