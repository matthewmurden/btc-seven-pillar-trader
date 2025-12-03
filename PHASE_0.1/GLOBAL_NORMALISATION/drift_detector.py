import logging
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


class DriftDetector:
    def __init__(
        self,
        window: int = 1000,
        threshold: float = 5.0,
        output_path: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.window = window
        self.threshold = threshold
        self.output_path = output_path
        self.logger = logger or logging.getLogger("phase0.drift")
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _feature_drift(self, series: pd.Series) -> pd.Series:
        mean_roll = series.rolling(self.window, min_periods=max(10, self.window // 10)).mean()
        mean_of_mean = mean_roll.rolling(self.window, min_periods=max(10, self.window // 10)).mean()
        std_of_mean = mean_roll.rolling(self.window, min_periods=max(10, self.window // 10)).std(ddof=0)
        std_of_mean = std_of_mean.replace(0, np.nan)
        zscore = (mean_roll - mean_of_mean) / std_of_mean
        return zscore

    def run(self, df: pd.DataFrame, features: Iterable[str]) -> None:
        start = time.time()
        features = list(features)
        total = len(features)
        self.logger.info(
            "Starting drift detection across %d feature(s) with window=%d threshold=%.2f",
            total,
            self.window,
            self.threshold,
        )
        last_log = start
        reports = []
        for idx, feat in enumerate(features, start=1):
            if feat not in df.columns:
                continue
            z = self._feature_drift(df[feat])
            breaches = z.abs() > self.threshold
            if breaches.any():
                count = int(breaches.sum())
                max_z = float(z.abs().max(skipna=True) or 0.0)
                first_idx = breaches[breaches].index[0]
                reports.append(f"{feat}: breaches={count}, max|z|={max_z:.2f}, first_index={first_idx}")
            now = time.time()
            if idx == total or now - last_log >= 5.0:
                pct = (idx / total * 100) if total else 100.0
                self.logger.info("Drift progress: %d/%d (%.1f%%) elapsed=%.1fs", idx, total, pct, now - start)
                last_log = now

        if not reports:
            msg = "No drift detected."
            self.logger.info(msg)
            if self.output_path:
                self.output_path.write_text(msg)
        else:
            content = "\n".join(reports)
            self.logger.warning("Drift detected for %d feature(s)", len(reports))
            if self.output_path:
                self.output_path.write_text(content)
            else:
                self.logger.warning(content)

        self.logger.info("Drift detection finished in %.2fs", time.time() - start)


__all__ = ["DriftDetector"]
