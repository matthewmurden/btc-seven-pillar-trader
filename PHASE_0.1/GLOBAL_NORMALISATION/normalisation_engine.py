import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd

from scalers import (
    MADScaler,
    LogScaler,
    MinMaxScaler,
    RegimeSplitScaler,
    VolScaler,
    ZScoreScaler,
)


DEFAULT_ZSCORE_FEATURES: List[str] = [
    "ret",
    "session_vol_bias",
    "session_vol_slope",
    "htf_vol_slope",
    "rv_5m",
    "rv_15m",
    "rv_1h",
]

DEFAULT_MAD_FEATURES: List[str] = [
    "wick_body_ratio",
    "micro_volatility",
    "micro_return",
    "entropy",
    "order_imbalance",
    "micro_delta",
    "delta_persistence",
    "micro_clustering",
    "alternation_index",
    "no_trend_stat",
]

DEFAULT_VOL_SCALED: Dict[str, str] = {
    "delta": "atr14",
    "micro_return": "atr14",
}

DEFAULT_MINMAX_FEATURES: List[str] = [
    "compression_flag",
    "expansion_flag",
    "session_asia",
    "session_london",
    "session_ny",
]

DEFAULT_LOG_FEATURES: List[str] = [
    "volume",
    "quote_volume",
    "taker_buy_base",
    "taker_buy_quote",
    "agg_base_volume",
    "agg_quote_volume",
]

DEFAULT_REGIME_SPLIT_FEATURES: List[str] = [
    "delta",
    "micro_return",
]


@dataclass
class EngineConfig:
    zscore_window: int = 1000
    mad_window: int = 1000
    minmax_window: int = 500
    regime_window: int = 500
    zscore_clip: float = 5.0
    mad_clip: float = 6.0
    regime_clip: float = 5.0
    minmax_clip_lower: float = 0.01
    minmax_clip_upper: float = 0.99
    epsilon: float = 1e-9
    regime_col: str = "vol_regime"
    progress_interval_seconds: float = 5.0


class NormalisationEngine:
    def __init__(self, config: Optional[EngineConfig] = None, logger: Optional[logging.Logger] = None):
        self.config = config or EngineConfig()
        self.logger = logger or logging.getLogger("phase0.normalisation")
        self.feature_map: List[Dict[str, Any]] = []
        self.scalers: Dict[str, object] = {}
        self.params: Dict[str, Dict] = {}

    def _maybe_log_progress(self, phase: str, idx: int, total: int, start_time: float, last_log: float) -> float:
        now = time.time()
        if idx == 1 or idx == total or now - last_log >= self.config.progress_interval_seconds:
            pct = (idx / total * 100) if total else 100.0
            self.logger.info("%s progress: %d/%d (%.1f%%) elapsed=%.1fs", phase, idx, total, pct, now - start_time)
            return now
        return last_log

    def _log_df_overview(self, df: pd.DataFrame, label: str) -> None:
        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        self.logger.info("%s shape=%s cols=%d approx_mem=%.2fMB", label, df.shape, len(df.columns), mem_mb)

    def _resolve_features(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        available = set(df.columns)
        used: Set[str] = set()
        feature_map: List[Dict[str, Any]] = []

        def register(features: Iterable[str], scaler_type: str, extra: Optional[Dict] = None, allow_duplicate: bool = False):
            for feat in features:
                if feat not in available or (feat in used and not allow_duplicate):
                    continue
                entry = {"feature": feat, "type": scaler_type}
                if extra:
                    entry.update(extra)
                feature_map.append(entry)
                used.add(feat)

        register(DEFAULT_LOG_FEATURES, "log")
        register(DEFAULT_VOL_SCALED.keys(), "vol", extra={"denominator": None})
        # set proper denominators
        for feat, denom in DEFAULT_VOL_SCALED.items():
            for entry in feature_map:
                if entry["feature"] == feat and entry["type"] == "vol":
                    entry["denominator"] = denom

        register(DEFAULT_MAD_FEATURES, "mad")
        register(DEFAULT_ZSCORE_FEATURES, "zscore")
        register(DEFAULT_MINMAX_FEATURES, "minmax")
        register(DEFAULT_REGIME_SPLIT_FEATURES, "regime", allow_duplicate=True)

        self.logger.info("Feature map built with %d entries", len(feature_map))
        return feature_map

    def fit(self, df: pd.DataFrame) -> "NormalisationEngine":
        start = time.time()
        df = df.copy()
        df = df.sort_values("open_time_ms") if "open_time_ms" in df.columns else df
        self._log_df_overview(df, "Fit input")
        self.feature_map = self._resolve_features(df)
        self.scalers = {}
        self.params = {}

        total = len(self.feature_map)
        last_log = start
        if total == 0:
            self.logger.warning("No features available for normalisation.")
            return self

        for idx, cfg in enumerate(self.feature_map, start=1):
            feat = cfg["feature"]
            ftype = cfg["type"]
            key = f"{feat}|{ftype}"
            if ftype == "zscore":
                scaler = ZScoreScaler(window=self.config.zscore_window, clip=self.config.zscore_clip)
                scaler.fit(df[feat])
            elif ftype == "mad":
                scaler = MADScaler(window=self.config.mad_window, clip=self.config.mad_clip)
                scaler.fit(df[feat])
            elif ftype == "vol":
                denom_col = cfg.get("denominator")
                if denom_col not in df.columns:
                    raise ValueError(f"Denominator {denom_col} missing for vol scaling of {feat}")
                scaler = VolScaler(denominator_col=denom_col, epsilon=self.config.epsilon)
                scaler.fit(df[feat], df[denom_col])
            elif ftype == "minmax":
                scaler = MinMaxScaler(
                    window=self.config.minmax_window,
                    clip_lower=self.config.minmax_clip_lower,
                    clip_upper=self.config.minmax_clip_upper,
                )
                scaler.fit(df[feat])
            elif ftype == "log":
                scaler = LogScaler(signed=True)
                scaler.fit(df[feat])
            elif ftype == "regime":
                regime_col = self.config.regime_col
                if regime_col not in df.columns:
                    raise ValueError(f"Regime column {regime_col} missing for regime scaling of {feat}")
                scaler = RegimeSplitScaler(
                    base_scaler_cls=ZScoreScaler,
                    window=self.config.regime_window,
                    clip=self.config.regime_clip,
                )
                scaler.fit(df[feat], df[regime_col])
            else:
                raise ValueError(f"Unknown scaler type {ftype}")

            self.scalers[key] = scaler
            self.params[key] = scaler.to_params()
            last_log = self._maybe_log_progress("Fit", idx, total, start, last_log)

        self.logger.info("Fitted %d scalers in %.2fs", len(self.scalers), time.time() - start)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_map or not self.scalers:
            raise RuntimeError("NormalisationEngine must be fitted before transform.")
        df = df.copy()
        df = df.sort_values("open_time_ms") if "open_time_ms" in df.columns else df
        self._log_df_overview(df, "Transform input")

        total = len(self.feature_map)
        start = time.time()
        last_log = start

        for idx, cfg in enumerate(self.feature_map, start=1):
            feat = cfg["feature"]
            if feat not in df.columns:
                continue
            key = f"{feat}|{cfg['type']}"
            scaler = self.scalers[key]
            ftype = cfg["type"]
            if ftype == "zscore":
                df[f"{feat}_zscore"] = scaler.transform(df[feat])
            elif ftype == "mad":
                df[f"{feat}_mad"] = scaler.transform(df[feat])
            elif ftype == "vol":
                denom_col = cfg["denominator"]
                df[f"{feat}_volnorm"] = scaler.transform(df[feat], df[denom_col])
            elif ftype == "minmax":
                df[f"{feat}_minmax"] = scaler.transform(df[feat])
            elif ftype == "log":
                df[f"{feat}_log"] = scaler.transform(df[feat])
            elif ftype == "regime":
                regime_col = self.config.regime_col
                df[f"{feat}_regime_norm"] = scaler.transform(df[feat], df[regime_col])
            last_log = self._maybe_log_progress("Transform", idx, total, start, last_log)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def to_params(self) -> Dict:
        return {
            "meta": {
                "config": asdict(self.config),
                "feature_map": self.feature_map,
            },
            "scalers": {key: scaler.to_params() for key, scaler in self.scalers.items()},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_params(), f, indent=2)
        self.logger.info("Saved normalisation params to %s", path)

    @classmethod
    def load(cls, path: Path, logger: Optional[logging.Logger] = None) -> "NormalisationEngine":
        with open(path, "r") as f:
            params = json.load(f)
        config = EngineConfig(**params["meta"]["config"])
        engine = cls(config=config, logger=logger)
        engine.feature_map = params["meta"]["feature_map"]
        scalers = {}
        for key, s_params in params["scalers"].items():
            stype = s_params.get("type")
            if stype == "zscore":
                scalers[key] = ZScoreScaler.from_params(s_params)
            elif stype == "mad":
                scalers[key] = MADScaler.from_params(s_params)
            elif stype == "vol":
                scalers[key] = VolScaler.from_params(s_params)
            elif stype == "minmax":
                scalers[key] = MinMaxScaler.from_params(s_params)
            elif stype == "log":
                scalers[key] = LogScaler.from_params(s_params)
            elif stype == "regime_split":
                scalers[key] = RegimeSplitScaler.from_params(s_params)
            else:
                raise ValueError(f"Unknown scaler type in params: {stype}")
        engine.scalers = scalers
        engine.params = params["scalers"]
        return engine


def _test_zscore() -> None:
    s = pd.Series(np.arange(6, dtype=float))
    scaler = ZScoreScaler(window=3, clip=5)
    out = scaler.fit_transform(s)
    expected_mean = np.mean([0, 1, 2])
    expected_std = np.std([0, 1, 2])
    expected = (3 - expected_mean) / expected_std
    assert np.isclose(out.iloc[3], expected)


def _test_minmax() -> None:
    s = pd.Series([0, 1, 2, 3, 4], dtype=float)
    scaler = MinMaxScaler(window=2)
    out = scaler.fit_transform(s)
    assert (out <= 0.99).all() and (out >= 0.01).all()


def _test_regime() -> None:
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    regimes = pd.Series(["compression", "normal", "compression", "expansion", "normal"])
    scaler = RegimeSplitScaler(window=2, clip=5)
    out = scaler.fit_transform(s, regimes)
    assert len(out) == len(s)


def _test_leak_free() -> None:
    s = pd.Series([10, 20, 30, 40], dtype=float)
    scaler = ZScoreScaler(window=2, clip=10)
    out = scaler.fit_transform(s)
    # For index 2, stats should only use first two points (10,20)
    expected_mean = np.mean([10, 20])
    expected_std = np.std([10, 20])
    expected = (30 - expected_mean) / expected_std
    assert np.isclose(out.iloc[2], expected)


if __name__ == "__main__":
    _test_zscore()
    _test_minmax()
    _test_regime()
    _test_leak_free()
    print("Basic tests passed.")


__all__ = ["NormalisationEngine", "EngineConfig"]
