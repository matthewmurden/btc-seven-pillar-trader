import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure imports work when run as a script
_ROOT = Path(__file__).resolve().parent
_PARENT = _ROOT.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from feature_normalisation.scalers.zscore_scaler import ZScoreScaler  # type: ignore
from feature_normalisation.scalers.mad_scaler import MADScaler  # type: ignore
from feature_normalisation.scalers.volatility_scaler import VolatilityScaler  # type: ignore
from feature_normalisation.scalers.minmax_scaler import MinMaxScaler  # type: ignore
from feature_normalisation.scalers.log_scaler import LogScaler  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class NormalisationEngine:
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "processed")
    params_path: Path = field(init=False)
    normalised_path: Path = field(init=False)
    scaler_map: Dict[str, str] = field(default_factory=dict)
    zscore_window: int = 1000
    zscore_clip: float = 5.0

    def __post_init__(self):
        self.params_path = self.output_dir / "normalisation_params.json"
        self.normalised_path = self.output_dir / "normalised_features.parquet"

    def _prepare_map(self, df: pd.DataFrame) -> Dict[str, str]:
        if self.scaler_map:
            return self.scaler_map
        # default: zscore for all numeric
        return {col: "zscore" for col in df.select_dtypes(include=[np.number]).columns}

    def fit_transform(self, df: pd.DataFrame, vol_series: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Training mode: fit scalers, apply, save params and normalised parquet.
        """
        scaler_map = self._prepare_map(df)

        params: Dict[str, Dict] = {"metadata": {"trained_at": datetime.utcnow().isoformat()}}
        out = df.copy()

        # group columns by scaler type
        groups: Dict[str, List[str]] = {}
        for col, scaler_name in scaler_map.items():
            groups.setdefault(scaler_name, []).append(col)

        # Z-score
        if "zscore" in groups:
            zs = ZScoreScaler(window=self.zscore_window, clip=self.zscore_clip)
            params["zscore"] = zs.fit(df[groups["zscore"]])
            out[[f"{c}_norm" for c in groups["zscore"]]] = zs.transform(df[groups["zscore"]], params["zscore"])

        # MAD
        if "mad" in groups:
            mad = MADScaler()
            params["mad"] = mad.fit(df[groups["mad"]])
            out[[f"{c}_norm" for c in groups["mad"]]] = mad.transform(df[groups["mad"]], params["mad"])

        # Volatility
        if "volatility" in groups:
            if vol_series is None:
                raise ValueError("vol_series required for volatility scaling")
            vol = VolatilityScaler()
            out[[f"{c}_norm" for c in groups["volatility"]]] = vol.transform(df[groups["volatility"]], vol_series)
            params["volatility"] = {"features": groups["volatility"]}

        # MinMax
        if "minmax" in groups:
            mm = MinMaxScaler()
            params["minmax"] = mm.fit(df[groups["minmax"]])
            out[[f"{c}_norm" for c in groups["minmax"]]] = mm.transform(df[groups["minmax"]], params["minmax"])

        # Log
        if "log" in groups:
            log_s = LogScaler()
            out[[f"{c}_norm" for c in groups["log"]]] = log_s.transform(df[groups["log"]])
            params["log"] = {"features": groups["log"]}

        self._save_params(params, scaler_map)
        self._write_parquet(out)
        return out

    def transform(self, df: pd.DataFrame, vol_series: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Inference mode: load params and apply scaling deterministically.
        """
        params, scaler_map = self._load_params()
        out = df.copy()
        groups: Dict[str, List[str]] = {}
        for col, scaler_name in scaler_map.items():
            groups.setdefault(scaler_name, []).append(col)

        if "zscore" in groups:
            zs = ZScoreScaler(window=self.zscore_window, clip=self.zscore_clip)
            out[[f"{c}_norm" for c in groups["zscore"]]] = zs.transform(df[groups["zscore"]], params.get("zscore"))

        if "mad" in groups:
            mad = MADScaler()
            out[[f"{c}_norm" for c in groups["mad"]]] = mad.transform(df[groups["mad"]], params.get("mad"))

        if "volatility" in groups:
            if vol_series is None:
                raise ValueError("vol_series required for volatility scaling")
            vol = VolatilityScaler()
            out[[f"{c}_norm" for c in groups["volatility"]]] = vol.transform(df[groups["volatility"]], vol_series)

        if "minmax" in groups:
            mm = MinMaxScaler()
            out[[f"{c}_norm" for c in groups["minmax"]]] = mm.transform(df[groups["minmax"]], params.get("minmax"))

        if "log" in groups:
            log_s = LogScaler()
            out[[f"{c}_norm" for c in groups["log"]]] = log_s.transform(df[groups["log"]])

        return out

    def detect_drift(self, df: pd.DataFrame) -> bool:
        params, scaler_map = self._load_params()
        drift = False
        # simple drift: compare mean/std to stored zscore params
        if "zscore" in scaler_map.values() and "zscore" in params:
            zparams = params["zscore"]
            for col, p in zparams.items():
                mean_ref = p.get("last_mean", 0.0)
                std_ref = p.get("last_std", 1e-9)
                mean_cur = df[col].mean()
                std_cur = df[col].std(ddof=0)
                if abs(mean_cur - mean_ref) > 3 * std_ref or std_cur > 3 * std_ref or std_cur < std_ref / 3:
                    drift = True
                    logger.warning("Drift detected on column %s", col)
        return drift

    def _save_params(self, params: Dict, scaler_map: Dict[str, str]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {"scaler_map": scaler_map, "params": params}
        with self.params_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved normalisation params to %s", self.params_path)

    def _load_params(self) -> Tuple[Dict, Dict]:
        if not self.params_path.exists():
            raise FileNotFoundError(f"Params not found at {self.params_path}")
        with self.params_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("params", {}), payload.get("scaler_map", {})

    def _write_parquet(self, df: pd.DataFrame) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.normalised_path, index=False)
        logger.info("Saved normalised features to %s", self.normalised_path)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Feature Normalisation Engine (Phase 0 Global Foundations)")
    parser.add_argument("--mode", choices=["train", "infer"], required=True, help="train: fit+transform, infer: transform only")
    parser.add_argument("--input", required=True, help="Path to input parquet/CSV containing raw features")
    parser.add_argument("--vol-col", default=None, help="Column name to use as volatility series for volatility scaling")
    parser.add_argument("--output-dir", default=None, help="Override default output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.suffix.lower() == ".csv":
        df_in = pd.read_csv(input_path)
    else:
        df_in = pd.read_parquet(input_path)

    vol_series = None
    if args.vol_col:
        if args.vol_col not in df_in.columns:
            raise ValueError(f"vol-col {args.vol_col} not found in input columns")
        vol_series = df_in[args.vol_col]

    engine = NormalisationEngine()
    if args.output_dir:
        engine.output_dir = Path(args.output_dir)
        engine.params_path = engine.output_dir / "normalisation_params.json"
        engine.normalised_path = engine.output_dir / "normalised_features.parquet"

    if args.mode == "train":
        engine.fit_transform(df_in, vol_series=vol_series)
    else:
        norm_df = engine.transform(df_in, vol_series=vol_series)
        # In inference mode we do not write parquet by spec of CLI; just show summary
        logger.info("Inference complete. Rows: %d, Cols: %d", *norm_df.shape)
