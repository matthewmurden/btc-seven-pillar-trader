from __future__ import annotations

"""
Run Pillar 4 (Regime Detection) end-to-end on the data under RUN_LAYER/src/raw.

Stages:
 1) Part 1: raw regime features
 2) Part 2: deterministic regime engines
 3) Part 3: ML regime classifier (uses existing code; trains internally)
 4) Part 4: fusion layer
 5) Part 5: stability layer

Outputs are written under RUN_LAYER/src/raw/pillar_four/part_*/ to keep
them separate from the original training artifacts.
"""

import logging
import sys
import importlib.util
from pathlib import Path
import joblib

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parents[2]  # repo root
RAW_DIR = THIS_DIR / "raw"
PF_OUT = RAW_DIR / "pillar_four"

# Input: normalised features produced by done.py
NORM_INPUT = RAW_DIR / "normalised_features.parquet"
NORM_INPUT_PATCHED = PF_OUT / "normalised_features_for_p4.parquet"

PART1_OUT = PF_OUT / "part_one" / "raw_regime_features.parquet"
PART2_OUT = PF_OUT / "part_two" / "deterministic_regime_engines.parquet"
PART3_OUT = PF_OUT / "part_three" / "regime_ml_predictions.parquet"
PART4_OUT = PF_OUT / "part_four" / "regime_labels_fused.parquet"
PART5_OUT = PF_OUT / "part_five" / "regime_labels_stable.parquet"
PART3_MODEL = PF_OUT / "part_three" / "regime_ml_model.pkl"
# Source model (from Pillar 4 repo processed outputs)
P4_BASE = Path(__file__).resolve().parents[2] / "PILLAR_FOUR_REGIME_DETECTION"
PART3_MODEL_SRC = P4_BASE / "data" / "processed" / "part_three" / "regime_ml_model.pkl"

# Make Pillar 4 src importable (force absolute path from this file)
P4_BASE = Path(__file__).resolve().parents[2] / "PILLAR_FOUR_REGIME_DETECTION"
P4_SRC = P4_BASE / "src"
for p in (P4_SRC, P4_BASE):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

if not P4_SRC.exists():
    raise FileNotFoundError(
        f"Pillar 4 src directory not found at expected path: {P4_SRC}"
    )


def _load_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import modules after sys.path update (fallback to direct file loading)
try:
    import build_raw_regime_features as part1  # type: ignore
except ModuleNotFoundError:
    part1 = _load_module("build_raw_regime_features", P4_SRC / "build_raw_regime_features.py")

try:
    import deterministic_regime_engines as part2  # type: ignore
except ModuleNotFoundError:
    part2 = _load_module("deterministic_regime_engines", P4_SRC / "deterministic_regime_engines.py")

try:
    from part_three import pillar4_part3_ml_regime_classifier as part3  # type: ignore
except ModuleNotFoundError:
    part3 = _load_module(
        "pillar4_part3_ml_regime_classifier",
        P4_SRC / "part_three" / "pillar4_part3_ml_regime_classifier.py",
    )

try:
    from part_four import pillar4_part4_fusion_layer as part4  # type: ignore
except ModuleNotFoundError:
    part4 = _load_module(
        "pillar4_part4_fusion_layer",
        P4_SRC / "part_four" / "pillar4_part4_fusion_layer.py",
    )

try:
    from part_five import pillar4_part5_stability_layer as part5  # type: ignore
except ModuleNotFoundError:
    part5 = _load_module(
        "pillar4_part5_stability_layer",
        P4_SRC / "part_five" / "pillar4_part5_stability_layer.py",
    )


def _ensure_dirs() -> None:
    for p in [PART1_OUT, PART2_OUT, PART3_OUT, PART4_OUT, PART5_OUT, NORM_INPUT_PATCHED]:
        p.parent.mkdir(parents=True, exist_ok=True)


def _prepare_norm_input() -> Path:
    """
    Pillar 4 expects some microstructure fields that may not be present
    in the pre-built normalised_features. Add safe zero-fill stubs if missing.
    """
    import pandas as pd

    if not NORM_INPUT.exists():
        raise FileNotFoundError(f"Normalised features not found at {NORM_INPUT}. Run done.py/full_pipeline first.")

    df = pd.read_parquet(NORM_INPUT)
    added = False
    for col in ["wick_body_ratio", "order_imbalance"]:
        if col not in df.columns:
            df[col] = 0.0
            added = True
            logging.warning("P4: %s missing in normalised features; filling zeros.", col)

    # Avoid rewriting if nothing changed
    if added or not NORM_INPUT_PATCHED.exists():
        df.to_parquet(NORM_INPUT_PATCHED, index=False)

    return NORM_INPUT_PATCHED


def _run_part_one() -> None:
    logging.info("P4 Part 1: raw regime features")
    part1.NORM_INPUT = _prepare_norm_input()
    part1.OUT = PART1_OUT
    part1.main()


def _run_part_two() -> None:
    logging.info("P4 Part 2: deterministic regime engines")
    part2.INPUT_PATH = PART1_OUT
    part2.OUTPUT_PATH = PART2_OUT
    part2.main()


def _run_part_three() -> None:
    logging.info("P4 Part 3: ML regime classifier (load pre-trained model)")
    import pandas as pd

    raw_feats = pd.read_parquet(PART1_OUT)
    det = pd.read_parquet(PART2_OUT)

    df = part3.prepare_features(raw_feats, det)
    labels = part3.generate_labels(det)

    feature_df = df.drop(columns=[c for c in df.columns if c in ["open_time_ms", "open_time"]], errors="ignore")
    feature_df = part3.encode_categoricals(feature_df).fillna(0)

    # Load pre-trained model (prefer run-layer copy, fallback to source processed model)
    model = None
    for path in (PART3_MODEL, PART3_MODEL_SRC):
        if path.exists():
            try:
                model = joblib.load(path)
                logging.info("Loaded pre-trained regime ML model from %s", path)
                break
            except Exception as exc:
                logging.warning("Failed to load model from %s: %s", path, exc)

    if model is None:
        raise FileNotFoundError(
            f"No pre-trained regime ML model found at {PART3_MODEL} or {PART3_MODEL_SRC}. "
            f"Run the Pillar 4 Part 3 trainer to produce regime_ml_model.pkl."
        )

    preds = part3.predict(model, feature_df)

    PART3_OUT.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(PART3_OUT, index=False)
    logging.info("Saved regime ML predictions -> %s", PART3_OUT)


def _run_part_four() -> None:
    logging.info("P4 Part 4: fusion layer")
    part4.DETERMINISTIC_PATH = PART2_OUT
    part4.ML_PATH = PART3_OUT
    part4.OUTPUT_PATH = PART4_OUT
    part4.PART_TWO_DIR = PART2_OUT.parent
    part4.PART_THREE_DIR = PART3_OUT.parent
    part4.PART_FOUR_DIR = PART4_OUT.parent
    part4.main()


def _run_part_five() -> None:
    logging.info("P4 Part 5: stability layer")
    part5.RAW_FEATURES_PATH = PART1_OUT
    part5.DETERMINISTIC_PATH = PART2_OUT
    part5.ML_PATH = PART3_OUT
    part5.FUSED_PATH = PART4_OUT
    part5.OUTPUT_PATH = PART5_OUT
    part5.PART_FIVE_DIR = PART5_OUT.parent
    part5.PART_ONE_DIR = PART1_OUT.parent
    part5.PART_TWO_DIR = PART2_OUT.parent
    part5.PART_THREE_DIR = PART3_OUT.parent
    part5.PART_FOUR_DIR = PART4_OUT.parent
    part5.main()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    _ensure_dirs()

    if not NORM_INPUT.exists():
        raise FileNotFoundError(f"Normalised features not found at {NORM_INPUT}. Run done.py/full_pipeline first.")

    _run_part_one()
    _run_part_two()
    _run_part_three()
    _run_part_four()
    _run_part_five()

    logging.info("Pillar 4 pipeline complete. Outputs under %s", PF_OUT)


if __name__ == "__main__":
    main()
