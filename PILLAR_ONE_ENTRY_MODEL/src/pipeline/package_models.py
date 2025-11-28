"""
Step 6 — Package trained models, calibrator, and metadata into final_package.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

MODEL_FILES = [
    "model_1m.pkl",
    "model_5m.pkl",
    "model_15m.pkl",
    "model_breakout.pkl",
    "model_reversion.pkl",
]

CALIBRATOR_FILE = "calibrator.pkl"

MODELS_DIR = Path("models")
PACKAGE_DIR = MODELS_DIR / "final_package"
PRED_PATH = Path("data/processed/features_with_preds.parquet")


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _copy_artifacts():
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    for fname in MODEL_FILES + [CALIBRATOR_FILE]:
        src = MODELS_DIR / fname
        if not src.exists():
            raise FileNotFoundError(f"Missing artifact: {src}")
        dst = PACKAGE_DIR / fname
        dst.write_bytes(src.read_bytes())


def _build_manifest():
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Prediction dataset not found at {PRED_PATH}")
    df = pd.read_parquet(PRED_PATH)
    feature_schema = [c for c in df.columns if c not in {"timestamp"}]
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "data_path": str(PRED_PATH),
        "model_files": MODEL_FILES,
        "calibrator_file": CALIBRATOR_FILE,
        "feature_schema": feature_schema,
        "git_hash": _git_hash(),
    }
    manifest_path = PACKAGE_DIR / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print("Manifest written to", manifest_path)


def main():
    print("Packaging models and calibrator into", PACKAGE_DIR)
    _copy_artifacts()
    _build_manifest()
    print("Packaging complete.")


if __name__ == "__main__":
    main()
