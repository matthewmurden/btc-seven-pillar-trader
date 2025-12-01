# ======================================================
# model_utils.py
# ======================================================

from __future__ import annotations
import pickle
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "PILLAR_FOUR_REGIME_DETECTION" / "regime" / "ml_regime_classifier" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_model(model, name: str):
    with open(MODEL_DIR / name, "wb") as f:
        pickle.dump(model, f)


def load_model(name: str):
    with open(MODEL_DIR / name, "rb") as f:
        return pickle.load(f)


def save_feature_list(features):
    with open(MODEL_DIR / "feature_list.pkl", "wb") as f:
        pickle.dump(features, f)


def load_feature_list():
    with open(MODEL_DIR / "feature_list.pkl", "rb") as f:
        import pickle
        return pickle.load(f)


def save_metadata(meta: dict):
    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)