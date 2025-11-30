"""
Utility functions for deterministic regime engines.
"""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def zscore(x: np.ndarray) -> np.ndarray:
    mean = np.nanmean(x)
    std = np.nanstd(x) + 1e-12
    return (x - mean) / std


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / (b + 1e-12)


def classify_threshold(value: np.ndarray, threshold: float) -> np.ndarray:
    return (value > threshold).astype(np.float32)
