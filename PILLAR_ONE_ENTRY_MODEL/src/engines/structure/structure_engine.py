import numpy as np
import pandas as pd

from .structure_engine_config import DEFAULT_STRUCTURE_CONFIG, StructureConfig


def _robust_z(series: pd.Series, clip: float) -> pd.Series:
    s = series.astype(float)
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    scale = iqr * 0.7413
    if not np.isfinite(scale) or scale < 1e-9:
        scale = s.std()
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    z = (s - med) / scale
    return z.clip(-clip, clip).fillna(0.0)


def compute_structure_engine_score(
    df: pd.DataFrame, config: StructureConfig = DEFAULT_STRUCTURE_CONFIG
) -> pd.DataFrame:
    """
    Build a structure score from swings, BOS, CHOCH, and structure signals.
    """
    out = df.copy()

    swing_up = out.get("swing_high", pd.Series(0.0, index=out.index)).astype(float)
    swing_down = out.get("swing_low", pd.Series(0.0, index=out.index)).astype(float)
    last_swing_up = out.get("last_swing_high", pd.Series(0.0, index=out.index)).astype(float)
    last_swing_down = out.get("last_swing_low", pd.Series(0.0, index=out.index)).astype(float)
    swing_score = _robust_z(
        (swing_up - swing_down) + 0.5 * (last_swing_up - last_swing_down),
        config.clip_value,
    )

    bos = _robust_z(
        out.get("bos_up", pd.Series(0.0, index=out.index)).astype(float) -
        out.get("bos_down", pd.Series(0.0, index=out.index)).astype(float),
        config.clip_value,
    )

    choch = _robust_z(
        out.get("choch_up", pd.Series(0.0, index=out.index)).astype(float) -
        out.get("choch_down", pd.Series(0.0, index=out.index)).astype(float),
        config.clip_value,
    )

    struct_sig = _robust_z(
        out.get("structure_signal", pd.Series(0.0, index=out.index)).astype(float),
        config.clip_value,
    )

    sweep_up = out.get("sweep_up", pd.Series(0.0, index=out.index)).astype(float)
    sweep_down = out.get("sweep_down", pd.Series(0.0, index=out.index)).astype(float)
    sweep_score = _robust_z(sweep_up - sweep_down, config.clip_value)

    weights = config
    raw_score = (
        weights.bos_weight * bos +
        weights.choch_weight * choch +
        weights.swing_weight * swing_score +
        weights.structure_signal_weight * struct_sig +
        weights.sweep_weight * sweep_score
    )

    out["structure_score"] = pd.Series(np.tanh(raw_score)).ewm(
        alpha=config.smoothing_alpha, adjust=False
    ).mean()
    return out
