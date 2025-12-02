import numpy as np
import pandas as pd

def _sigmoid(x: pd.Series) -> pd.Series:
    return 1 / (1 + np.exp(-x))


def compute_volatility_engine(df: pd.DataFrame) -> pd.DataFrame:
    # Choose a representative realised volatility input
    vol = df.get("realised_vol_15m", df.get("realised_vol_5m", pd.Series(0, index=df.index)))
    vol_mean = vol.mean()
    vol_std = vol.std()
    vol_z = (vol - vol_mean) / (vol_std if vol_std != 0 else 1.0)

    high_vol = vol_z > 1
    low_vol = vol_z < -1
    regime = pd.Series("normal", index=df.index)
    regime = regime.mask(high_vol, "expansion")
    regime = regime.mask(low_vol, "compression")

    expansion_probability = _sigmoid(vol_z)
    compression_probability = _sigmoid(-vol_z)

    return pd.DataFrame(
        {
            "volatility_regime": regime,
            "expansion_probability": expansion_probability,
            "compression_probability": compression_probability,
        },
        index=df.index,
    )
