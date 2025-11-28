import numpy as np
import pandas as pd


class SignalPersistence:
    def __init__(
        self,
        max_life_1m: int = 3,
        max_life_5m: int = 8,
        max_life_15m: int = 20,
        base_decay_rate: float = 0.15,
        regime_multiplier_trend: float = 1.3,
        regime_multiplier_range: float = 0.7,
        regime_multiplier_breakout: float = 1.6,
    ):
        self.max_life_1m = max_life_1m
        self.max_life_5m = max_life_5m
        self.max_life_15m = max_life_15m
        self.base_decay_rate = base_decay_rate
        self.regime_mult = {
            0: regime_multiplier_range,
            1: regime_multiplier_trend,
            2: regime_multiplier_breakout,
        }

    def compute_decay_curve(self, df: pd.DataFrame) -> pd.DataFrame:
        persistence = (
            df["momentum_score_decayed"].abs() * 0.4 +
            df["trend_score_decayed"].abs() * 0.4 +
            df["orderflow_score_decayed"].abs() * 0.2
        )

        persistence = persistence.clip(0.15, 1.0)

        df["signal_lifetime_estimate"] = (
            persistence * 0.33 * self.max_life_1m +
            persistence * 0.33 * self.max_life_5m +
            persistence * 0.33 * self.max_life_15m
        )

        df["signal_lifetime_estimate"] *= df["regime_state"].map(self.regime_mult)

        return df

    def apply_persistence_decay(self, df: pd.DataFrame) -> pd.DataFrame:
        if "raw_entry_confidence" not in df.columns:
            raise ValueError("raw_entry_confidence missing")

        if "signal_age" not in df.columns:
            df["signal_age"] = 0
            prev_signal = 0.0
            for i in range(1, len(df)):
                if df.loc[i, "raw_entry_confidence"] * prev_signal > 0:
                    df.loc[i, "signal_age"] = df.loc[i - 1, "signal_age"] + 1
                else:
                    df.loc[i, "signal_age"] = 0
                prev_signal = df.loc[i, "raw_entry_confidence"]

        life = df["signal_lifetime_estimate"].clip(1, None)
        age = df["signal_age"]

        decay_factor = np.exp(-self.base_decay_rate * (age / life))
        df["raw_entry_confidence"] *= decay_factor

        return df
