import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from mini_engines.adverse_move import compute_adverse_move  # noqa: E402
from mini_engines.favourable_move import compute_favourable_move  # noqa: E402
from mini_engines.path_shape import compute_path_shape  # noqa: E402
from mini_engines.regime_exit import compute_regime_exit  # noqa: E402
from mini_engines.volatility_excursion import compute_volatility_excursion  # noqa: E402


def _base_df(size: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "imbalance": rng.normal(0, 0.1, size),
            "delta": rng.normal(0, 0.2, size),
            "volume_expansion_5": rng.uniform(0.5, 2.0, size),
            "distance_to_low": rng.uniform(0.0, 1.0, size),
            "wick_imbalance_score": rng.uniform(-0.5, 0.5, size),
            "ATR_14": rng.uniform(0.1, 1.0, size),
            "momentum_score": rng.normal(0, 1, size),
            "trend_score": rng.normal(0, 1, size),
            "structure_score": rng.normal(0, 1, size),
            "volatility_score": rng.normal(0, 1, size),
            "short_term_exhaustion": rng.uniform(0, 100, size),
            "micro_trend_entropy": rng.uniform(0, 1, size),
            "bar_clustering_score": rng.uniform(0, 1, size),
            "realized_vol_5": rng.uniform(0, 0.02, size),
            "realized_vol_15": rng.uniform(0, 0.03, size),
            "realized_vol_30": rng.uniform(0, 0.04, size),
            "volatility_curve_5": rng.uniform(0, 0.02, size),
            "volatility_curve_15": rng.uniform(0, 0.02, size),
            "volatility_curve_30": rng.uniform(0, 0.02, size),
            "drift_direction": rng.uniform(-0.5, 0.5, size),
            "volume_expansion_5_dup": rng.uniform(0.5, 2.0, size),  # unused helper
            "wick_ratio": rng.uniform(0, 1, size),
            "spread": rng.uniform(0.01, 0.1, size),
            "volatility_compression_score": rng.uniform(0.1, 1.0, size),
            "ATR_slope": rng.normal(0, 0.05, size),
        }
    )


class TestAdverseMove:
    def test_basic_output(self):
        df = _base_df(10)
        out = compute_adverse_move(df)
        assert "mae_risk_score" in out
        assert len(out) == len(df)
        assert out.index.equals(df.index)
        assert pd.api.types.is_float_dtype(out["mae_risk_score"])
        assert out["mae_risk_score"].between(0, 1).all()
        assert out["mae_risk_score"].notna().all()

    def test_overwrite_protection(self):
        df = _base_df(3)
        out = compute_adverse_move(df)
        with pytest.raises(ValueError):
            compute_adverse_move(out)

    def test_directional_imbalance(self):
        df = _base_df(2)
        df.loc[0, "imbalance"] = -0.5
        df.loc[1, "imbalance"] = 0.2
        out = compute_adverse_move(df)
        assert out.loc[0, "mae_risk_score"] > out.loc[1, "mae_risk_score"]

    def test_directional_delta(self):
        df = _base_df(2)
        df.loc[0, "delta"] = -1.0
        df.loc[1, "delta"] = 0.5
        out = compute_adverse_move(df)
        assert out.loc[0, "mae_risk_score"] > out.loc[1, "mae_risk_score"]

    def test_directional_distance_to_low(self):
        df = _base_df(2)
        df.loc[0, "distance_to_low"] = 0.0
        df.loc[1, "distance_to_low"] = 1.0
        out = compute_adverse_move(df)
        assert out.loc[0, "mae_risk_score"] > out.loc[1, "mae_risk_score"]

    def test_directional_wick_imbalance(self):
        df = _base_df(2)
        df.loc[0, "wick_imbalance_score"] = 1.0
        df.loc[1, "wick_imbalance_score"] = 0.0
        out = compute_adverse_move(df)
        assert out.loc[0, "mae_risk_score"] > out.loc[1, "mae_risk_score"]

    def test_directional_atr(self):
        df = _base_df(2)
        df.loc[0, "ATR_14"] = 2.0
        df.loc[1, "ATR_14"] = 0.1
        out = compute_adverse_move(df)
        assert out.loc[0, "mae_risk_score"] > out.loc[1, "mae_risk_score"]

    def test_edge_cases_zero_and_constants(self):
        df_zero = pd.DataFrame(
            {
                "imbalance": [0.0],
                "delta": [0.0],
                "volume_expansion_5": [0.0],
                "distance_to_low": [0.0],
                "wick_imbalance_score": [0.0],
                "ATR_14": [0.0],
            }
        )
        out = compute_adverse_move(df_zero)
        assert out["mae_risk_score"].between(0, 1).all()
        assert out["mae_risk_score"].notna().all()

        df_const = pd.concat([df_zero] * 3, ignore_index=True)
        out_const = compute_adverse_move(df_const)
        assert out_const["mae_risk_score"].notna().all()

    def test_nan_handling(self):
        df = _base_df(3)
        df.loc[:, ["imbalance", "delta", "volume_expansion_5", "distance_to_low"]] = np.nan
        out = compute_adverse_move(df)
        assert out["mae_risk_score"].notna().all()
        assert out["mae_risk_score"].between(0, 1).all()

    def test_large_dataframe(self):
        df = _base_df(3000)
        out = compute_adverse_move(df)
        assert len(out) == 3000
        assert out["mae_risk_score"].notna().all()
        assert out["mae_risk_score"].between(0, 1).all()

    def test_reproducibility(self):
        df = _base_df(20)
        out1 = compute_adverse_move(df.copy())
        out2 = compute_adverse_move(df.copy())
        pd.testing.assert_series_equal(out1["mae_risk_score"], out2["mae_risk_score"])


class TestFavourableMove:
    def test_basic_output(self):
        df = _base_df(10)
        out = compute_favourable_move(df)
        assert "trend_continuation_score" in out
        assert len(out) == len(df)
        assert out.index.equals(df.index)
        assert pd.api.types.is_float_dtype(out["trend_continuation_score"])
        assert out["trend_continuation_score"].between(0, 1).all()
        assert out["trend_continuation_score"].notna().all()

    def test_overwrite_protection(self):
        df = _base_df(3)
        out = compute_favourable_move(df)
        with pytest.raises(ValueError):
            compute_favourable_move(out)

    def test_directional_scores(self):
        df = _base_df(2)
        df.loc[0, ["momentum_score", "trend_score", "structure_score"]] = [1.0, 1.0, 1.0]
        df.loc[1, ["momentum_score", "trend_score", "structure_score"]] = [-1.0, -1.0, -1.0]
        out = compute_favourable_move(df)
        assert out.loc[0, "trend_continuation_score"] > out.loc[1, "trend_continuation_score"]

    def test_volatility_penalty_for_extremes(self):
        df = _base_df(2)
        median_like = 0.0
        df.loc[0, "volatility_score"] = median_like
        df.loc[1, "volatility_score"] = 5.0
        out = compute_favourable_move(df)
        assert out.loc[0, "trend_continuation_score"] > out.loc[1, "trend_continuation_score"]

    def test_rsi_exhaustion_inversion(self):
        df = _base_df(2)
        df.loc[0, "short_term_exhaustion"] = 10  # low RSI
        df.loc[1, "short_term_exhaustion"] = 90  # high RSI
        out = compute_favourable_move(df)
        assert out.loc[0, "trend_continuation_score"] > out.loc[1, "trend_continuation_score"]

    def test_edge_cases_zero_and_constant(self):
        df_zero = pd.DataFrame(
            {
                "momentum_score": [0.0],
                "trend_score": [0.0],
                "structure_score": [0.0],
                "volatility_score": [0.0],
                "short_term_exhaustion": [50.0],
            }
        )
        out = compute_favourable_move(df_zero)
        assert out["trend_continuation_score"].between(0, 1).all()
        assert out["trend_continuation_score"].notna().all()

        df_const = pd.concat([df_zero] * 4, ignore_index=True)
        out_const = compute_favourable_move(df_const)
        assert out_const["trend_continuation_score"].notna().all()

    def test_nan_handling(self):
        df = _base_df(3)
        df.loc[:, ["momentum_score", "trend_score", "structure_score", "volatility_score"]] = np.nan
        out = compute_favourable_move(df)
        assert out["trend_continuation_score"].notna().all()
        assert out["trend_continuation_score"].between(0, 1).all()

    def test_small_and_large(self):
        df_small = _base_df(1)
        out_small = compute_favourable_move(df_small)
        assert len(out_small) == 1
        assert out_small["trend_continuation_score"].between(0, 1).all()

        df_large = _base_df(3000)
        out_large = compute_favourable_move(df_large)
        assert len(out_large) == 3000
        assert out_large["trend_continuation_score"].notna().all()

    def test_reproducibility(self):
        df = _base_df(15)
        out1 = compute_favourable_move(df.copy())
        out2 = compute_favourable_move(df.copy())
        pd.testing.assert_series_equal(out1["trend_continuation_score"], out2["trend_continuation_score"])


class TestPathShape:
    def test_basic_output(self):
        df = _base_df(10)
        out = compute_path_shape(df)
        assert "path_entropy_score" in out
        assert len(out) == len(df)
        assert out.index.equals(df.index)
        assert pd.api.types.is_float_dtype(out["path_entropy_score"])
        assert out["path_entropy_score"].between(0, 1).all()
        assert out["path_entropy_score"].notna().all()

    def test_overwrite_protection(self):
        df = _base_df(3)
        out = compute_path_shape(df)
        with pytest.raises(ValueError):
            compute_path_shape(out)

    def test_directional_entropy(self):
        df = _base_df(2)
        df.loc[0, "micro_trend_entropy"] = 0.9
        df.loc[1, "micro_trend_entropy"] = 0.1
        out = compute_path_shape(df)
        assert out.loc[0, "path_entropy_score"] < out.loc[1, "path_entropy_score"]

    def test_directional_clustering(self):
        df = _base_df(2)
        df.loc[0, "bar_clustering_score"] = 0.9
        df.loc[1, "bar_clustering_score"] = 0.1
        out = compute_path_shape(df)
        assert out.loc[0, "path_entropy_score"] > out.loc[1, "path_entropy_score"]

    def test_directional_realized_vol(self):
        df = _base_df(2)
        df.loc[0, "realized_vol_5"] = 0.05
        df.loc[1, "realized_vol_5"] = 0.0
        out = compute_path_shape(df)
        assert out.loc[0, "path_entropy_score"] < out.loc[1, "path_entropy_score"]

    def test_edge_cases(self):
        df_zero = pd.DataFrame(
            {
                "micro_trend_entropy": [0.0],
                "bar_clustering_score": [0.0],
                "realized_vol_5": [0.0],
            }
        )
        out = compute_path_shape(df_zero)
        assert out["path_entropy_score"].between(0, 1).all()
        assert out["path_entropy_score"].notna().all()

        df_const = pd.concat([df_zero] * 3, ignore_index=True)
        out_const = compute_path_shape(df_const)
        assert out_const["path_entropy_score"].notna().all()

    def test_nan_and_small_large(self):
        df_nan = _base_df(2)
        df_nan.loc[:, ["micro_trend_entropy", "bar_clustering_score", "realized_vol_5"]] = np.nan
        out_nan = compute_path_shape(df_nan)
        assert out_nan["path_entropy_score"].notna().all()

        df_large = _base_df(3000)
        out_large = compute_path_shape(df_large)
        assert len(out_large) == 3000
        assert out_large["path_entropy_score"].between(0, 1).all()

    def test_reproducibility(self):
        df = _base_df(12)
        out1 = compute_path_shape(df.copy())
        out2 = compute_path_shape(df.copy())
        pd.testing.assert_series_equal(out1["path_entropy_score"], out2["path_entropy_score"])


class TestRegimeExit:
    def test_basic_output(self):
        df = _base_df(10)
        out = compute_regime_exit(df)
        assert "exit_regime" in out
        assert len(out) == len(df)
        assert out.index.equals(df.index)
        assert pd.api.types.is_integer_dtype(out["exit_regime"])
        assert out["exit_regime"].isin([0, 1, 2]).all()
        assert out["exit_regime"].notna().all()

    def test_overwrite_protection(self):
        df = _base_df(3)
        out = compute_regime_exit(df)
        with pytest.raises(ValueError):
            compute_regime_exit(out)

    def test_classification_breakout_trend_range(self):
        df = pd.DataFrame(
            {
                "volatility_curve_5": [0.9, 0.5, 0.1],
                "volatility_curve_15": [0.7, 0.6, 0.2],
                "volatility_curve_30": [0.6, 0.6, 0.2],
                "drift_direction": [0.9, 0.4, 0.05],
                "volume_expansion_5": [2.0, 1.0, 0.1],
                "wick_ratio": [0.5, 0.5, 0.5],
                "spread": [0.2, 0.1, 0.01],
            }
        )
        out = compute_regime_exit(df)
        assert out.loc[0, "exit_regime"] == 2  # breakout
        assert out.loc[1, "exit_regime"] == 1  # trend
        assert out.loc[2, "exit_regime"] == 0  # range

    def test_nan_safety(self):
        df = _base_df(3)
        df.loc[:, ["volatility_curve_5", "volatility_curve_15", "volatility_curve_30"]] = np.nan
        out = compute_regime_exit(df)
        assert out["exit_regime"].notna().all()
        assert out["exit_regime"].isin([0, 1, 2]).all()

    def test_small_and_large(self):
        df_small = _base_df(1)
        out_small = compute_regime_exit(df_small)
        assert len(out_small) == 1
        assert out_small["exit_regime"].isin([0, 1, 2]).all()

        df_large = _base_df(3000)
        out_large = compute_regime_exit(df_large)
        assert len(out_large) == 3000

    def test_reproducibility(self):
        df = _base_df(10)
        out1 = compute_regime_exit(df.copy())
        out2 = compute_regime_exit(df.copy())
        pd.testing.assert_series_equal(out1["exit_regime"], out2["exit_regime"])


class TestVolatilityExcursion:
    def test_basic_output(self):
        df = _base_df(10)
        out = compute_volatility_excursion(df)
        assert "mfe_potential_score" in out
        assert len(out) == len(df)
        assert out.index.equals(df.index)
        assert pd.api.types.is_float_dtype(out["mfe_potential_score"])
        assert out["mfe_potential_score"].between(0, 1).all()
        assert out["mfe_potential_score"].notna().all()

    def test_overwrite_protection(self):
        df = _base_df(3)
        out = compute_volatility_excursion(df)
        with pytest.raises(ValueError):
            compute_volatility_excursion(out)

    def test_directional_atr_slope(self):
        df = _base_df(2)
        df.loc[0, "ATR_slope"] = 0.5
        df.loc[1, "ATR_slope"] = -0.5
        out = compute_volatility_excursion(df)
        assert out.loc[0, "mfe_potential_score"] > out.loc[1, "mfe_potential_score"]

    def test_directional_compression(self):
        df = _base_df(2)
        df.loc[0, "volatility_compression_score"] = 0.0
        df.loc[1, "volatility_compression_score"] = 1.0
        out = compute_volatility_excursion(df)
        assert out.loc[0, "mfe_potential_score"] > out.loc[1, "mfe_potential_score"]

    def test_directional_volume_expansion(self):
        df = _base_df(2)
        df.loc[0, "volume_expansion_5"] = 3.0
        df.loc[1, "volume_expansion_5"] = 0.5
        out = compute_volatility_excursion(df)
        assert out.loc[0, "mfe_potential_score"] > out.loc[1, "mfe_potential_score"]

    def test_directional_curvature(self):
        df = _base_df(2)
        df.loc[0, ["realized_vol_5", "realized_vol_15", "realized_vol_30"]] = [0.04, 0.01, 0.0]
        df.loc[1, ["realized_vol_5", "realized_vol_15", "realized_vol_30"]] = [0.0, 0.0, 0.0]
        out = compute_volatility_excursion(df)
        assert out.loc[0, "mfe_potential_score"] > out.loc[1, "mfe_potential_score"]

    def test_directional_wick_penalty(self):
        df = _base_df(2)
        df.loc[0, "wick_ratio"] = 0.5  # neutral
        df.loc[1, "wick_ratio"] = 1.0  # extreme
        out = compute_volatility_excursion(df)
        assert out.loc[0, "mfe_potential_score"] > out.loc[1, "mfe_potential_score"]

    def test_directional_spread(self):
        df = _base_df(2)
        df.loc[0, "spread"] = 0.2
        df.loc[1, "spread"] = 0.01
        out = compute_volatility_excursion(df)
        assert out.loc[0, "mfe_potential_score"] > out.loc[1, "mfe_potential_score"]

    def test_edge_cases(self):
        df_zero = pd.DataFrame(
            {
                "ATR_14": [0.0],
                "ATR_slope": [0.0],
                "realized_vol_5": [0.0],
                "realized_vol_15": [0.0],
                "realized_vol_30": [0.0],
                "volatility_compression_score": [0.0],
                "spread": [0.0],
                "wick_ratio": [0.5],
                "volume_expansion_5": [0.0],
            }
        )
        out_zero = compute_volatility_excursion(df_zero)
        assert out_zero["mfe_potential_score"].between(0, 1).all()
        assert out_zero["mfe_potential_score"].notna().all()

        df_nan = df_zero.copy()
        df_nan.loc[0] = np.nan
        out_nan = compute_volatility_excursion(df_nan)
        assert out_nan["mfe_potential_score"].notna().all()

    def test_small_and_large(self):
        df_small = _base_df(1)
        out_small = compute_volatility_excursion(df_small)
        assert len(out_small) == 1
        assert out_small["mfe_potential_score"].between(0, 1).all()

        df_large = _base_df(3000)
        out_large = compute_volatility_excursion(df_large)
        assert len(out_large) == 3000
        assert out_large["mfe_potential_score"].between(0, 1).all()

    def test_reproducibility(self):
        df = _base_df(15)
        out1 = compute_volatility_excursion(df.copy())
        out2 = compute_volatility_excursion(df.copy())
        pd.testing.assert_series_equal(out1["mfe_potential_score"], out2["mfe_potential_score"])
