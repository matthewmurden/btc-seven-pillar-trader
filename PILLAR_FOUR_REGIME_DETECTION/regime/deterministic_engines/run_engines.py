# ============================================
# run_engines.py
# Master deterministic engine executor
# ============================================

import pandas as pd

from .trend_engine import compute_trend_engine
from .chop_engine import compute_chop_engine
from .volatility_engine import compute_volatility_engine
from .liquidity_engine import compute_liquidity_engine
from .mean_reversion_engine import compute_mean_reversion_engine


def run_all_engines(df: pd.DataFrame) -> pd.DataFrame:
    """Run all deterministic engines and return unified output."""
    
    out = pd.DataFrame(index=df.index)

    # Trend
    trend = compute_trend_engine(df)
    out["trend_strength_score"] = trend["trend_strength_score"]
    out["trend_flag"] = trend["trend_flag"]

    # Chop
    chop = compute_chop_engine(df)
    out["chop_strength_score"] = chop["chop_strength_score"]
    out["chop_flag"] = chop["chop_flag"]

    # Volatility
    vol = compute_volatility_engine(df)
    out["volatility_regime"] = vol["volatility_regime"]
    out["volatility_regime_score"] = vol["volatility_regime_score"]

    # Liquidity
    liq = compute_liquidity_engine(df)
    out["liquidity_regime"] = liq["liquidity_regime"]
    out["liquidity_score"] = liq["liquidity_score"]

    # Mean reversion
    mr = compute_mean_reversion_engine(df)
    out["mean_reversion_flag"] = mr["mean_reversion_flag"]
    out["mean_reversion_score"] = mr["mean_reversion_score"]

    return out.fillna(0)
