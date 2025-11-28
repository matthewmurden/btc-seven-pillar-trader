from src.feature_engine.feature_engine import compute_all_features
from src.data_loader.binance_candles import load_candles_5m
from src.data_loader.binance_trades import load_trades
from data_loader.preprocess.resample_trades_to_5m import aggregate_trades

df_c = load_candles_5m("BTCUSDT")
df_t = aggregate_trades(load_trades("BTCUSDT"))
df_htf = {"15m": load_candles("15m"), "1h": load_candles("1h")}

df_features = compute_all_features(df_c, df_t, df_htf)

print(df_features.tail())