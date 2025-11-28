from src.data_loader.api.api_candles import load_candles
import pandas as pd
import requests
from datetime import datetime, timedelta

BASE = "https://fapi.binance.com"

def download_all_candles(symbol="BTCUSDT", interval="5m", limit=1500, outfile="data/raw/candles_5m.csv"):
    """
    Downloads exactly 36 months of candles ending October 31st, 2025.
    Uses proper time-based pagination to scroll backwards.
    """

    df_all = []
    print("[*] Downloading 36 months of 5m candles...")

    # Set end date to October 31st, 2025 at 23:59:59 (end of October, excluding November 2025)
    end_date = datetime(2025, 10, 31, 23, 59, 59)
    # Set start date to November 1st, 2022 (exactly 36 months back)
    start_date = datetime(2022, 11, 1, 0, 0, 0)
    
    # Convert to milliseconds
    end_time = int(end_date.timestamp() * 1000)
    start_time = int(start_date.timestamp() * 1000)

    print(f"[*] Date range: {start_date} to {end_date}")
    print(f"[*] Exactly 36 months of data (November 2025 excluded)")

    chunk_count = 0
    while True:
        # Make direct API call with endTime parameter
        url = f"{BASE}/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "endTime": int(end_time)
        }
        
        resp = requests.get(url, params=params).json()
        
        if len(resp) == 0:
            print("[*] No more historical data available")
            break
            
        # Convert to DataFrame with correct 12 columns
        df = pd.DataFrame(resp, columns=[
            "open_time","open","high","low","close","volume","close_time",
            "quote_volume","trades","taker_buy_vol","taker_buy_quote","ignore"
        ])

        df = df[["open_time","open","high","low","close","volume","taker_buy_vol"]]
        df.columns = ["timestamp","open","high","low","close","volume","taker_buy_vol"]

        df["timestamp"] = df["timestamp"] // 1000  # ms → seconds
        df = df.astype({
            "open": float, "high": float, "low": float,
            "close": float, "volume": float, "taker_buy_vol": float
        })
        
        new_oldest = df["timestamp"].min()
        
        # Check if we've reached the start date
        if new_oldest <= (start_time // 1000):
            print(f"[*] Reached target start date: {start_date}")
            # Filter out candles before start date
            df = df[df["timestamp"] >= (start_time // 1000)]
            if len(df) > 0:
                df_all.append(df)
            break

        chunk_count += 1
        print(f"Downloaded chunk {chunk_count} ending at: {pd.to_datetime(new_oldest, unit='s')}")
        df_all.append(df)
        
        # Update end_time for next request
        end_time = new_oldest * 1000

    df_final = pd.concat(df_all).drop_duplicates().sort_values("timestamp")
    df_final.to_csv(outfile, index=False)

    print(f"[*] Downloaded {len(df_final)} total candles")
    print(f"[*] Date range: {pd.to_datetime(df_final['timestamp'].min(), unit='s')} to {pd.to_datetime(df_final['timestamp'].max(), unit='s')}")
    print("[*] Saved to:", outfile)


if __name__ == "__main__":
    download_all_candles()