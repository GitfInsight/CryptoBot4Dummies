import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time

# List of symbols to fetch (Binance.US format)
symbols = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'DOGEUSDT']
interval = '1m'
base_url = 'https://api.binance.us'  # Swap to .com if outside the US

# Pull 14 days of data for each symbol
for symbol in symbols:
    print(f"⏳ Fetching {symbol}...")
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - 14 * 24 * 60 * 60 * 1000  # 14 days in ms

    rows = []
    while start_ts < end_ts:
        try:
            r = requests.get(
                f"{base_url}/api/v3/klines",
                params={
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': start_ts,
                    'endTime': end_ts,
                    'limit': 1000
                },
                timeout=10
            )
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            start_ts = batch[-1][0] + 1
            time.sleep(0.05)
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            break

    if not rows:
        print(f"⚠️ No data returned for {symbol}, skipping...")
        continue

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base",
        "taker_quote","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    df = df[["open","high","low","close","volume"]].astype(float)

    # Save to file
    filename = f"{symbol.lower()}_1m_last2w.parquet"
    pq.write_table(pa.Table.from_pandas(df), filename)
    print(f"✅ {filename} saved with {len(df)} rows.")