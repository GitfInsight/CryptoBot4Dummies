# Imports and setup
import os
import pandas as pd
import numpy as np
import ccxt
import vectorbt as vbt
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator

# --- Config ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
prediction_length = 10
context_length = 120
init_cash = 100000.0
position_size = 0.05  # 5% per trade

# --- Step 1: Fetch data from Binance and cache to Parquet ---
exchange = ccxt.binanceus()  # Use Binance US due to regional restrictions
exchange.load_markets()
usdt_symbols = [s for s in exchange.symbols if s.endswith('/USDT')]
# Enable dry run to only process a few symbols quickly
DRY_RUN = False
if DRY_RUN:
    usdt_symbols = usdt_symbols[:2]

# Get 6 weeks of data to split: 4 weeks train, 2 weeks test
six_weeks_ms = 42 * 24 * 60 * 60 * 1000
end_time = exchange.milliseconds()
start_time = end_time - six_weeks_ms

market_data = {}
def fetch_symbol_data(symbol: str, since_ms: int):
    fname = symbol.replace("/", "_") + ".parquet"
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        return pd.read_parquet(path)
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, '1m', since=since_ms, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since_ms = ohlcv[-1][0] + 60_000
        if len(ohlcv) < 1000:
            break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.to_parquet(path)
    return df

for symbol in usdt_symbols:
    try:
        df = fetch_symbol_data(symbol, start_time)
        market_data[symbol] = df
    except Exception as e:
        print(f"{symbol} fetch failed: {e}")

# --- Step 2: Train DeepAR on all time series ---
import torch
print("✅ GPU is available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using device:", torch.cuda.get_device_name(0))

device = "cuda" if torch.cuda.is_available() else "cpu"
train_data = []
for symbol, df in market_data.items():
    n = len(df)
    if n < (42 * 24 * 60):
        print(f"Skipping {symbol}: insufficient data ({n} rows)")
        continue
    train_cutoff = n - (14 * 24 * 60)  # last 2 weeks for testing
    close_arr = df['close'].iloc[:train_cutoff].to_numpy()
    vol_arr = df['volume'].iloc[:train_cutoff].to_numpy()
    rsi_arr = df['close'].iloc[:train_cutoff].rolling(14).apply(
        lambda x: 100 - (100 / (1 + ((x[-1] - x.min()) / (x.max() - x[-1] + 1e-9)))), raw=True
    ).fillna(0).to_numpy()
    train_data.append({
        "target": close_arr,
        "feat_dynamic_real": [vol_arr, rsi_arr],
        "start": df.index[0],
        "item_id": symbol
    })
training_ds = ListDataset(train_data, freq="T")

estimator = DeepAREstimator(
    freq="T",
    prediction_length=prediction_length,
    context_length=context_length,
    num_layers=3,
    hidden_size=100,
    trainer_kwargs={"max_epochs": 10, "accelerator": "gpu" if device == "cuda" else "cpu", "devices": 1}
)
predictor = estimator.train(training_data=training_ds)

# --- Step 3: Generate signals from DeepAR and backtest ---
entries, exits = {}, {}
for symbol, df in market_data.items():
    n = len(df)
    test_start = n - (14 * 24 * 60)
    e_sig = np.zeros(n, dtype=bool)
    x_sig = np.zeros(n, dtype=bool)
    in_position = False

    for i in range(test_start, n - prediction_length):
        # Prepare history window
        c_hist = df['close'].iloc[max(0, i-context_length):i].to_numpy()
        vol_hist = df['volume'].iloc[max(0, i-context_length):i].to_numpy()
        rsi_hist = df['close'].iloc[max(0, i-context_length):i].rolling(14).apply(
            lambda x: 100 - (100 / (1 + ((x[-1] - x.min()) / (x.max() - x[-1] + 1e-9)))), raw=True
        ).fillna(0).to_numpy()

        # Forecast
        pred_ds = ListDataset([{"target": c_hist, "feat_dynamic_real": [vol_hist, rsi_hist], "start": df.index[max(0, i-context_length)]}], freq="T")
        forecast = next(predictor.predict(pred_ds))
        predicted_mean = float(forecast.mean[-1])

        # Safely index current price and timestamp
        try:
            current_price = df['close'].iat[i]
            current_time = df.index[i]
        except IndexError:
            print(f"[SKIP] Index {i} out of bounds for {symbol} (len={n})")
            continue

        # Log forecast
        with open(os.path.join(DATA_DIR, "forecast_log.csv"), "a") as log_file:
            log_file.write(f"{symbol},{current_time},{current_price},{predicted_mean}\n")

        entry_threshold = 1.05  # Buy if forecast > 5%
        exit_threshold = 1.00   # Sell if forecast <= current

        if in_position:
            # Exit logic
            if predicted_mean <= exit_threshold * current_price:
                x_sig[i] = True
                in_position = False
            continue

        # Entry logic
        if predicted_mean >= entry_threshold * current_price:
            e_sig[i] = True
            x_idx = i + prediction_length
            if x_idx < n:
                x_sig[x_idx] = True
            in_position = True

    # Store signals
    if len(e_sig) == n:
        entries[symbol] = e_sig
        exits[symbol] = x_sig
    else:
        print(f"Signal length mismatch for {symbol}: e_sig={len(e_sig)}, expected={n}")

# Build price and signal DataFrames
price_df = pd.DataFrame({s: df['close'] for s, df in market_data.items()}).dropna(axis=1)
n = min(map(len, price_df.values.T))
price_df = price_df.iloc[-n:]
entries_df = pd.DataFrame(entries).iloc[-n:]
exits_df = pd.DataFrame(exits).iloc[-n:]

# --- Step 4: Backtest using vectorbt ---
pf = vbt.Portfolio.from_signals(
    close=price_df,
    entries=entries_df,
    exits=exits_df,
    init_cash=init_cash,
    fees=0.001,
    slippage=0.001,
    size=position_size,
    size_type='Percent',
    cash_sharing=True
)

# Output stats
stats = pf.stats()
print(stats)
stats.to_csv(os.path.join(DATA_DIR, "backtest_stats.csv"))

# Plot results
pf.plot().show()