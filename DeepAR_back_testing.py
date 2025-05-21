# Imports and setup
import os
import glob
import pandas as pd
import numpy as np
import ccxt
import vectorbt as vbt
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
import concurrent.futures
import torch

# Use Tensor Cores for faster ops
torch.set_float32_matmul_precision('high')
print("Starting DeepAR backtest pipeline with enhanced logging...")

# --- Config ---
DATA_DIR = "data/binance_backtesting"
os.makedirs(DATA_DIR, exist_ok=True)
prediction_length = 10
context_length = 120
init_cash = 100000.0
position_size = 0.05  # 5% per trade
stop_loss = 0.03      # 3% stop-loss
take_profit = 0.07    # 7% take-profit
max_workers = 32      # threads for signal generation
fetch_workers = 8     # threads for fetching

# Bollinger Bands params
bb_window = 20
bb_std_factor = 2

# VMA params
vma_window = 120

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Step 1: Parallel fetch & cache data ---
print("[Step 1] Fetching OHLCV in parallel...")
exchange = ccxt.binanceus()
exchange.load_markets()
usdt_symbols = [s for s in exchange.symbols if s.endswith('/USDT')]
if not usdt_symbols:
    raise RuntimeError("No USDT symbols found")
if len(usdt_symbols) > 10:
    print(f"    Found {len(usdt_symbols)} symbols; fetching first 10 for demo.")
    usdt_symbols = usdt_symbols[:10]

six_weeks_ms = 42 * 24 * 60 * 60 * 1000
end_time = exchange.milliseconds()
start_time = end_time - six_weeks_ms

def fetch_and_cache(symbol: str):
    print(f"    [Fetch] {symbol}")
    fname = symbol.replace('/', '_') + '.parquet'
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        print(f"      Cached, skipping")
        return
    all_ohlcv = []
    since_ms = start_time
    while True:
        chunk = exchange.fetch_ohlcv(symbol, '1m', since=since_ms, limit=1000)
        if not chunk:
            break
        all_ohlcv.extend(chunk)
        since_ms = chunk[-1][0] + 60_000
        if len(chunk) < 1000:
            break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['symbol'] = symbol
    df.to_parquet(path)
    print(f"      Saved {len(df)} rows for {symbol}")

with concurrent.futures.ThreadPoolExecutor(max_workers=fetch_workers) as ex:
    ex.map(fetch_and_cache, usdt_symbols)
print("[Step 1] Fetch complete\n")

# --- Step 1b: Load data + compute indicators ---
print("[Step 1b] Loading data and computing RSI, Bollinger Bands, VMA...")
market_data = {}
for path in glob.glob(os.path.join(DATA_DIR, '*.parquet')):
    sym = os.path.basename(path).replace('.parquet','').replace('_','/')
    df = pd.read_parquet(path).set_index('timestamp')
    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    ma_up = up.ewm(span=14, adjust=False).mean()
    ma_down = down.ewm(span=14, adjust=False).mean()
    df['rsi'] = 100 - (100/(1 + ma_up/(ma_down + 1e-9)))
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(bb_window).mean()
    std = df['close'].rolling(bb_window).std()
    df['bb_upper'] = df['bb_mid'] + bb_std_factor * std
    df['bb_lower'] = df['bb_mid'] - bb_std_factor * std
    # Volume Moving Average
    df['vma'] = df['volume'].rolling(vma_window).mean()
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    market_data[sym] = df
print(f"    Prepared indicators for {len(market_data)} symbols\n")

# --- Step 2: Train DeepAR with sliding windows ---
print("[Step 2] Preparing sliding-window training data...")
train_items = []
window_stride = prediction_length
for symbol, df in market_data.items():
    n = len(df)
    cutoff = n - (14 * 24 * 60)
    if cutoff <= context_length + prediction_length:
        print(f"    Skipping {symbol}: insufficient data (n={n})")
        continue
    for end_i in range(context_length + prediction_length, cutoff, window_stride):
        start_i = end_i - (context_length + prediction_length)
        train_items.append({
            'target': df['close'].iloc[start_i:end_i].values,
            'feat_dynamic_real': [
                df['volume'].iloc[start_i:end_i].values,
                df['rsi'].iloc[start_i:end_i].values,
                df['bb_upper'].iloc[start_i:end_i].values,
                df['bb_lower'].iloc[start_i:end_i].values,
                df['vma'].iloc[start_i:end_i].values
            ],
            'start': df.index[start_i],
            'item_id': symbol
        })
print(f"    Training on {len(train_items)} windows")
training_ds = ListDataset(train_items, freq='T')
estimator = DeepAREstimator(
    freq='T',
    prediction_length=prediction_length,
    context_length=context_length,
    num_layers=3,
    hidden_size=100,
    batch_size=1024,
    trainer_kwargs={
        'max_epochs': 6,
        'accelerator': 'gpu' if device.type=='cuda' else 'cpu',
        'devices': 1,
        'precision': 16,
        'logger': False,
        'limit_val_batches': 0,
    }
)
predictor = estimator.train(training_data=training_ds)
if hasattr(predictor, 'network'):
    predictor.network.to(device)
    predictor.network.eval()
    predictor.network.half()
print("    Training complete\n")

# --- Step 3: Precompute and inference separation ---
print("[Step 3] Precomputing sliding windows on CPU...")
from numpy.lib.stride_tricks import sliding_window_view
windows_data = {}
for symbol, df in market_data.items():
    n = len(df)
    test_win = 14 * 24 * 60
    start_i = max(context_length, n - test_win)
    end_i   = n - prediction_length
    if end_i - start_i <= 0:
        continue
    closes_np = sliding_window_view(df['close'].values, context_length)[start_i-context_length:end_i-context_length]
    vols_np   = sliding_window_view(df['volume'].values, context_length)[start_i-context_length:end_i-context_length]
    rsi_np    = sliding_window_view(df['rsi'].values, context_length)[start_i-context_length:end_i-context_length]
    bb_up_np  = sliding_window_view(df['bb_upper'].values, context_length)[start_i-context_length:end_i-context_length]
    bb_low_np = sliding_window_view(df['bb_lower'].values, context_length)[start_i-context_length:end_i-context_length]
    vma_np    = sliding_window_view(df['vma'].values, context_length)[start_i-context_length:end_i-context_length]
    idxs = np.arange(start_i, end_i)
    windows_data[symbol] = (closes_np, vols_np, rsi_np, bb_up_np, bb_low_np, vma_np, idxs)
print("[Step 3] CPU aggregation complete")

print("[Step 3] Preparing global dataset for GPU inference...")
all_ds_items, index_map = [], []
for symbol, (closes_np, vols_np, rsi_np, bb_up_np, bb_low_np, vma_np, idxs) in windows_data.items():
    for close_win, vol_win, rsi_win, bb_up_win, bb_low_win, vma_win, idx in zip(
        closes_np, vols_np, rsi_np, bb_up_np, bb_low_np, vma_np, idxs
    ):
        all_ds_items.append({
            'target': close_win.tolist(),
            'feat_dynamic_real': [
                vol_win.tolist(),
                rsi_win.tolist(),
                bb_up_win.tolist(),
                bb_low_win.tolist(),
                vma_win.tolist()
            ],
            'start': market_data[symbol].index[idx-context_length]
        })
        index_map.append((symbol, idx))
print(f"    Aggregated {len(all_ds_items)} windows into one dataset")

dataset = ListDataset(all_ds_items, freq='T')
print("[Step 3] Running inference on GPU…")
with torch.amp.autocast(device.type, dtype=torch.float16):
    forecasts = list(predictor.predict(dataset, num_samples=1))

# --- Step 4: Map signals and backtest ---
price_df = pd.DataFrame({s: market_data[s]['close'] for s in market_data}).dropna(axis=1)
n = min(map(len, price_df.values.T))
price_df = price_df.iloc[-n:]
symbols = price_df.columns.tolist()
entries = {sym: np.zeros(len(price_df), dtype=bool) for sym in symbols}
exits   = {sym: np.zeros(len(price_df), dtype=bool) for sym in symbols}
for (symbol, idx), fc in zip(index_map, forecasts):
    if symbol not in symbols:
        continue
    rel_idx = idx - (len(market_data[symbol]) - len(price_df))
    if not 0 <= rel_idx < len(price_df):
        continue
    price = price_df[symbol].values[rel_idx]
    mean_pred = float(fc.mean[-1])
    if mean_pred >= price * 1.05:
        entries[symbol][rel_idx] = True
        exit_point = rel_idx + prediction_length
        if exit_point < len(price_df):
            exits[symbol][exit_point] = True
print("[Step 4] Backtesting with vectorbt with SL/TP...")
entries_df = pd.DataFrame(entries, index=price_df.index)
exits_df   = pd.DataFrame(exits,   index=price_df.index)
# Create portfolio without stops
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
print("    Base backtest complete")

# Apply stop-loss and take-profit
print("    Applying stop-loss and take-profit overlays...")
try:
    pf = pf.apply_stop_loss(stop_loss, sl_stop_type='percent')
    pf = pf.apply_take_profit(take_profit, tp_stop_type='percent')
    print("    Stop-loss and take-profit applied")
except AttributeError:
    print("    Stop-loss/take-profit methods not available in this VectorBT version. Please upgrade to v0.26+ to use overlays.")

print("    Final backtest ready    Backtest complete")
stats = pf.stats()
print(stats)
stats.to_csv(os.path.join(DATA_DIR, "backtest_stats.csv"))
pf.plot().show()
print("Pipeline finished")
