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
from kelly_criterion import calculate_kelly_fraction

# Set vectorbt plotting backend to matplotlib to align with plt.subplots usage
vbt.settings.plotting['backend'] = 'matplotlib'

# Use Tensor Cores for faster ops
torch.set_float32_matmul_precision('high')
print("Starting DeepAR backtest pipeline with enhanced logging...")

# --- Config ---
DATA_DIR = "data/binance_backtesting"
os.makedirs(DATA_DIR, exist_ok=True)
prediction_length = 10
context_length = 120
init_cash = 100000.0
max_position_size = 0.10  # Maximum 10% per trade
stop_loss = 0.03      # 3% stop-loss
take_profit = 0.07    # 7% take-profit
max_workers = 32      # threads for signal generation
fetch_workers = 8     # threads for fetching
MAX_THREADS = 32
# Bollinger Bands params
bb_window = 20
bb_std_factor = 2

# VMA params
vma_window = 120

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Determine precision for PyTorch Lightning Trainer
if device.type == 'cpu':
    effective_precision = 32
    print(f"    Device is CPU, setting precision for trainer to {effective_precision} to avoid BFloat16 issues.")
else: # For 'cuda' (NVIDIA or AMD with ROCm)
    effective_precision = 16
    print(f"    Device is {device.type}, setting precision for trainer to {effective_precision} (mixed precision).")

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
    print(f"    [Fetch Attempt] {symbol}")
    fname = symbol.replace('/', '_') + '.parquet'
    path = os.path.join(DATA_DIR, fname)

    if os.path.exists(path):
        try:
            df_check = pd.read_parquet(path)
            if not df_check.empty:
                print(f"      [Cache] Found valid cached data for {symbol}, {len(df_check)} rows. Skipping fetch.")
                return
            else:
                print(f"      [Cache] Found empty/invalid cached file for {symbol}. Attempting to re-fetch.")
                # Optionally remove the empty/invalid file
                # os.remove(path) 
        except Exception as e:
            print(f"      [Cache] Error reading cached file for {symbol}: {e}. Attempting to re-fetch.")
            try:
                os.remove(path)
                print(f"      [Cache] Removed potentially corrupt/empty file: {path}")
            except OSError as oe:
                print(f"      [Cache] Failed to remove corrupt/empty file {path}: {oe}")
    
    all_ohlcv = []
    current_since_ms = start_time
    print(f"      [Fetch] Starting download for {symbol} from {pd.to_datetime(current_since_ms, unit='ms')} up to {pd.to_datetime(end_time, unit='ms')}")
    try:
        while current_since_ms < end_time:
            # print(f"        Fetching 1m KLV for {symbol} since {pd.to_datetime(current_since_ms, unit='ms')}")
            chunk = exchange.fetch_ohlcv(symbol, '1m', since=current_since_ms, limit=1000)
            
            if not chunk:
                print(f"      [Fetch] No more data returned for {symbol} at timestamp {pd.to_datetime(current_since_ms, unit='ms')}. Total fetched: {len(all_ohlcv)} records.")
                break
            
            all_ohlcv.extend(chunk)
            last_timestamp_in_chunk = chunk[-1][0]
            
            # print(f"        Fetched {len(chunk)} records for {symbol}. Last timestamp: {pd.to_datetime(last_timestamp_in_chunk, unit='ms')}.")

            if len(chunk) < 1000: # Less than limit means it's the last available chunk
                print(f"      [Fetch] Fetched last chunk (size {len(chunk)}) for {symbol}. Total: {len(all_ohlcv)} records.")
                break

            new_since_ms = last_timestamp_in_chunk + 60_000 
            if new_since_ms <= current_since_ms:
                print(f"      [Fetch Warning] Timestamp did not advance for {symbol}. current_since_ms={current_since_ms}, new_since_ms={new_since_ms}. Breaking fetch loop to prevent stall.")
                break
            current_since_ms = new_since_ms
            
            # Safety break if we somehow exceed end_time significantly due to loop logic
            if current_since_ms > end_time + (60*60*1000): # an hour past end_time
                print(f"      [Fetch Warning] Exceeded target end_time significantly for {symbol}. Breaking.")
                break


    except ccxt.NetworkError as e:
        print(f"    [Fetch Error] NetworkError for {symbol}: {e}. Skipping.")
        return
    except ccxt.ExchangeError as e:
        print(f"    [Fetch Error] ExchangeError for {symbol}: {e}. Skipping.")
        return
    except Exception as e:
        print(f"    [Fetch Error] Unexpected error during fetch for {symbol}: {e}")
        # import traceback # Uncomment for full traceback if needed
        # traceback.print_exc()
        return

    if not all_ohlcv:
        print(f"      [Save] No OHLCV data fetched for {symbol} after attempts. Skipping save.")
        return

    try:
        df = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','volume'])
        if df.empty:
            print(f"      [Save] DataFrame is empty for {symbol} even after fetching data ({len(all_ohlcv)} records). Skipping save.")
            return
        
        # Filter data to be within the [start_time, end_time] range precisely if needed,
        # though fetch_ohlcv 'since' and loop logic should manage this.
        # df = df[df['timestamp'] >= start_time] # Ensure start_time
        # df = df[df['timestamp'] <= end_time] # Ensure end_time (approx, as last chunk might exceed slightly)

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol # Add symbol column

        if df.empty: # Check again after potential filtering
            print(f"      [Save] DataFrame became empty after time filtering for {symbol}. Skipping save.")
            return

        df.to_parquet(path)
        print(f"      [Save] Saved {len(df)} rows for {symbol} to {path}")
    except Exception as e:
        print(f"      [Save Error] Failed to create DataFrame or save parquet for {symbol}: {e}")
        # import traceback # Uncomment for full traceback if needed
        # traceback.print_exc()

with concurrent.futures.ThreadPoolExecutor(max_workers=fetch_workers) as ex:
    ex.map(fetch_and_cache, usdt_symbols)
print("[Step 1] Fetch complete\n")

# --- Step 1b: Load data + compute indicators ---
print("[Step 1b] Loading data and computing RSI, Bollinger Bands, VMA...")
market_data = {}
for path in glob.glob(os.path.join(DATA_DIR, '*.parquet')):
    sym = os.path.basename(path).replace('.parquet','').replace('_','/')
    df = pd.read_parquet(path)
    
    # Handle the case where timestamp might already be an index or have a different name
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    elif df.index.name != 'timestamp' and df.index.dtype != 'datetime64[ns]':
        # Try to find a datetime column to use as index
        datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        if datetime_cols:
            df = df.set_index(datetime_cols[0])
        else:
            print(f"Warning: No timestamp column found for {sym}, using default index")
    
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
    batch_size=256,
    trainer_kwargs={
        'max_epochs': 6,
        'accelerator': 'gpu' if device.type == 'cuda' else 'cpu',
        'devices': 1,
        'precision': effective_precision, # Use the determined precision
        'logger': False,
        'limit_val_batches': 0,
        # Remove num_workers from here
    }
)

predictor = estimator.train(
    training_data=training_ds,
    num_workers=4 
)
if hasattr(predictor, 'network'):
    predictor.network.to(device)
    if effective_precision == 16 and device.type == 'cuda': # Only for 16-bit GPU training
        print("    Converting predictor network to half precision for GPU inference.")
        predictor.network.half()
    else:
        print(f"    Predictor network on {device.type} with precision derived from training ({effective_precision}). Not converting to half.")
    predictor.network.eval()
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
print("[Step 3] Running inference...") # Generic message
if device.type == 'cuda' and effective_precision == 16:
    print(f"    Using AMP (float16) for {device.type} inference.")
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
        forecasts = list(predictor.predict(dataset, num_samples=1))
else: # CPU with float32, or other cases
    print(f"    Running inference on {device.type} (model precision from training: {effective_precision}). AMP for float16 not applied.")
    forecasts = list(predictor.predict(dataset, num_samples=1))

# --- Step 4: Map signals and backtest ---
price_df = pd.DataFrame({s: market_data[s]['close'] for s in market_data}).dropna(axis=1)

# --- Critical Data Check ---
if price_df.empty or not list(price_df.columns):
    print("[CRITICAL ERROR] price_df is empty or has no columns after loading market data.")
    print("    This means no valid data was loaded or processed in Step 1b.")
    print("    Please check the logs from [Step 1] and [Step 1b] for fetch/load errors.")
    print("    Backtesting cannot proceed. Setting up empty structures to prevent further crashes.")
    
    # Setup empty structures to allow script to finish without crashing
    symbols = []
    entries = {}
    exits = {}
    confidence_scores = {}
    position_sizes = {}
    
    # Create empty DataFrames
    entries_df = pd.DataFrame()
    exits_df = pd.DataFrame()
    position_size_df = pd.DataFrame()
    
    # Create a dummy portfolio object or skip portfolio steps
    # For simplicity, we'll create a minimal Portfolio object if possible or skip plots
    try:
        # Create a dummy price_df for portfolio creation if it doesn't have columns
        # This helps avoid crashes in Portfolio.from_signals if it expects some columns
        dummy_index = pd.date_range(start='1/1/2020', periods=1, freq='T')
        dummy_price_df = pd.DataFrame(index=dummy_index, columns=['DUMMY'])
        dummy_price_df['DUMMY'] = 100.0 # Fill with some dummy price data
        
        pf = vbt.Portfolio.from_signals(
            close=dummy_price_df, 
            entries=pd.DataFrame(index=dummy_index, columns=['DUMMY'], data=False),
            exits=pd.DataFrame(index=dummy_index, columns=['DUMMY'], data=False),
            init_cash=init_cash,
            size=0.0 # No size
        )
        stats = pf.stats()
        print("    Generated dummy portfolio stats as no data was available.")
    except Exception as e:
        print(f"    Could not create dummy portfolio: {e}. Stats and plots will be skipped.")
        stats = pd.Series(name="empty_stats") # Empty series for stats
        pf = None # Ensure pf is None if creation fails

else:
    # Original logic if price_df is fine
    n = min(map(len, price_df.values.T))
    price_df = price_df.iloc[-n:]
    symbols = price_df.columns.tolist()
    entries = {sym: np.zeros(len(price_df), dtype=bool) for sym in symbols}
    exits   = {sym: np.zeros(len(price_df), dtype=bool) for sym in symbols}

    # Store prediction confidence for dynamic position sizing
    confidence_scores = {sym: np.zeros(len(price_df)) for sym in symbols}

    for (symbol, idx), fc in zip(index_map, forecasts):
        if symbol not in symbols:
            continue
        rel_idx = idx - (len(market_data[symbol]) - len(price_df))
        if not 0 <= rel_idx < len(price_df):
            continue
        price = price_df[symbol].values[rel_idx]
        mean_pred = float(fc.mean[-1])
        pred_percentile = float(fc.quantile(0.975)[-1])
        
        # Calculate prediction confidence based on the mean prediction relative to current price
        # Normalize confidence between 0.0 and 1.0
        predicted_return = (mean_pred / price) - 1.0
        
        # Only consider positive expected returns
        if predicted_return > 0.05:  # 5% threshold for entry
            entries[symbol][rel_idx] = True
            
            # Store confidence score based on expected return (capped at 0.5 for safety)
            confidence = min(predicted_return, 0.5)
            confidence_scores[symbol][rel_idx] = confidence
            
            exit_point = rel_idx + prediction_length
            if exit_point < len(price_df):
                exits[symbol][exit_point] = True

    print("[Step 4] Calculating dynamic position sizes based on prediction confidence...")
    # Create dynamic position sizing based on prediction confidence
    position_sizes = {}
    for symbol in symbols:
        # Scale the confidence scores to position sizes between 0.01 (1%) and max_position_size (10%)
        # Higher confidence = larger position size, but never exceeding max_position_size
        position_sizes[symbol] = np.zeros(len(price_df))
        for i in range(len(price_df)):
            if entries[symbol][i]:
                # Calculate position size: min position size (1%) + scaled confidence (up to 9% more)
                # This ensures position size is between 1% and max_position_size
                position_sizes[symbol][i] = 0.01 + (confidence_scores[symbol][i] * (max_position_size - 0.01))
                # Ensure we don't exceed max position size
                position_sizes[symbol][i] = min(position_sizes[symbol][i], max_position_size)

    # Convert position sizes to DataFrame
    position_size_df = pd.DataFrame(position_sizes, index=price_df.index)

    print("[Step 4] Backtesting with vectorbt with dynamic sizing, SL/TP...")
    entries_df = pd.DataFrame(entries, index=price_df.index)
    exits_df   = pd.DataFrame(exits,   index=price_df.index)

    # Create portfolio with dynamic position sizing
    pf = vbt.Portfolio.from_signals(
        close=price_df,
        entries=entries_df,
        exits=exits_df,
        init_cash=init_cash,
        fees=0.002,
        slippage=0.001,
        size=position_size_df,  # Now using the dynamic position sizes dataframe
        size_type='Percent',
        cash_sharing=True
    )
    print("    Dynamic position sizing backtest complete")

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

# Save position sizing statistics
# Check if position_size_df is not empty before trying to get stats from it
if not position_size_df.empty:
    position_size_stats = {
        'mean': position_size_df.mean(),
        'median': position_size_df.median(),
        'min': position_size_df.min(),
        'max': position_size_df.max(),
        'std': position_size_df.std()
    }
    pd.DataFrame(position_size_stats).to_csv(os.path.join(DATA_DIR, "position_size_stats.csv"))
else:
    print("    Position size DataFrame is empty. Skipping position size stats saving.")


# Calculate and print Kelly Criterion
print("\n--- Kelly Criterion Calculation ---")
if isinstance(stats, pd.Series):
    # Assuming stats is a Series for the aggregate portfolio
    kelly_fraction = calculate_kelly_fraction(stats)
    print(f"Aggregate Portfolio Kelly Fraction: {kelly_fraction:.4f}")

    # If you want to apply this, you would typically adjust 'position_size'
    # for a *new* backtest run. For example:
    # if kelly_fraction > 0:
    #     new_position_size = kelly_fraction # Or a fraction of it, e.g., kelly_fraction / 2
    #     print(f"Recommended new position_size based on Kelly: {new_position_size:.4f}")
    # else:
    #     print("Kelly Criterion suggests not to bet or insufficient data.")
    
else:
    print("Could not determine the structure of portfolio_stats for Kelly Criterion calculation.")

print("--- End Kelly Criterion ---\n")


# Create plots
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot overall performance
if pf is not None: # Check if pf object exists
    try:
        pf.plot(ax=ax1)
        ax1.set_title('Portfolio Performance (or dummy if no data)')
    except Exception as e:
        print(f"    Error plotting portfolio performance: {e}")
        ax1.text(0.5, 0.5, "Portfolio plot unavailable (no data or error)", ha='center', va='center')
        ax1.set_title('Portfolio Performance')
else:
    ax1.text(0.5, 0.5, "Portfolio data unavailable", ha='center', va='center')
    ax1.set_title('Portfolio Performance')

# Plot position sizes over time
# Check if position_size_df has data and symbols are defined
if 'symbols' in locals() and symbols and not position_size_df.empty:
    for symbol in symbols:
        if symbol in position_size_df.columns: # Check if symbol column exists
            sizes = position_size_df[symbol][position_size_df[symbol] > 0]
            if len(sizes) > 0:
                ax2.scatter(sizes.index, sizes.values, label=symbol, alpha=0.7)
    ax2.set_ylim(0, max_position_size * 1.1)
    ax2.legend()
elif 'symbols' in locals() and not symbols: # No symbols, probably due to no data
    ax2.text(0.5, 0.5, "Position size data unavailable (no symbols processed)", ha='center', va='center')
else:
    ax2.text(0.5, 0.5, "Position size data unavailable", ha='center', va='center')
ax2.set_title('Dynamic Position Sizes Over Time')
ax2.set_ylabel('Position Size (%)')

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "backtest_results.png"))
plt.show()

print("Pipeline finished") 