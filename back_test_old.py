import requests
import pandas as pd
import vectorbt as vbt
import os
from datetime import datetime, timedelta

# ─── CONFIG ─────────────────────────────────────────────────────────────
BASE_URL   = "https://frontend-api-v3.pump.fun"
DATA_DIR   = "data"
RAW_DIR    = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

DAYS_BACK    = 14       # backtest window in days
MAX_RECORDS  = 20_000   # max bars to fetch per mint
TIMEFRAME    = 1        # minutes per bar
API_LIMIT    = 1_000    # max bars per call
INIT_CASH    = 1_000
ALLOCATION   = 0.10
HOLD_MIN     = 15       # minutes to hold positions
QUANTILE     = 0.90     # dynamic threshold percentile

# ─── SOLANA RPC CONFIG ───────────────────────────────────────────────────
SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"

# ─── HELPERS ─────────────────────────────────────────────────────────────

def fetch_all_tokens():
    tokens, offset = [], 0
    while True:
        res = requests.get(
            f"{BASE_URL}/coins", params={"offset": offset, "limit": 100}
        )
        res.raise_for_status()
        batch = res.json()
        if not batch:
            break
        tokens += batch
        offset += len(batch)
    return [t['mint'] for t in tokens if 'mint' in t]


def fetch_supply_onchain(mint):
    """
    Query Solana RPC getTokenSupply for current total supply of a mint.
    Returns float supply, or None on failure.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenSupply",
        "params": [mint]
    }
    try:
        r = requests.post(SOLANA_RPC_URL, json=payload)
        r.raise_for_status()
        resp = r.json()
        # uiAmount is human-readable supply
        supply = resp.get("result", {}).get("value", {}).get("uiAmount")
        return float(supply) if supply is not None else None
    except Exception:
        return None


def fetch_token_supply_map():
    """
    Build a map of mint -> total supply by querying Solana RPC.
    """
    supply_map = {}
    mints = fetch_all_tokens()
    for mint in mints:
        supply = fetch_supply_onchain(mint)
        if supply:
            supply_map[mint] = supply
    return supply_map


def fetch_all_candles(mint):
    bars, offset = [], 0
    cutoff_ts = (datetime.utcnow() - timedelta(days=DAYS_BACK)).timestamp()
    while True:
        params = {"timeframe": TIMEFRAME, "offset": offset, "limit": API_LIMIT}
        res = requests.get(f"{BASE_URL}/candlesticks/{mint}", params=params)
        res.raise_for_status()
        data = res.json()
        if not data:
            break
        bars.extend(data)
        if len(bars) >= MAX_RECORDS or min(d['timestamp'] for d in data) <= cutoff_ts:
            break
        offset += len(data)
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df


def fetch_market_caps(supply_map):
    cutoff = datetime.utcnow() - timedelta(days=DAYS_BACK)
    markets = {}
    for mint, supply in supply_map.items():
        raw_file = os.path.join(RAW_DIR, f"{mint}.parquet")
        if os.path.exists(raw_file):
            raw_df = pd.read_parquet(raw_file)
        else:
            raw_df = fetch_all_candles(mint)
            if raw_df.empty:
                continue
            raw_df.to_parquet(raw_file)

        window = raw_df[raw_df.index >= cutoff]
        if window.empty or 'close' not in window.columns:
            continue

        window['market_cap'] = window['close'] * supply
        markets[mint] = window[['market_cap']]
    return markets


def trading_signals_all(markets, quantile=QUANTILE, hold_minutes=HOLD_MIN):
    entries, exits = {}, {}
    for mint, df in markets.items():
        thresh = df['market_cap'].quantile(quantile)
        e = df['market_cap'] >= thresh
        e &= ~e.shift(1, fill_value=False)
        entries[mint] = e
        x = pd.Series(False, index=df.index)
        for t in df.index[e]:
            idx = df.index.searchsorted(t + timedelta(minutes=hold_minutes), side='left')
            if idx < len(df):
                x.iat[idx] = True
        exits[mint] = x
    return entries, exits


def backtest_all():
    supply_map = fetch_token_supply_map()
    print(f"Fetched {len(supply_map)} supplies via on-chain RPC")

    markets = fetch_market_caps(supply_map)
    print(f"Loaded {len(markets)} symbols with market-cap data")

    if markets:
        first = next(iter(markets))
        df0 = markets[first]
        print(f"Sample {first}: {df0.index.min()} to {df0.index.max()}")
        print(f"  90th pct cap: {df0['market_cap'].quantile(QUANTILE)}")
        print(f"  Entries: {(df0['market_cap'] >= df0['market_cap'].quantile(QUANTILE)).sum()}\n")

    entries, exits = trading_signals_all(markets)

    prices = {m: caps['market_cap'] / supply_map[m] for m, caps in markets.items()}
    price_df = pd.concat(prices, axis=1).ffill().bfill().astype(float)
    price_df.columns.name = 'asset'

    ent_df = pd.concat(entries, axis=1).reindex(index=price_df.index, fill_value=False).astype(bool)
    ext_df = pd.concat(exits, axis=1).reindex(index=price_df.index, fill_value=False).astype(bool)
    ent_df.columns.name = ext_df.columns.name = 'asset'

    pf = vbt.Portfolio.from_signals(
        close=price_df,
        entries=ent_df,
        exits=ext_df,
        init_cash=INIT_CASH,
        freq=f'{TIMEFRAME}min',
        fees=0.005,
        size=ALLOCATION,
        size_type='percent',
        cash_sharing=True,
    )
    stats = pf.stats()
    print(f"Total trades: {stats['Total Trades']}")
    return pf

if __name__ == "__main__":
    pf = backtest_all()
    print("\nPortfolio stats:\n", pf.stats())
    pf.plot().show()