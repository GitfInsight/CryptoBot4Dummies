import os
import time
from datetime import datetime, timedelta
from itertools import islice

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

# ────────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────────
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
DATA_DIR = "./data"      # minute‑klines
os.makedirs(DATA_DIR, exist_ok=True)

pd.options.display.float_format = "{:,}".format  # ditch scientific notation

# ────────────────────────────────────────────────────────────────────────────────
# Helper utils
# ────────────────────────────────────────────────────────────────────────────────

def _chunks(iterable, n):
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk

# ────────────────────────────────────────────────────────────────────────────────
# Binance (pairs + klines)
# ────────────────────────────────────────────────────────────────────────────────

def get_binance_us_symbols():
    """Return every USDT trading pair listed on Binance US (e.g. BTCUSDT)."""
    r = requests.get("https://api.binance.us/api/v3/exchangeInfo", timeout=10)
    r.raise_for_status()
    symbols = r.json()["symbols"]
    return [s["symbol"] for s in symbols if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"]


def fetch_and_save_klines(pairs, supply_map, interval="1m", days=14):
    """Download klines and compute minute‑level market_cap = close * circulating_supply.

    Args:
        pairs (list[str]): Binance pairs like BTCUSDT
        supply_map (dict): {SYMBOL: circulating_supply}
        interval (str): Binance kline interval
        days (int): look‑back window

    Returns:
        dict: {PAIR: parquet_path_written}
    """
    base_url = "https://api.binance.us"
    out = {}

    for pair in pairs:
        sym = pair.replace("USDT", "").upper()
        supply = supply_map.get(sym)
        if supply is None or pd.isna(supply):
            print(f"⚠️  No circulating_supply for {sym}; market_cap will be NA")
        print(f"⏳  Fetching klines {pair}…")
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 24 * 60 * 60 * 1000
        rows = []
        while start_ms < end_ms:
            try:
                resp = requests.get(
                    f"{base_url}/api/v3/klines",
                    params={
                        "symbol": pair,
                        "interval": interval,
                        "startTime": start_ms,
                        "endTime": end_ms,
                        "limit": 1000,
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                rows.extend(batch)
                start_ms = batch[-1][0] + 1
                time.sleep(0.05)
            except Exception as e:
                print(f"❌  Error klines {pair}: {e}")
                break

        if not rows:
            print(f"⚠️  No klines for {pair}")
            continue

        df = pd.DataFrame(
            rows,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "qav", "num_trades", "taker_base",
                "taker_quote", "ignore",
            ],
        )
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        # compute minute‑level market cap via supply * close_price
        if supply is not None and not pd.isna(supply):
            df["market_cap"] = df["close"] * supply
        else:
            df["market_cap"] = pd.NA

        fname = os.path.abspath(os.path.join(DATA_DIR, f"{pair.lower()}_{interval}_last{days}d.parquet"))
        pq.write_table(pa.Table.from_pandas(df), fname)
        out[pair.upper()] = fname
        print(f"✅  {fname}  ({len(df):,} rows | market_cap computed)")
    return out

# ────────────────────────────────────────────────────────────────────────────────
# CoinGecko (symbol map, current market data)
# ────────────────────────────────────────────────────────────────────────────────

_SYMBOL_TO_ID = None

def _load_symbol_map():
    global _SYMBOL_TO_ID
    if _SYMBOL_TO_ID is None:
        print("🔄  Loading CoinGecko symbol list…")
        resp = requests.get(f"{COINGECKO_BASE}/coins/list", timeout=30)
        resp.raise_for_status()
        _SYMBOL_TO_ID = {c["symbol"].upper(): c["id"] for c in resp.json()}
    return _SYMBOL_TO_ID


def get_coingecko_ids(symbols):
    mapping = _load_symbol_map()
    return {s: mapping.get(s) for s in symbols}


def fetch_current_market_data(ids):
    """Return list of market data dicts from /coins/markets for given coin IDs."""
    results = []
    for chunk in _chunks(list(ids), 250):
        r = requests.get(
            f"{COINGECKO_BASE}/coins/markets",
            params={"vs_currency": "usd", "ids": ",".join(chunk), "price_change_percentage": "24h"},
            timeout=20,
        )
        if r.status_code == 429:
            print("⏳  Markets rate‑limited — sleep 60 s…")
            time.sleep(60)
            r = requests.get(
                f"{COINGECKO_BASE}/coins/markets",
                params={"vs_currency": "usd", "ids": ",".join(chunk), "price_change_percentage": "24h"},
                timeout=20,
            )
        r.raise_for_status()
        results.extend(r.json())
    return results

# ────────────────────────────────────────────────────────────────────────────────
# Master DataFrame
# ────────────────────────────────────────────────────────────────────────────────

def build_master_dataframe(pairs, market_data, kline_paths):
    md_df = pd.DataFrame(market_data)
    md_df["symbol"] = md_df["symbol"].str.upper()

    base_syms = [p.replace("USDT", "") for p in pairs]
    df = pd.DataFrame({"binance_pair": pairs, "base_symbol": base_syms}).merge(
        md_df, how="left", left_on="base_symbol", right_on="symbol"
    )

    df["klines_parquet"] = df["binance_pair"].map(kline_paths)

    keep = [
        "binance_pair", "name", "symbol", "current_price", "market_cap", "klines_parquet",
        "circulating_supply", "total_supply", "max_supply",
    ]
    return df[keep].sort_values("market_cap", ascending=False)

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DAYS = 14

    # 1 Binance pairs
    pairs = get_binance_us_symbols()

    # 2 CoinGecko IDs & current market data (includes circulating_supply)
    base_syms = [p.replace("USDT", "") for p in pairs]
    symbol_to_id = get_coingecko_ids(base_syms)
    market_data = fetch_current_market_data([cid for cid in symbol_to_id.values() if cid])

    # build supply map for quick lookup
    supply_map = {d["symbol"].upper(): d.get("circulating_supply") for d in market_data}

    # 3 Minute klines with computed market_cap column
    kline_paths = fetch_and_save_klines(pairs, supply_map, days=DAYS)

    # 4 Merge summary
    master_df = build_master_dataframe(pairs, market_data, kline_paths)
    master_path = "binance_coingecko_market_caps.parquet"
    master_df.to_parquet(master_path, index=False)

    print(master_df.head())
    print(f"📝  Master saved → {master_path}")
