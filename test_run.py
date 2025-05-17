"""Back‑test:
Keeps rows even if some market‑cap values are missing (they simply won't trigger
entries). Requires:
    • ./data/<pair>.parquet files with 'close' and 'market_cap'
    • binance_coingecko_market_caps.parquet mapping symbols → parquet
"""

from datetime import timedelta
import os

import pandas as pd
import pyarrow.parquet as pq
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType

# ────────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────────
DATA_DIR = "./data"
MASTER_PARQUET = "binance_coingecko_market_caps.parquet"

TARGET_MCAP_USD = 39_060_000
POSITION_PCT = 0.10
HOLD_MINUTES = 15

# ────────────────────────────────────────────────────────────────────────────────
# 1 Load data
# ────────────────────────────────────────────────────────────────────────────────
if not os.path.exists(MASTER_PARQUET):
    raise FileNotFoundError("Missing master parquet. Run data‑prep first.")

master = pd.read_parquet(MASTER_PARQUET)
files = {
    row.symbol: row.klines_parquet
    for row in master.itertuples()
    if isinstance(row.klines_parquet, str) and os.path.exists(row.klines_parquet)
}
if not files:
    raise RuntimeError("No klines parquet files found.")

close_cols, mcap_cols = {}, {}
for sym, path in files.items():
    df = pq.read_table(path).to_pandas()
    close_cols[sym] = df["close"]
    mcap_cols[sym] = df["market_cap"]

price = pd.concat(close_cols, axis=1)
mcap = pd.concat(mcap_cols, axis=1)
price.index.name = "timestamp"

# Align indices (no row dropping)
common_index = price.index.union(mcap.index)
price = price.reindex(common_index)
mcap = mcap.reindex(common_index)

# ────────────────────────────────────────────────────────────────────────────────
# 2 Weights: adaptive trigger
#    If TARGET_MCAP_USD never hits for a coin, fall back to that coin’s 5th‑percentile
#    market cap as a dynamic threshold. Then buy, hold 15 m, 10 % equity.
# ────────────────────────────────────────────────────────────────────────────────
weights = pd.DataFrame(0.0, index=price.index, columns=price.columns)

for sym in price.columns:
    series = mcap[sym]
    if series.isna().all():
        continue  # skip coins with no mcap data at all

    # primary rule
    entry_mask = series <= TARGET_MCAP_USD

    # if never triggered, use adaptive percentile
    if not entry_mask.any():
        dynamic_thresh = np.nanpercentile(series, 5)
        print(f"ℹ️  {sym}: static $980k never hit, using 5th‑percentile ≈ {dynamic_thresh:,.0f} USD")
        entry_mask = series <= dynamic_thresh

    for ts in entry_mask[entry_mask].index:
        end_ts = ts + timedelta(minutes=HOLD_MINUTES - 1)
        weights.loc[ts:end_ts, sym] = POSITION_PCT

# keep total allocation ≤ 100 %
row_tot = weights.sum(axis=1)
excess = row_tot > 1
if excess.any():
    weights.loc[excess] = weights.loc[excess].div(row_tot[excess], axis=0)

# Quick sanity: show how many orders will be generated
print("Total non‑zero weight rows:", (weights.sum(axis=1) > 0).sum())

# ────────────────────────────────────────────────────────────────────────────────
weights = pd.DataFrame(0.0, index=price.index, columns=price.columns)

for sym in price.columns:
    # NaNs will evaluate as False
    entry_times = mcap.index[mcap[sym] <= TARGET_MCAP_USD]
    for ts in entry_times:
        end_ts = ts + timedelta(minutes=HOLD_MINUTES - 1)
        weights.loc[ts:end_ts, sym] = POSITION_PCT

# keep total allocation ≤ 100 %
row_tot = weights.sum(axis=1)
excess = row_tot > 1
if excess.any():
    weights.loc[excess] = weights.loc[excess].div(row_tot[excess], axis=0)

# ────────────────────────────────────────────────────────────────────────────────
# 3 Run Portfolio
# ────────────────────────────────────────────────────────────────────────────────
portfolio = vbt.Portfolio.from_orders(
    close=price,
    size=weights,
    size_type=SizeType.TargetPercent,
    cash_sharing=True,
    fees=0.001,
    freq="1T",
)

print(portfolio.stats())

fig = portfolio.plot()  # default benchmark disabled

# add BTC benchmark line if BTC data exists
if 'BTC' in price.columns:
    btc_equity = price['BTC'] / price['BTC'].iloc[0] * portfolio.initial_cash
    fig.add_scatter(x=btc_equity.index, y=btc_equity.values, name='BTC Benchmark', line=dict(color='gray', dash='dot'))

fig.show()
