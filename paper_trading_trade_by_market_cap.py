from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import requests
import time
import random

# --- Configuration ---
INITIAL_BALANCE = 1000.0
SUPPLY = 1_000_000_000          # assumed token supply
MARKETCAP_THRESHOLD = 100_000.0 # USD market cap trigger
INVEST_FRACTION = 0.10          # invest 10% of current balance per trade
HOLD_DURATION_MIN = 1           # hold period in minutes
FEE_RATE = 0.002                # 0.2% trading fee per execution (buy & sell)
BASE_URL = "https://frontend-api-v3.pump.fun"
PAGE_SIZE = 100                 # pagination size for live mode

# --- CSV Logging Setup ---
log_path = Path("data/trades")
log_path.mkdir(parents=True, exist_ok=True)
log_file = log_path / "all_trades.csv"
if not log_file.exists():
    headers = ["timestamp", "token", "action", "price", "amount_invested",
               "amount_returned", "gain_loss_usd", "gain_loss_pct", "net_worth"]
    pd.DataFrame(columns=headers).to_csv(log_file, index=False)


def log_trade(timestamp, token, action, price, amount_invested, amount_returned, gain_loss_usd, gain_loss_pct, net_worth):
    entry = {
        "timestamp": [timestamp.strftime("%Y-%m-%d %H:%M:%S")],
        "token": [token],
        "action": [action],
        "price": [round(price, 10)],
        "amount_invested": [round(amount_invested, 2) if amount_invested else 0],
        "amount_returned": [round(amount_returned, 2) if amount_returned else 0],
        "gain_loss_usd": [round(gain_loss_usd, 2) if gain_loss_usd else 0],
        "gain_loss_pct": [round(gain_loss_pct, 2) if gain_loss_pct else 0],
        "net_worth": [round(net_worth, 2)]
    }
    pd.DataFrame(entry).to_csv(log_file, mode='a', header=False, index=False)

class Trade:
    def __init__(self, token, buy_price, buy_time, amount_invested):
        self.token = token
        self.buy_price = buy_price
        self.buy_time = buy_time
        self.amount_invested = amount_invested
        # Calculate buy fee and net tokens bought
        self.fee_buy = self.amount_invested * FEE_RATE
        net_invest = self.amount_invested - self.fee_buy
        self.tokens_bought = net_invest / self.buy_price

    def sell(self, sell_price, sell_time):
        # Calculate gross proceeds and sell fee
        gross_return = self.tokens_bought * sell_price
        fee_sell = gross_return * FEE_RATE
        net_return = gross_return - fee_sell
        # Profit after fees = net_return - original amount_invested
        profit_loss_usd = net_return - self.amount_invested
        profit_loss_pct = (profit_loss_usd / self.amount_invested * 100) if self.amount_invested else 0
        return {
            "token": self.token,
            "buy_price": self.buy_price,
            "sell_price": sell_price,
            "amount_invested": self.amount_invested,
            "amount_returned": net_return,
            "gain_loss_usd": profit_loss_usd,
            "gain_loss_pct": profit_loss_pct
        }

class PaperTradingSimulator:
    """
    Paper Trading Simulator in mock and live modes.
    - mock: simulate tokens with random walk
    - live: fetch real token USD market caps with fees factored

    Usage:
      sim = PaperTradingSimulator(INITIAL_BALANCE, mode="live")
      sim.run(total_minutes=1)
    """
    def __init__(self, initial_balance, mode="live", tokens=None):
        self.balance = initial_balance
        self.active_trade = None
        self.mode = mode
        if mode == "mock":
            if not tokens:
                raise ValueError("In mock mode, you must provide a tokens list.")
            self.tokens = tokens
            init_price = (MARKETCAP_THRESHOLD * 0.9) / SUPPLY
            self.prices = {t: init_price for t in tokens}

    def get_current_prices(self):
        if self.mode == "mock":
            for t in self.tokens:
                drift = 0.000000001
                noise = random.uniform(-0.000000002, 0.000000005)
                self.prices[t] = max(self.prices[t] + drift + noise, 0)
            return self.prices.copy()

        prices = {}
        offset = 0
        total_fetched = 0
        while True:
            try:
                resp = requests.get(
                    f"{BASE_URL}/coins",
                    params={"offset": offset, "limit": PAGE_SIZE}
                )
                resp.raise_for_status()
                coins = resp.json()
            except Exception as e:
                print(f"Error fetching /coins page at offset {offset}: {e}")
                break
            if not isinstance(coins, list) or not coins:
                break
            total_fetched += len(coins)
            for coin in coins:
                mint = coin.get("mint")
                usd_mc = coin.get("usd_market_cap") or coin.get("market_cap")
                if mint and usd_mc is not None:
                    try:
                        usd_mc = float(usd_mc)
                        prices[mint] = usd_mc / SUPPLY
                    except:
                        continue
            offset += len(coins)
        print(f"Total live tokens fetched: {total_fetched}, usable prices: {len(prices)}")
        return prices

    def simulate_minute(self, current_time, prices):
        print(f"[{current_time.strftime('%H:%M:%S')}] Scanning {len(prices)} tokens")

        # SELL logic
        if self.active_trade:
            t = self.active_trade
            if current_time >= t.buy_time + timedelta(minutes=HOLD_DURATION_MIN):
                res = t.sell(prices.get(t.token, t.buy_price), current_time)
                self.balance += res["amount_returned"]
                # Log sell with net worth
                log_trade(current_time, t.token, "sell",
                          res["sell_price"], 0,
                          res["amount_returned"],
                          res["gain_loss_usd"], res["gain_loss_pct"], self.balance)
                sign = "+" if res["gain_loss_pct"] >= 0 else ""
                print(f"BOUGHT {t.token} at ${t.buy_price:.6f} | "
                      f"SOLD at ${res['sell_price']:.6f} | "
                      f"{sign}{res['gain_loss_pct']:.2f}% | P/L "
                      f"{sign}${abs(res['gain_loss_usd']):.2f}")
                print(f"Total Net Worth: ${self.balance:.2f}")
                self.active_trade = None

        # BUY logic
        if not self.active_trade:
            for token, price in prices.items():
                if price * SUPPLY >= MARKETCAP_THRESHOLD:
                    invest_amt = self.balance * INVEST_FRACTION
                    if invest_amt <= 0:
                        break
                    fee_buy = invest_amt * FEE_RATE
                    total_cost = invest_amt + fee_buy
                    self.balance -= total_cost
                    self.active_trade = Trade(token, price, current_time, invest_amt)
                    log_trade(current_time, token, "buy",
                              price, invest_amt, 0, 0, 0, self.balance)
                    print(f"Buying {token} at ${price:.6f} (Invested ${invest_amt:.2f}, Fee ${fee_buy:.2f})")
                    print(f"Remaining Balance: ${self.balance:.2f}")
                    break

    def run(self, total_minutes=None):
        current_time = datetime.now()
        count = 0
        while total_minutes is None or count < total_minutes:
            prices = self.get_current_prices()
            self.simulate_minute(current_time, prices)
            current_time += timedelta(minutes=1)
            count += 1
            time.sleep(60)

# Usage example:
# sim = PaperTradingSimulator(INITIAL_BALANCE, mode="live")
# sim.run(total_minutes=1)

if __name__ == '__main__':
    sim = PaperTradingSimulator(INITIAL_BALANCE, mode="live")
    sim.run(total_minutes=5)