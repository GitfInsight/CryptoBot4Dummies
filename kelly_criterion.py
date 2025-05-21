import pandas as pd
import numpy as np

def calculate_kelly_fraction(portfolio_stats: pd.Series) -> float:
    """
    Calculates the Kelly Criterion fraction based on portfolio statistics.

    The Kelly Criterion formula used is: K % = W – [(1 – W) / R]
    where:
    - W is the probability of a win.
    - R is the win/loss ratio (average win / average loss).

    Args:
        portfolio_stats: A pandas Series containing portfolio statistics
                         from vectorbt, specifically requiring:
                         'Win Rate [%]'
                         'Avg Winning Trade [%]'
                         'Avg Losing Trade [%]'
                         It also implicitly depends on 'Total Closed Trades'
                         for the meaningfulness of these stats.

    Returns:
        The Kelly fraction, capped between 0.0 and 1.0.
        Returns 0.0 if required statistics are missing, NaN, or if the
        scenario does not warrant a bet (e.g., no wins, or R is zero
        with non-100% win rate).
    """
    required_stats = ['Win Rate [%]', 'Avg Winning Trade [%]', 'Avg Losing Trade [%]']
    if not all(stat in portfolio_stats for stat in required_stats):
        # print("Warning: Missing required statistics for Kelly Criterion calculation.") # Consider logging
        return 0.0

    win_rate_pct = portfolio_stats['Win Rate [%]']
    avg_win_pct = portfolio_stats['Avg Winning Trade [%]']
    # Avg Losing Trade [%] from vectorbt is typically negative
    avg_loss_pct = portfolio_stats['Avg Losing Trade [%]']

    # Check if essential stats are NaN (e.g., if there are no trades)
    if pd.isna(win_rate_pct) or pd.isna(avg_win_pct):
        # print("Warning: Win rate or average win is NaN.") # Consider logging
        return 0.0

    W = win_rate_pct / 100.0  # Winning probability

    # If Win Rate is 100%
    if W == 1.0:
        if avg_win_pct > 0:
            return 1.0  # Bet full capital if 100% win rate and positive average win
        else:
            return 0.0 # Don't bet if 100% win rate but wins are zero or negative

    # If Win Rate is 0%
    if W == 0.0:
        return 0.0 # Don't bet if 0% win rate

    # Handle cases with no losses or zero average loss
    if pd.isna(avg_loss_pct) or avg_loss_pct == 0:
        # If there are no losses (avg_loss_pct is NaN or 0) and W < 1.0:
        # Kelly Criterion implies R is infinite, so K = W.
        # This assumes that non-winning trades are breakeven and winning trades are positive.
        if avg_win_pct > 0:
            return W
        else:
            # If avg_win_pct is also zero or negative, don't bet.
            return 0.0

    # R is the win/loss ratio (average win / absolute average loss)
    # Ensure avg_loss_pct is used as a positive value for R calculation
    if abs(avg_loss_pct) == 0: # Should have been caught by avg_loss_pct == 0, but defensive
        if avg_win_pct > 0: # Infinite R if losses are truly zero
            return W
        else:
            return 0.0

    R = avg_win_pct / abs(avg_loss_pct)

    # If average wins are zero or negative, R will be <= 0. Don't bet.
    if R <= 0:
        return 0.0

    # Calculate Kelly fraction
    kelly_fraction = W - ((1 - W) / R)

    # Clamp Kelly fraction between 0 (no bet) and 1 (full capital)
    # A negative Kelly fraction means do not bet.
    return max(0.0, min(kelly_fraction, 1.0))

if __name__ == '__main__':
    # Example Usage:
    # Scenario 1: Profitable strategy
    stats1 = pd.Series({
        'Win Rate [%]': 60.0,
        'Avg Winning Trade [%]': 20.0,
        'Avg Losing Trade [%]': -10.0,
        'Total Closed Trades': 100
    })
    kelly1 = calculate_kelly_fraction(stats1)
    print(f"Scenario 1 Kelly Fraction: {kelly1:.4f}") # Expected: 0.4000

    # Scenario 2: Losing strategy (Kelly should be 0)
    stats2 = pd.Series({
        'Win Rate [%]': 40.0,
        'Avg Winning Trade [%]': 5.0,
        'Avg Losing Trade [%]': -10.0,
        'Total Closed Trades': 100
    })
    kelly2 = calculate_kelly_fraction(stats2)
    print(f"Scenario 2 Kelly Fraction: {kelly2:.4f}") # Expected: 0.0000 (0.4 - (0.6 / 0.5) = 0.4 - 1.2 = -0.8 -> 0)

    # Scenario 3: 100% Win rate
    stats3 = pd.Series({
        'Win Rate [%]': 100.0,
        'Avg Winning Trade [%]': 5.0,
        'Avg Losing Trade [%]': np.nan, # No losing trades
        'Total Closed Trades': 100
    })
    kelly3 = calculate_kelly_fraction(stats3)
    print(f"Scenario 3 Kelly Fraction (100% Win Rate): {kelly3:.4f}") # Expected: 1.0000

    # Scenario 4: No losses recorded, win rate < 100%
    stats4 = pd.Series({
        'Win Rate [%]': 70.0,
        'Avg Winning Trade [%]': 10.0,
        'Avg Losing Trade [%]': 0.0, # or np.nan
        'Total Closed Trades': 100
    })
    kelly4 = calculate_kelly_fraction(stats4)
    print(f"Scenario 4 Kelly Fraction (No Losses, WR < 100%): {kelly4:.4f}") # Expected: 0.7000 (W)
    
    stats4b = pd.Series({
        'Win Rate [%]': 70.0,
        'Avg Winning Trade [%]': 10.0,
        'Avg Losing Trade [%]': np.nan, 
        'Total Closed Trades': 100
    })
    kelly4b = calculate_kelly_fraction(stats4b)
    print(f"Scenario 4b Kelly Fraction (NaN Losses, WR < 100%): {kelly4b:.4f}") # Expected: 0.7000 (W)


    # Scenario 5: Zero average win
    stats5 = pd.Series({
        'Win Rate [%]': 50.0,
        'Avg Winning Trade [%]': 0.0,
        'Avg Losing Trade [%]': -10.0,
        'Total Closed Trades': 100
    })
    kelly5 = calculate_kelly_fraction(stats5)
    print(f"Scenario 5 Kelly Fraction (Zero Avg Win): {kelly5:.4f}") # Expected: 0.0000

    # Scenario 6: Missing stats
    stats6 = pd.Series({
        'Win Rate [%]': 50.0,
        # 'Avg Winning Trade [%]': 0.0, # Missing
        'Avg Losing Trade [%]': -10.0,
        'Total Closed Trades': 100
    })
    kelly6 = calculate_kelly_fraction(stats6)
    print(f"Scenario 6 Kelly Fraction (Missing Stats): {kelly6:.4f}") # Expected: 0.0000
    
    # Scenario 7: All stats NaN (e.g. no trades)
    stats7 = pd.Series({
        'Win Rate [%]': np.nan,
        'Avg Winning Trade [%]': np.nan,
        'Avg Losing Trade [%]': np.nan,
        'Total Closed Trades': 0
    })
    kelly7 = calculate_kelly_fraction(stats7)
    print(f"Scenario 7 Kelly Fraction (All NaN Stats): {kelly7:.4f}") # Expected: 0.0000
    
    # Scenario 8: Win rate 100% but avg win is 0%
    stats8 = pd.Series({
        'Win Rate [%]': 100.0,
        'Avg Winning Trade [%]': 0.0,
        'Avg Losing Trade [%]': np.nan, 
        'Total Closed Trades': 100
    })
    kelly8 = calculate_kelly_fraction(stats8)
    print(f"Scenario 8 Kelly Fraction (100% WR, 0% Avg Win): {kelly8:.4f}") # Expected: 0.0000 