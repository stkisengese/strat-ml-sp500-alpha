import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import joblib
from cv import plot_strategy_performance

def signal_to_positions(signal: pd.Series, k: int = 10) -> pd.DataFrame:
    """
    Translates ML probabilities into trading positions using a rank-based stock-picking strategy.
    Long the top-k stocks, short the bottom-k stocks.
    Maintains a total absolute exposure of $1 per day.
    """
    daily_positions = []
    for date, group in signal.groupby(level="date"):
        if len(group) < 2 * k:
            continue
        
        # Rank tickers by probability and select extremes
        ranked = group.sort_values(ascending=False)
        long_tickers = ranked.head(k).index
        short_tickers = ranked.tail(k).index
        
        # Initialize zero positions for the current day
        pos = pd.Series(0.0, index=group.index, name="position")
        
        # Distribute $1 equally across 2k positions: $0.50 long, $0.50 short
        pos.loc[long_tickers] = 1 / (2 * k)   
        pos.loc[short_tickers] = -1 / (2 * k) 
        
        daily_positions.append(pos)
    
    return pd.concat(daily_positions).sort_index().to_frame("position")

def calculate_cumulative_pnl(positions: pd.DataFrame, returns: pd.Series) -> pd.Series:
    """
    Computes daily strategy PnL by aligning positions with forward returns.
    """
    # Daily PnL = Sum(Position_D * Return_D+1_to_D+2)
    daily_pnl = (positions["position"] * returns).groupby(level="date").sum()
    return daily_pnl.cumsum()

def main():
    print("--- Starting Strategy Backtesting & Performance Analysis ---")

    # 1. Load walk-forward ML signals (training period)
    signal_train_path = "results/selected-model/ml_signal.csv"
    if not os.path.exists(signal_train_path):
        raise FileNotFoundError(f"Signal file not found at {signal_train_path}. Run create_signal.py first.")
    ml_signal_train = pd.read_csv(signal_train_path, parse_dates=["date"]).set_index(["date", "Name"])["ml_signal"]

    # 2. Generate out-of-sample test signals (2017+) using the final trained model
    X_test = pd.read_csv("data/processed/X_test.csv", parse_dates=["date"]).set_index(["date", "Name"])
    pipe = joblib.load("results/selected-model/selected_model.pkl")
    
    test_proba = pipe.predict_proba(X_test.values)[:, 1]
    ml_signal_test = pd.Series(test_proba, index=X_test.index, name="ml_signal")

    # Concatenate train and test signals for a full-period analysis
    full_signal = pd.concat([ml_signal_train, ml_signal_test]).sort_index()

    # 3. Load actual pricing data to compute benchmark returns
    # The signal on day D predicts the return observed between D+1 and D+2.
    raw = pd.read_csv("data/all_stocks_5yr.csv", parse_dates=["date"]).set_index(["date", "Name"]).sort_index()
    raw["daily_return"] = raw.groupby(level="Name")["close"].transform(lambda x: np.log(x / x.shift(1)))
    
    # We shift the returns back by 2 to align the return (D+1 -> D+2) with the row for day D.
    target_returns = raw.groupby(level="Name")["daily_return"].shift(-2)

    # 4. Generate positions and compute performance
    print("Generating positions and calculating cumulative PnL...")
    positions = signal_to_positions(full_signal, k=10)
    cum_pnl = calculate_cumulative_pnl(positions, target_returns.reindex(positions.index))

    # 5. Load S&P 500 benchmark data for comparison
    spx = pd.read_csv("data/HistoricalPrices.csv", parse_dates=["Date"], date_format="%m/%d/%y")
    spx = spx.rename(columns={"Date": "date", " Close": "close"}).set_index("date").sort_index()
    spx_daily_return = np.log(spx["close"] / spx["close"].shift(1))
    spx_cum_pnl = spx_daily_return.cumsum().reindex(cum_pnl.index).ffill()

    # 6. Generate visualizations
    print("Generating performance plots...")
    plot_strategy_performance(cum_pnl, spx_cum_pnl)

    print("\n Backtesting COMPLETE.")
    print("   • Performance plot saved to: results/strategy/strategy.png")

if __name__ == "__main__":
    main()
