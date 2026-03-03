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

if __name__ == "__main__":
    main()
