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
