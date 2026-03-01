import pandas as pd
import numpy as np
import pandas_ta as ta
import os

def compute_features(group):
    # Bollinger Bands
    # length=20, std=2
    bbands = group.ta.bbands(length=20, std=2)
    if bbands is None:
        return group
    # print(bbands.columns) # Debug
    # pandas_ta bbands returns: BBL_20_2.0 (lower), BBM_20_2.0 (mid), BBU_20_2.0 (upper), BBB_20_2.0 (bandwidth), BBP_20_2.0 (%B)
    group['bb_percent'] = bbands.iloc[:, 4] # BBP is usually the 5th column
    group['bb_width'] = bbands.iloc[:, 3]   # BBB is usually the 4th column

    # RSI
    # length=14
    group['rsi'] = group.ta.rsi(length=14)
    group['rsi_change'] = group['rsi'].diff()

    # MACD
    # fast=12, slow=26, signal=9
    macd = group.ta.macd(fast=12, slow=26, signal=9)
    if macd is not None:
        # returns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        group['macd'] = macd.iloc[:, 0] / group['close']
        group['macd_signal'] = macd.iloc[:, 2] / group['close']
        group['macd_hist'] = macd.iloc[:, 1] / group['close']

    return group

def main():
     # Combine features
    features = ['bb_percent', 'bb_width', 'rsi', 'rsi_change', 'macd', 'macd_signal', 'macd_hist']


    print("Loading data...")
    df = pd.read_csv('data/all_stocks_5yr.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print("Setting MultiIndex and sorting...")
    df = df.sort_values(['Name', 'date']).set_index(['date', 'Name'])

    # Target calculation
    # On day D, target is sign(return(D+1, D+2))
    # log_return[t] = log(close[t] / close[t-1]) -> return from t-1 to t
    # return(D+1, D+2) is log_return[D+2]
    # So target[D] = sign(log_return[D+2])
    print("Computing target...")
    df['log_return'] = df.groupby(level='Name')['close'].transform(lambda x: np.log(x / x.shift(1)))
    # We use shift(-2) to get return(D+1, D+2) on day D
    df['target'] = df.groupby(level='Name')['log_return'].transform(lambda x: np.sign(x.shift(-2)))

    print("Computing features for all data...")
    df = df.groupby(level='Name', group_keys=False).apply(compute_features)
    print("Dropping NaNs...")
    df = df.dropna(subset=features + ['target']) # Drop rows where features or target are NaN
    
if __name__ == "__main__":
    main()
