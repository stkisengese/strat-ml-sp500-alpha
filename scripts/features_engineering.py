import pandas as pd
import numpy as np
import pandas_ta as ta
import os

def compute_features(group):
    """All features are computed on the close price series, then merged back to the group DataFrame."""
    close = group['close']

    # Bollinger Bands (20-day, 2 std)
    bbands = ta.bbands(close, length=20, std=2)
    if bbands is not None:
        group['bb_percent'] = bbands['BBP_20_2.0_2.0'].values  # position within bands [0,1]
        group['bb_width']   = bbands['BBB_20_2.0_2.0'].values  # volatility regime

    # RSI (14-day) — bounded [0, 100], no normalisation needed
    rsi = ta.rsi(close, length=14)
    if rsi is not None:
        group['rsi']        = rsi.values
        group['rsi_change'] = rsi.diff().values  # momentum acceleration

    # MACD (12, 26, 9) — normalised by close to make comparable across tickers
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None:
        group['macd']        = (macd_df['MACD_12_26_9']  / close).values
        group['macd_signal'] = (macd_df['MACDs_12_26_9'] / close).values
        group['macd_hist']   = (macd_df['MACDh_12_26_9'] / close).values

    return group


def main():
    # Define features list first — used in dropna and downstream scripts
    features = ['bb_percent', 'bb_width', 'rsi', 'rsi_change',
                'macd', 'macd_signal', 'macd_hist']

    print("Loading data...")
    df = pd.read_csv('data/all_stocks_5yr.csv')
    df['date'] = pd.to_datetime(df['date'])

    print("Setting MultiIndex and sorting...")
    df = df.sort_values(['Name', 'date']).set_index(['date', 'Name'])

    # Target: on day D, predict sign(return(D+1 → D+2))
    # log_return[t] = log(close[t] / close[t-1]) is the return arriving ON day t
    # return(D+1, D+2) = log_return[D+2], so shift(-2) places it on row D
    print("Computing target...")
    df['log_return'] = df.groupby(level='Name')['close'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df['target'] = df.groupby(level='Name')['log_return'].transform(
        lambda x: np.sign(x.shift(-2))
    )

    # Compute features on full dataset — trailing windows have no leakage risk
    # Scaler/imputer (the actual leakage risk) are fit inside CV folds later
    print("Computing features...")
    df = df.groupby(level='Name', group_keys=False).apply(compute_features)

    # Drop NaN rows introduced by rolling warm-up periods and target shift
    print("Dropping NaNs...")
    df = df.dropna(subset=features + ['target'])

    # Split after features are clean
    print("Splitting train/test...")
    train = df[df.index.get_level_values('date') < '2017-01-01'].copy()
    test  = df[df.index.get_level_values('date') >= '2017-01-01'].copy()

    # Remove last 2 trading days from train: their targets use close prices
    # from Jan 2017 (test period) due to shift(-2) crossing the boundary
    train_dates = train.index.get_level_values('date').unique().sort_values()
    cutoff = train_dates[-2]
    train = train[train.index.get_level_values('date') < cutoff]

    X_train, y_train = train[features], train['target']
    X_test,  y_test  = test[features],  test['target']

    print(f"Saving processed data... Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv')
    y_train.to_csv('data/processed/y_train.csv')
    X_test.to_csv('data/processed/X_test.csv')
    y_test.to_csv('data/processed/y_test.csv')

    print(f"\nTrain rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print(f"Date range: {df.index.get_level_values('date').min().date()} "
          f"to {df.index.get_level_values('date').max().date()}")
    print(f"Tickers: {df.index.get_level_values('Name').nunique()}")
    print(f"\nTarget class balance (train):\n{y_train.value_counts(normalize=True).round(3)}")


if __name__ == "__main__":
    main()