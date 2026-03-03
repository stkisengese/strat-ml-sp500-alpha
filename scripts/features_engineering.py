import pandas as pd
import numpy as np
import pandas_ta as ta
import os

def compute_features(group):
    """
    Computes technical indicators for a single ticker's price history.
    Uses only trailing windows to prevent data leakage.
    """
    close = group['close']

    # Bollinger Bands (20-day): Measures price position relative to volatility bands.
    bbands = ta.bbands(close, length=20, std=2)
    if bbands is not None:
        group['bb_percent'] = bbands['BBP_20_2.0_2.0'].values  # %B: (price - lower) / (upper - lower)
        group['bb_width']   = bbands['BBB_20_2.0_2.0'].values  # Bandwidth: Measures volatility 'squeezes'

    # RSI (14-day): Momentum oscillator that measures speed and change of price moves.
    rsi = ta.rsi(close, length=14)
    if rsi is not None:
        group['rsi']        = rsi.values
        group['rsi_change'] = rsi.diff().values  # Captures momentum acceleration/deceleration

    # MACD (12, 26, 9): Trend-following momentum indicator.
    # We normalize these by price to ensure comparability across different stock price scales.
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None:
        group['macd']        = (macd_df['MACD_12_26_9']  / close).values
        group['macd_signal'] = (macd_df['MACDs_12_26_9'] / close).values
        group['macd_hist']   = (macd_df['MACDh_12_26_9'] / close).values

    return group

def run_data_validation_assertions(X_train, y_train, X_test, y_test, features):
    """
    Performs critical sanity checks on the processed datasets to ensure integrity and prevent leakage.
    """
    # 1. Temporal Integrity: Ensure no overlap between training and testing dates.
    last_train_date = X_train.index.get_level_values('date').max()
    assert last_train_date < pd.Timestamp('2017-01-01'), f"Leakage: Training data extends into 2017 ({last_train_date})"

    train_dates = set(X_train.index.get_level_values('date'))
    test_dates  = set(X_test.index.get_level_values('date'))
    assert train_dates.isdisjoint(test_dates), "Leakage: Overlapping dates between Train and Test sets."

    # 2. Target Distribution: Ensure the target classes are reasonably balanced.
    pos_ratio = (y_train == 1).mean()
    assert 0.40 < pos_ratio < 0.60, f"Unbalanced target: {pos_ratio:.2%} are positive. Verify shift logic."

    # 3. Data Quality: Ensure no missing values remain in the final matrices.
    assert not X_train.isnull().any().any(), "X_train contains NaN values."
    assert not X_test.isnull().any().any(), "X_test contains NaN values."

    # 4. Feature-Target Correlation: Identify potential target leakage into features.
    for feat in features:
        correlation = y_train.corr(X_train[feat])
        assert abs(correlation) < 0.8, f"Suspicious correlation: {feat} vs Target = {correlation:.2f}"

    print("✅ All data validation assertions passed.")


def main():
    features = ['bb_percent', 'bb_width', 'rsi', 'rsi_change',
                'macd', 'macd_signal', 'macd_hist']

    print("Loading raw OHLCV data...")
    df = pd.read_csv('data/all_stocks_5yr.csv')
    df['date'] = pd.to_datetime(df['date'])

    # Data is sorted by ticker then date for correct indicator calculation
    df = df.sort_values(['Name', 'date']).set_index(['date', 'Name'])

    # --- Target Construction ---
    # Goal: Predict the return sign for the interval (D+1 -> D+2) based on information up to day D.
    # log_return[t] is the return from t-1 to t. Therefore, log_return[D+2] is the return (D+1 -> D+2).
    # We shift this value back by 2 to align it with features on row D.
    print("Calculating target variable...")
    df['log_return'] = df.groupby(level='Name')['close'].transform(lambda x: np.log(x / x.shift(1)))
    df['target'] = df.groupby(level='Name')['log_return'].transform(lambda x: np.sign(x.shift(-2)))

    # --- Feature Engineering ---
    print("Calculating technical indicators...")
    df = df.groupby(level='Name', group_keys=False).apply(compute_features)

    # Remove rows with NaNs introduced by rolling windows or target shifts
    df = df.dropna(subset=features + ['target'])

    # --- Train/Test Split ---
    # The split is performed based on time to simulate real-world prediction conditions.
    print("Splitting data into Training (< 2017) and Testing (>= 2017) sets...")
    train = df[df.index.get_level_values('date') < '2017-01-01'].copy()
    test  = df[df.index.get_level_values('date') >= '2017-01-01'].copy()

    # Safety buffer: Drop the final two days of training to ensure no overlap 
    # of target values (which look 2 days ahead) into the test period.
    train_dates = train.index.get_level_values('date').unique().sort_values()
    cutoff = train_dates[-2]
    train = train[train.index.get_level_values('date') < cutoff]

    X_train, y_train = train[features], train['target']
    X_test,  y_test  = test[features],  test['target']

    # Final validation
    run_data_validation_assertions(X_train, y_train, X_test, y_test, features)

    # Save processed datasets
    print(f"Saving processed data: {len(X_train):,} train rows, {len(X_test):,} test rows.")
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv')
    y_train.to_csv('data/processed/y_train.csv')
    X_test.to_csv('data/processed/X_test.csv')
    y_test.to_csv('data/processed/y_test.csv')

    print("\nFeature Engineering COMPLETE.")

if __name__ == "__main__":
    main()
