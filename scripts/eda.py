import pandas as pd
import numpy as np
import matplotlib
# use non-interactive backend for environments without display (headless)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Load the dataset
try:
    df = pd.read_csv('data/all_stocks_5yr.csv', parse_dates=['date'])
except FileNotFoundError:
    print("Error: 'data/all_stocks_5yr.csv' not found. Make sure the data file is in the 'data' directory.")
    exit()

# --- EDA Tasks ---

# 1. Check date range, number of unique tickers, missing values per ticker
print("--- EDA Summary ---")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Number of unique tickers: {df['Name'].nunique()}")
print("\nMissing values per ticker:")
print(df.groupby('Name').apply(lambda x: x.isnull().sum()))

# 2. Plot price series for 5 sample tickers
print("\nPlotting price series for 5 sample tickers...")
sample_tickers = df['Name'].unique()[:5]
plt.figure(figsize=(15, 8))
for ticker in sample_tickers:
    plt.plot(df[df['Name'] == ticker]['date'], df[df['Name'] == ticker]['close'], label=ticker)
plt.title('Price Series for 5 Sample Tickers')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.savefig('results/price_series_sample.png')
print("Saved 'results/price_series_sample.png'")

# 3. Plot log-return series for the same tickers
print("\nPlotting log-return series for the same tickers...")
plt.figure(figsize=(15, 8))
for ticker in sample_tickers:
    log_returns = np.log(df[df['Name'] == ticker]['close'] / df[df['Name'] == ticker]['close'].shift(1))
    plt.plot(df[df['Name'] == ticker]['date'], log_returns, label=ticker)
plt.title('Log-Return Series for 5 Sample Tickers')
plt.xlabel('Date')
plt.ylabel('Log Return')
plt.legend()
plt.savefig('results/log_return_series_sample.png')
print("Saved 'results/log_return_series_sample.png'")

# 4. Check for gaps, duplicate rows, or zero-volume days
print("\nChecking for data quality issues...")
# Check for gaps in trading days for each ticker
gaps = df.groupby('Name')['date'].diff().dt.days.gt(1).sum()
print(f"Number of gaps (missing trading days) > 1 day: {gaps}")

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Check for zero-volume days
zero_volume_days = (df['volume'] == 0).sum()
print(f"Number of zero-volume days: {zero_volume_days}")

# 5. Confirm the dataset spans the expected range and that 2017 is a clean cut point for train/test
print("\nConfirming date range and train/test split point...")
start_date = df['date'].min()
end_date = df['date'].max()
print(f"Dataset starts on {start_date} and ends on {end_date}.")
if pd.Timestamp('2017-01-01') > start_date and pd.Timestamp('2017-01-01') < end_date:
    print("The year 2017 is a valid split point for train/test sets.")
else:
    print("Warning: 2017 is not a suitable split point as it's outside the data range.")

# 6. Document the survivor bias caveat
print("\n--- Important Caveat ---")
print("Survivor Bias: This dataset contains only tickers that survived to the collection date. This means that stocks that were delisted due to bankruptcy, acquisition, or other reasons are not included. This can lead to an optimistic bias in backtesting results, as the strategy is not tested on a representative sample of the market.")

print("\nEDA script finished.")
