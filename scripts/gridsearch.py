import pandas as pd
import numpy as np
import matplotlib
# use non-interactive backend for environments without display (headless)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def dates_to_mask(df: pd.DataFrame, dates: np.ndarray) -> pd.Series:
    """Convert set of dates to boolean row mask on MultiIndex (date, ticker)."""
    return df.index.get_level_values("date").isin(dates)

def date_aware_split(unique_dates, n_splits=10, min_train_days=504, gap=2, mode="blocking"):
    """Generic date-aware splitter (used internally by the two schemes)."""
    if mode == "blocking":
        return blocking_time_series_split(unique_dates, n_splits, min_train_days, gap)
    elif mode == "walk_forward":
        return walk_forward_split(unique_dates, n_splits, min_train_days, gap)
    raise ValueError("mode must be 'blocking' or 'walk_forward'")

def blocking_time_series_split(unique_dates, n_splits=5, min_train_days=504, gap=2):
    """Non-overlapping blocks. Each fold's train and val are independent."""
    unique_dates = np.array(unique_dates)
    n = len(unique_dates)
    # Adapt block size to respect min_train_days while trying to reach requested folds
    effective_n_splits = min(n_splits, (n - gap) // (min_train_days + 63))
    block_size = max(min_train_days, (n - effective_n_splits * gap) // (effective_n_splits * 2))
    for i in range(effective_n_splits):
        train_start = i * 2 * block_size
        train_end = train_start + block_size
        val_start = train_end + gap
        val_end = val_start + block_size
        if val_end > n:
            break
        yield unique_dates[train_start:train_end], unique_dates[val_start:val_end]


def walk_forward_split(unique_dates, n_splits=10, min_train_days=504, gap=2):
    """Issue #11: Expanding window (walk-forward). Mirrors live deployment."""
    unique_dates = np.array(unique_dates)
    n = len(unique_dates)
    val_size = max(30, (n - min_train_days) // n_splits)  # ~1 month val per fold
    for i in range(n_splits):
        train_end = min_train_days + i * val_size
        if train_end >= n:
            break
        val_start = train_end + gap
        val_end = val_start + val_size
        if val_end > n:
            break
        yield unique_dates[:train_end], unique_dates[val_start:val_end]


def main():
    # Load the clean processed data from features_engineering.py
    print("Loading processed data...")
    X_train = pd.read_csv('data/processed/X_train.csv', index_col=[0, 1], parse_dates=True)
    y_train = pd.read_csv('data/processed/y_train.csv', index_col=[0, 1], parse_dates=True).iloc[:, 0]  # Convert to Series

    unique_dates = sorted(X_train.index.get_level_values("date").unique())
    print(f"Unique training dates: {len(unique_dates)} (from {unique_dates[0].date()} to {unique_dates[-1].date()})")


if __name__ == "__main__":
    main()