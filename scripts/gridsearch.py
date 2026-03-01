import pandas as pd
import numpy as np
import matplotlib
# use non-interactive backend for environments without display (headless)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def main():
    # Load the clean processed data from features_engineering.py
    print("Loading processed data...")
    X_train = pd.read_csv('data/processed/X_train.csv', index_col=[0, 1], parse_dates=True)
    y_train = pd.read_csv('data/processed/y_train.csv', index_col=[0, 1], parse_dates=True).iloc[:, 0]  # Convert to Series

    unique_dates = sorted(X_train.index.get_level_values("date").unique())
    print(f"Unique training dates: {len(unique_dates)} (from {unique_dates[0].date()} to {unique_dates[-1].date()})")


if __name__ == "__main__":
    main()