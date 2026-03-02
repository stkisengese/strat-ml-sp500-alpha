import pandas as pd
import numpy as np
import matplotlib
# use non-interactive backend for environments without display (headless)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import timedelta

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
    """Expanding window (walk-forward). Mirrors live deployment."""
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

def plot_cv_scheme(cv_splits, unique_dates, title, filename):
    """Save a visualisation of the cross-validation scheme."""
    fig, ax = plt.subplots(figsize=(14, 7))
    for fold, (train_d, val_d) in enumerate(cv_splits):
        # Train
        ax.hlines(fold, train_d[0], train_d[-1], colors='blue', linewidth=12, label='Train' if fold == 0 else "")
        # Val
        ax.hlines(fold, val_d[0], val_d[-1], colors='orange', linewidth=12, label='Validation' if fold == 0 else "")
        # Gap
        gap_start = train_d[-1] + timedelta(days=1)
        gap_end = val_d[0] - timedelta(days=1)
        if gap_start < gap_end:
            ax.hlines(fold, gap_start, gap_end, colors='grey', linewidth=8, label='Gap' if fold == 0 else "")
    # Test set (red)
    ax.hlines(-1, pd.to_datetime("2017-01-01"), unique_dates[-1], colors='red', linewidth=12, label='Test Set')
    ax.set_yticks(range(len(cv_splits)))
    ax.set_yticklabels([f"Fold {i}" for i in range(len(cv_splits))])
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    os.makedirs("results/cross-validation", exist_ok=True)
    plt.savefig(f"results/cross-validation/{filename}", dpi=150, bbox_inches="tight")
    plt.close()

def main():
    # Load the clean processed data from features_engineering.py
    print("Loading processed data...")
    X_train = pd.read_csv('data/processed/X_train.csv', index_col=[0, 1], parse_dates=True, date_format="%Y-%m-%d")
    y_train = pd.read_csv('data/processed/y_train.csv', index_col=[0, 1], parse_dates=True, date_format="%Y-%m-%d").iloc[:, 0]  # Convert to Series

    unique_dates = sorted(X_train.index.get_level_values("date").unique())
    print(f"Unique training dates: {len(unique_dates)} (from {unique_dates[0].date()} to {unique_dates[-1].date()})")

    # Generate both schemes
    blocking_splits = list(blocking_time_series_split(unique_dates, n_splits=5, min_train_days=504, gap=2))
    walk_splits = list(walk_forward_split(unique_dates, n_splits=10, min_train_days=504, gap=2))
    
    print(f"Blocking CV folds generated: {len(blocking_splits)}")
    print(f"Walk-Forward CV folds generated: {len(walk_splits)}")
    
    # ── Test walk-forward split ────────────────────────────────────────────
    print("\n=== Walk-Forward Split ===")
    wf_splits = list(walk_forward_split(unique_dates, n_splits=10, min_train_days=504, gap=2))

    for i, (tr, val) in enumerate(wf_splits):
        print(f"Fold {i:2d} | "
              f"Train: {tr[0].date()} → {tr[-1].date()} ({len(tr)} days) | "
              f"Val:   {val[0].date()} → {val[-1].date()} ({len(val)} days)")

    # Verify expanding: each fold's training set is a superset of the previous
    for i in range(1, len(wf_splits)):
        prev_train = set(wf_splits[i-1][0])
        curr_train = set(wf_splits[i][0])
        assert prev_train.issubset(curr_train), \
            f"Walk-forward fold {i} training set is not a superset of fold {i-1}!"
    print("✓ Walk-forward training sets are strictly expanding")

    # Visualisations for the report
    plot_cv_scheme(walk_splits, unique_dates, "Walk-Forward (Time Series) Split", "Time_series_split.png")
    
    # Safety assertion (no test leakage)
    for name, splits in [("Blocking", blocking_splits), ("Walk-Forward", walk_splits)]:
        for train_d, val_d in splits:
            assert max(val_d) < pd.to_datetime("2017-01-01"), f"{name} validation leaks into test period!"
    
    # Choose one for the grid search 
    chosen_mode = "walk_forward"  # or "blocking"
    cv_splits = list(date_aware_split(unique_dates, n_splits=5 if chosen_mode == "blocking" else 10,
                                      min_train_days=504, gap=2, mode=chosen_mode))
    print(f"\nUsing {chosen_mode.upper()} CV for grid search with {len(cv_splits)} folds.")
    print("CV visualisations saved → results/cross-validation/")
    print("Leakage-free date-aware splitting ready for grid search.")

if __name__ == "__main__":
    main()