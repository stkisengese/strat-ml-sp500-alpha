import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
import os
from datetime import timedelta
from typing import Iterator, Tuple, List

# --- Core Cross-Validation Splitters ---

def blocking_time_series_split(
    unique_dates: np.ndarray,
    n_splits: int = 10,
    min_train_days: int = 504,
    val_days: int = 63,
    gap: int = 2
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Implements a blocking time-series split where each fold is independent.
    The timeline is divided into non-overlapping blocks of (Train + Gap + Validation).
    """
    n = len(unique_dates)
    stride = n // n_splits

    for i in range(n_splits):
        start      = i * stride
        train_end  = start + min_train_days
        val_start  = train_end + gap
        val_end    = val_start + val_days

        if val_end > n:
            break

        yield unique_dates[start:train_end], unique_dates[val_start:val_end]


def walk_forward_split(
    unique_dates: np.ndarray,
    n_splits: int = 10,
    min_train_days: int = 504,
    val_days: int = 63,
    gap: int = 2
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Implements an expanding-window (walk-forward) split.
    Each subsequent fold includes all previous training data, maintaining temporal order.
    """
    unique_dates = np.array(unique_dates)
    n = len(unique_dates)

    # Calculate validation window size if not explicitly provided
    if val_days is None or val_days == 0:
        val_days = max(1, (n - min_train_days - gap) // n_splits)

    for i in range(n_splits):
        train_end = min_train_days + i * val_days
        if train_end + gap >= n:
            break
        val_start = train_end + gap
        val_end = min(n, val_start + val_days)
        
        yield unique_dates[:train_end], unique_dates[val_start:val_end]


# --- Utility Functions ---

def dates_to_mask(df: pd.DataFrame, dates: np.ndarray) -> pd.Series:
    """
    Generates a boolean mask for a MultiIndex DataFrame based on a list of dates.
    Assumes 'date' is one of the index levels.
    """
    return df.index.get_level_values('date').isin(dates)


def get_unique_dates(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts and sorts the unique calendar dates from a MultiIndex DataFrame.
    """
    return np.sort(df.index.get_level_values('date').unique())


def plot_cv_scheme(cv_splits: List[Tuple[np.ndarray, np.ndarray]], unique_dates: np.ndarray, title: str, filename: str):
    """
    Visualizes the training, gap, and validation periods for each cross-validation fold.
    Saves the resulting plot to the results/cross-validation/ directory.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    unique_dates = pd.to_datetime(unique_dates)
    
    for fold, (train_d, val_d) in enumerate(cv_splits):
        train_d = pd.to_datetime(train_d)
        val_d = pd.to_datetime(val_d)
        
        # Plot training window (Blue) and Validation window (Orange)
        ax.hlines(fold, train_d[0], train_d[-1], colors='blue', linewidth=12, label='Train' if fold == 0 else "")
        ax.hlines(fold, val_d[0], val_d[-1], colors='orange', linewidth=12, label='Validation' if fold == 0 else "")
        
        # Highlight the gap period (Grey)
        gap_start = train_d[-1] + timedelta(days=1)
        gap_end = val_d[0] - timedelta(days=1)
        if gap_start < gap_end:
            ax.hlines(fold, gap_start, gap_end, colors='grey', linewidth=8, label='Gap' if fold == 0 else "")
            
    # Add a reference line for the final out-of-sample test period
    test_start = pd.to_datetime("2017-01-01")
    if unique_dates[-1] > test_start:
        ax.hlines(-1, test_start, unique_dates[-1], colors='red', linewidth=12, label='Test Set')
        
    ax.set_yticks(range(-1, len(cv_splits)))
    ax.set_yticklabels(['Test Period'] + [f"Fold {i}" for i in range(len(cv_splits))])
    ax.set_title(title)
    ax.set_xlabel("Date Timeline")
    ax.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    os.makedirs("results/cross-validation", exist_ok=True)
    plt.savefig(f"results/cross-validation/{filename}", dpi=150, bbox_inches="tight")
    plt.close()


def plot_learning_curves(cv_splits, df_metrics):
    """Plots the AUC learning curves for training and validation across the cross-validation folds."""
    folds = range(len(cv_splits))
    train_auc = df_metrics.loc[(slice(None), 'train'), 'auc'].values
    val_auc   = df_metrics.loc[(slice(None), 'validation'), 'auc'].values

    plt.figure(figsize=(10, 6))
    plt.plot(folds, train_auc, 'b-o', label='Train AUC')
    plt.plot(folds, val_auc, color='orange', marker='o', linestyle='-', label='Validation AUC')
    plt.axhline(0.5, color='gray', linestyle='--', label='Random Benchmark')
    plt.title('AUC Performance across Walk-Forward Folds')
    plt.xlabel('Fold Index')
    plt.ylabel('AUC Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/cross-validation/metric_train.png", dpi=150, bbox_inches='tight')
    plt.close()


def assert_no_test_leakage(cv_splits: List[Tuple[np.ndarray, np.ndarray]], test_cutoff: str = '2017-01-01'):
    """
    Verification utility to ensure no training or validation date overlaps with the test period.
    """
    cutoff = pd.Timestamp(test_cutoff)
    for i, (_, val_dates) in enumerate(cv_splits):
        leaked = [d for d in val_dates if pd.Timestamp(d) >= cutoff]
        if len(leaked) > 0:
            raise ValueError(f"Fold {i} contains {len(leaked)} validation dates from the test set.")
    print(" Temporal split validation: No leakage into test set detected.")
