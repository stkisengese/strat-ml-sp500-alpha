import numpy as np
import pandas as pd
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

