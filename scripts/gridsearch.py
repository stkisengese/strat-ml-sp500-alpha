import pandas as pd
import numpy as np
import os
from itertools import product
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed

# Import cross-validation utilities from cv.py
from cv import walk_forward_split, dates_to_mask, plot_cv_scheme, get_unique_dates, plot_learning_curves

def evaluate_combo(n_est, max_d, lr, cv_splits, X_train, y_train):
    """
    Evaluates a single hyperparameter combination across all cross-validation folds.
    Returns the mean validation AUC and the list of per-fold AUCs.
    """
    fold_val_aucs = []
    for train_dates, val_dates in cv_splits:
        train_mask = dates_to_mask(X_train, train_dates)
        val_mask   = dates_to_mask(X_train, val_dates)
        
        X_tr = X_train[train_mask].values
        y_tr = (y_train[train_mask].values == 1).astype(int)
        X_val = X_train[val_mask].values
        y_val = (y_train[val_mask].values == 1).astype(int)
        
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler',  StandardScaler()),
            ('model', HistGradientBoostingClassifier(
                max_iter=n_est,
                max_depth=max_d,
                learning_rate=lr,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=42
            ))
        ])
        pipe.fit(X_tr, y_tr)
        
        proba_val = pipe.predict_proba(X_val)[:, 1]
        auc_val = roc_auc_score(y_val, proba_val)
        fold_val_aucs.append(auc_val)
    
    return (n_est, max_d, lr, np.mean(fold_val_aucs), fold_val_aucs)

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