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
    print("--- Starting Hyperparameter Optimization ---")
    
    # Load processed training data
    X_train = pd.read_csv("data/processed/X_train.csv", parse_dates=["date"]).set_index(["date", "Name"])
    y_train = pd.read_csv("data/processed/y_train.csv", parse_dates=["date"]).set_index(["date", "Name"]).iloc[:, 0]

    # Generate cross-validation splits based on unique dates
    unique_dates = get_unique_dates(X_train)
    cv_splits = list(walk_forward_split(unique_dates, n_splits=10, min_train_days=504, gap=2))
    
    # Visualize the CV scheme (Training, Validation, and Gap periods)
    plot_cv_scheme(cv_splits, unique_dates, "Walk-Forward Time Series Split", "Time_series_split.png")

    # Define hyperparameter combinations for the grid search
    param_combos = list(product([100, 200, 300], [3, 5, 7], [0.05, 0.1]))
    
    # Execute grid search using parallel processing for efficiency
    print(f"Evaluating {len(param_combos)} combinations in parallel...")
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(evaluate_combo)(n_est, max_d, lr, cv_splits, X_train, y_train)
        for n_est, max_d, lr in param_combos
    )
    
    # Identify the best hyperparameter set based on mean validation AUC
    best_idx = np.argmax([r[3] for r in results])
    best_n_est, best_max_d, best_lr, best_mean_auc, _ = results[best_idx]
    best_params = {'max_iter': best_n_est, 'max_depth': best_max_d, 'learning_rate': best_lr}
    
    print(f"\nOptimal Parameters: {best_params} (Mean Val AUC: {best_mean_auc:.4f})")

    # Save best parameters to a text file for subsequent pipeline steps
    os.makedirs("results/selected-model", exist_ok=True)
    with open("results/selected-model/selected_model.txt", "w") as f:
        f.write(str(best_params))

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