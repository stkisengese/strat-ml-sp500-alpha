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

# Import cross-validation utilities from cv_utils.py
from cv_utils import (walk_forward_split, dates_to_mask, plot_cv_scheme,
                       get_unique_dates, plot_learning_curves)

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

    # --- Detailed Performance Metrics & Feature Importance for the Best Model ---
    print("\n--- Performing Detailed Evaluation of Optimal Model ---")
    metrics_records = []
    importance_records = []
    feature_names = X_train.columns.tolist()

    for fold_idx, (train_dates, val_dates) in enumerate(cv_splits):
        train_mask = dates_to_mask(X_train, train_dates)
        val_mask   = dates_to_mask(X_train, val_dates)
        
        X_tr = X_train[train_mask].values
        y_tr = (y_train[train_mask].values == 1).astype(int)
        X_val = X_train[val_mask].values
        y_val = (y_train[val_mask].values == 1).astype(int)
        
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', HistGradientBoostingClassifier(**best_params, random_state=42))
        ])
        pipe.fit(X_tr, y_tr)
        
        # Calculate training and validation metrics (AUC, Accuracy, Log Loss)
        for name, X, y_bin in [("train", X_tr, y_tr), ("validation", X_val, y_val)]:
            proba = pipe.predict_proba(X)[:, 1]
            metrics_records.append({
                'fold': fold_idx, 'split': name,
                'auc': roc_auc_score(y_bin, proba),
                'accuracy': accuracy_score(y_bin, (proba > 0.5).astype(int)),
                'log_loss': log_loss(y_bin, proba)
            })
        
        # Compute feature importances using permutation (robust for tree-based models)
        perm = permutation_importance(pipe, X_tr, y_tr, n_repeats=5, random_state=42, n_jobs=-1)
        for feat, imp in zip(feature_names, perm.importances_mean):
            importance_records.append({'fold': fold_idx, 'feature': feat, 'importance': imp})

    # Save detailed metrics and feature importance data to CSV files
    os.makedirs("results/cross-validation", exist_ok=True)
    df_metrics = pd.DataFrame(metrics_records).set_index(['fold', 'split'])
    df_metrics.to_csv("results/cross-validation/ml_metrics_train.csv")
    
    df_imp = pd.DataFrame(importance_records).pivot(index='fold', columns='feature', values='importance')
    df_imp.to_csv("results/cross-validation/top_10_feature_importance.csv")

    plot_learning_curves(cv_splits, df_metrics)

    print("\nGrid Search and Detailed Evaluation COMPLETE.")
    print("Next Step: Run model_selection.py to finalize the pipeline.")

if __name__ == "__main__":
    main()
