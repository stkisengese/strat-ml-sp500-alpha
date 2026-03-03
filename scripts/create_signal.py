import pandas as pd
import numpy as np
import joblib
import os

# Import utility functions from cv.py
from cv import walk_forward_split, dates_to_mask, get_unique_dates

def main():
    print("--- Starting ML Signal Generation (Walk-Forward) ---")

    # Load the full training dataset to recreate the CV splits
    X_train = pd.read_csv("data/processed/X_train.csv", parse_dates=["date"]).set_index(["date", "Name"])
    y_train = pd.read_csv("data/processed/y_train.csv", parse_dates=["date"]).set_index(["date", "Name"]).iloc[:, 0]

    # Load the serialized pipeline (best model)
    model_path = "results/selected-model/selected_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run model_selection.py first.")
    
    pipe = joblib.load(model_path)
    print(f"Loaded Optimal Model: {pipe.named_steps['model'].__class__.__name__}")

    # Re-generate the identical walk-forward folds used during the optimization phase
    unique_dates = get_unique_dates(X_train)
    cv_splits = list(walk_forward_split(unique_dates, n_splits=10, min_train_days=504, gap=2))

    print(f"Generating signals across {len(cv_splits)} walk-forward folds...")

    signal_parts = []

    # For each fold, we re-train the model on that fold's training set and predict on its validation set.
    # This ensures that the generated signal for the entire training period is truly out-of-sample.
    for fold_idx, (train_dates, val_dates) in enumerate(cv_splits):
        train_mask = dates_to_mask(X_train, train_dates)
        val_mask   = dates_to_mask(X_train, val_dates)
        
        X_tr = X_train[train_mask].values
        y_tr = (y_train[train_mask].values == 1).astype(int)
        X_val = X_train[val_mask].values
        
        # Fit model on the current fold's training data
        pipe.fit(X_tr, y_tr)
        
        # Predict the probability of the 'Up' class (return D+1 -> D+2 > 0)
        proba = pipe.predict_proba(X_val)[:, 1]
        
        # Store results with the original MultiIndex
        signal_fold = pd.Series(
            proba,
            index=X_train[val_mask].index,
            name="ml_signal"
        )
        signal_parts.append(signal_fold)
        
        print(f"  Fold {fold_idx:2d}: {len(val_dates)} days processed.")

    # Combine all fold predictions into a single, chronologically sorted signal series
    ml_signal = pd.concat(signal_parts).sort_index()

    # Save the generated signal for backtesting
    os.makedirs("results/selected-model", exist_ok=True)
    ml_signal.to_csv("results/selected-model/ml_signal.csv")

    print("\n--- ML Signal Generation COMPLETE ---")
    print(f"Total predictions: {len(ml_signal):,}")
    print(f"Signal saved to: results/selected-model/ml_signal.csv")

if __name__ == "__main__":
    main()
