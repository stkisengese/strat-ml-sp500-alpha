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

if __name__ == "__main__":
    main()
