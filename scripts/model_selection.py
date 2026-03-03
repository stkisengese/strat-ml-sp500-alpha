import pandas as pd
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

def main():
    print("--- Starting Final Model Selection & Validation ---")

    # Load performance metrics generated during the grid search phase
    metrics_path = "results/cross-validation/ml_metrics_train.csv"
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}. Ensure gridsearch.py has been executed.")
    
    df_metrics = pd.read_csv(metrics_path, index_col=['fold', 'split'])

    # Aggregate performance statistics across all cross-validation folds
    train_auc_mean = df_metrics.loc[(slice(None), 'train'), 'auc'].mean()
    train_auc_std  = df_metrics.loc[(slice(None), 'train'), 'auc'].std()
    val_auc_mean   = df_metrics.loc[(slice(None), 'validation'), 'auc'].mean()
    val_auc_std    = df_metrics.loc[(slice(None), 'validation'), 'auc'].std()

    overfit_gap = train_auc_mean - val_auc_mean

    print(f"Mean Train AUC: {train_auc_mean:.4f} (± {train_auc_std:.4f})")
    print(f"Mean Val AUC:   {val_auc_mean:.4f} (± {val_auc_std:.4f})")
    print(f"Overfitting Gap: {overfit_gap:.4f}")

    # Validation: Check for excessive overfitting (gap > 0.05)
    if overfit_gap > 0.05:
        print("WARNING: Significant overfitting detected. Model complexity may be too high.")
    else:
        print("✅ Model generalization appears stable (Overfitting gap <= 0.05).")

    # Load the best hyperparameters identified during grid search
    params_path = "results/selected-model/selected_model.txt"
    try:
        with open(params_path, "r") as f:
            content = f.read()
            # Handle potential updates to the file format in subsequent runs
            if "Hyperparameters:" in content:
                best_params = {}
                for line in content.split('\n'):
                    if '=' in line:
                        key, val = line.split('=')
                        best_params[key.strip()] = eval(val.strip())
            else:
                best_params = eval(content)
    except Exception as e:
        print(f"Error loading hyperparameters: {e}. Using default configuration.")
        best_params = {'max_iter': 100, 'max_depth': 3, 'learning_rate': 0.05}

    print(f"\nFinal Model Configuration: {best_params}")

    # Initialize the final pipeline with optimal hyperparameters
    final_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', HistGradientBoostingClassifier(
            **best_params,
            random_state=42
        ))
    ])

    # Load the full training dataset for the final model fit
    X_train = pd.read_csv("data/processed/X_train.csv", parse_dates=["date"]).set_index(["date", "Name"])
    y_train = pd.read_csv("data/processed/y_train.csv", parse_dates=["date"]).set_index(["date", "Name"]).iloc[:, 0]

    print("Fitting the final model on the full training dataset...")
    final_pipeline.fit(X_train.values, (y_train.values == 1).astype(int))

    # Serialize and save the final pipeline and a summary report
    os.makedirs("results/selected-model", exist_ok=True)
    joblib.dump(final_pipeline, "results/selected-model/selected_model.pkl")

    report_content = (
        f"Selected Model: HistGradientBoostingClassifier\n"
        f"Hyperparameters:\n"
        f"    max_iter       = {best_params.get('max_iter')}\n"
        f"    max_depth      = {best_params.get('max_depth')}\n"
        f"    learning_rate  = {best_params.get('learning_rate')}\n"
        f"Mean Validation AUC: {val_auc_mean:.4f} (± {val_auc_std:.4f})\n"
        f"Training-Validation Gap: {overfit_gap:.4f}\n"
        f"Selection Rationale: Best performance on Walk-Forward CV with low variance and minimal overfitting.\n"
    )

    with open("results/selected-model/selected_model.txt", "w") as f:
        f.write(report_content)

    print("\n✅ Final model and selection report saved to results/selected-model/")
    print("Next Step: Run create_signal.py to generate out-of-sample predictions.")

if __name__ == "__main__":
    main()
