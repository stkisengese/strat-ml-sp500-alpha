# S&P 500 Alpha Generation: ML-Driven Long-Short Framework

This repository implements a complete, end-to-end quantitative trading pipeline designed to generate an alpha signal for the S&P 500 constituents using Machine Learning. The project adheres to professional quantitative standards, focusing on rigorous **leakage-free** feature engineering, **date-aware** time-series cross-validation, and a **rank-based** stock-picking strategy.

## 🚀 Project Overview

The core objective is to predict the sign of the next-day return for individual S&P 500 stocks and convert those predictions into a market-neutral long-short strategy.

### Key Technical Pillars:
- **Leakage-Free Feature Engineering:** All features are computed per-ticker using trailing windows only.
- **Stationarity:** Target and features are constructed to ensure stationarity, mitigating the risk of overfitting to non-stationary price data.
- **Walk-Forward Cross-Validation:** A 10-fold expanding window CV scheme specifically designed for time-series data to simulate realistic deployment.
- **Date-Aware Splitting:** Prevents data from the same calendar day from appearing in both training and validation sets.

---

## 📂 Repository Structure

```text
strat-ml-sp500-alpha/
├── data/
│   ├── processed/            # Cleaned, leakage-free train/test sets
│   └── *.csv                 # Raw OHLCV data for S&P 500 constituents
├── results/
│   ├── cross-validation/     # CV visualizations, metrics, and feature importances
│   ├── selected-model/       # Serialized model (pkl) and hyperparameter reports
│   └── strategy/             # PnL plots and performance metrics
├── scripts/
│   ├── cv.py                 # Core CV splitters and helper utilities (reusable)
│   ├── features_engineering.py # Pipeline step 1: Feature & target construction
│   ├── gridsearch.py         # Pipeline step 2: Parallel optimization & fold metrics
│   ├── model_selection.py    # Pipeline step 3: Final model selection & full-train
│   ├── create_signal.py      # Pipeline step 4: Out-of-sample signal generation
│   └── strategy.py           # Pipeline step 5: Backtesting & performance analysis
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

## 🛠 Installation & Requirements

Ensure you have Python 3.8+ installed.

```bash
# Clone the repository
git clone https://github.com/stkisengese/strat-ml-sp500-alpha.git
cd strat-ml-sp500-alpha

# Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Execution Pipeline

To reproduce the results, execute the scripts in the following order:

1.  **Feature Engineering:**
    ```bash
    python scripts/features_engineering.py
    ```
2.  **Grid Search & Evaluation:** (Optimizes params and computes per-fold metrics)
    ```bash
    python scripts/gridsearch.py
    ```
3.  **Model Selection:** (Finalizes parameters and fits the model on full train data)
    ```bash
    python scripts/model_selection.py
    ```
4.  **Signal Generation:** (Produces out-of-sample walk-forward predictions)
    ```bash
    python scripts/create_signal.py
    ```
5.  **Backtesting:** (Generates PnL plots and strategy analysis)
    ```bash
    python scripts/strategy.py
    ```

---

## 📊 Methodology Summary

### 1. Features & Target
We utilize 7 primary technical features: **Bollinger Bands (%B, Width)**, **RSI (Level, Change)**, and **MACD (Line, Signal, Histogram)**. The target is defined as the sign of the forward return:
$$Target_D = \text{sign}(Return_{D+1 \rightarrow D+2})$$

### 2. Validation Strategy
The pipeline uses a **10-fold Walk-Forward Cross-Validation** to simulate actual trading conditions. Detailed metrics (AUC, Accuracy, Log Loss) and permutation importances are computed for the optimal model across all folds to ensure robustness.

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.

**Author:** [Stephen Kisengese](github.com/stkisengese)  
**Status:** Completed - March 2026
