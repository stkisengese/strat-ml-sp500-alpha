# strat-ml-sp500-alpha
Strat-ML: S&P 500 Alpha Generation Framework:- This repository contains a complete quantitative pipeline designed to outperform the S&P 500 Index using Machine Learning. The project focuses on out-of-sample signal generation using constituent-level OHLCV data, rigorous Blocking Time Series Cross-Validation, and a Long/Short Stock Picking strategy.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the EDA script:**
    ```bash
    python scripts/eda.py
    ```
3.  **Run the scripts in order:**
    ```bash
    python scripts/features_engineering.py
    python scripts/gridsearch.py
    python scripts/model_selection.py
    python scripts/create_signal.py
    python scripts/strategy.py
    ```
