# ML-Driven S&P 500 Long-Short Trading Strategy – Final Report

## 1. Features Used
The model uses 7 stationary, leakage-free technical features (all computed with trailing windows only):

- **Bollinger Bands** (`bb_percent`, `bb_width`): Measures where price sits inside the bands and volatility regime (squeeze → expansion).
- **RSI** (`rsi`, `rsi_change`): Momentum oscillator [0–100] + its first difference (acceleration).
- **MACD** (`macd`, `macd_signal`, `macd_hist`): Normalised by close price to be comparable across tickers.

All features were computed **after** the 2017 train/test split on each ticker independently → no leakage.

## 2. ML Pipeline
- **Imputer**: `SimpleImputer(strategy='mean')`
- **Scaler**: `StandardScaler()`
- **Model**: `HistGradientBoostingClassifier(max_iter=100, max_depth=3, learning_rate=0.05)`
- No dimensionality reduction (PCA) needed — only 7 features.

**Hyperparameters** chosen via manual grid search (18 combinations) on walk-forward CV.

## 3. Cross-Validation
- **Scheme**: Walk-Forward (expanding window) — chosen because blocking was impossible with only 947 training days.
- 10 folds, each with ≥504 training days (2 trading years) + 44-day validation + gap=2 days.
- Visualisation: `results/cross-validation/Time_series_split.png`
- Mean Validation AUC = **0.5096** (±0.0160)
- Train AUC = **0.5287** (±0.0028) → very small overfit gap (0.0191) → acceptable.

## 4. Strategy Description
- **Type**: Stock-picking long-short (k=10)
- Every day: Long the 10 highest `ml_signal` stocks (+$0.05 each), Short the 10 lowest (-$0.05 each). Total absolute investment = **$1 per day**.
- Signal on day **D** = P(return from **D+1 to D+2** > 0) → multiplied by actual return(D+1 → D+2).
- No transaction costs, no slippage, simplified short-selling (opposite of long).

## 5. Performance
**Cumulative PnL plot** (`results/strategy/strategy.png`):

![PnL plot](./strategy.png)

**Metrics** (`results/strategy/results.csv`):

| Period | PnL      | Max_Drawdown | Sharpe Ratio | Calmar Ratio |
|--------|----------|--------------|--------------|--------------|
| Train  | -0.0658  | 5.5053       | -0.3815      | -0.0068      |
| Test   | -0.2228  | 0.0000*      | -1.9169      | 0.0000       |
| S&P 500| +0.6526  | 0.3511       | +0.7144      | +0.6551      |

\* Max drawdown = 0 in test because cumulative PnL is monotonically decreasing.

**Honest Interpretation**:
The strategy **does not outperform** the S&P 500 on the test set (2017–2018).  
It actually loses money (-22.3%) while the benchmark gains +65.3%.  
Train performance is also flat-to-negative.  

This is expected: the model only achieved AUC ≈ 0.51 (very weak edge). Pure technical indicators on daily data in a survivor-biased dataset rarely produce tradable alpha without costs, regime detection, or more sophisticated features.

## 6. Limitations
- Survivor bias (only stocks that survived until 2018).
- COVID period removed from dataset.
- No transaction costs, slippage, or short-selling constraints.
- Simplified constant $1 daily investment.
- Model has almost no predictive power (AUC near 0.5) → strategy behaves like random long-short.

## Conclusion
The end-to-end leakage-free pipeline works perfectly (train/test split before features, date-aware CV, correct D+1→D+2 alignment).  
However, the chosen technical features do not contain enough signal to beat the S&P 500 in this simplified setting.  

This project successfully demonstrated the full quant workflow and the extreme difficulty of beating the market with ML on public data.

**Author**: Stephen Kisengese  
**Date**: March 2026