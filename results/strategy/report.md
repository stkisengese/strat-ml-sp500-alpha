# ML-Driven S&P 500 Long-Short Trading Strategy – Final Report

## 1. Features Used
The model uses 11 stationary, leakage-free technical features (all computed with trailing windows only):

- **Bollinger Bands** (`bb_percent`, `bb_width`): Measures price position relative to volatility bands and the volatility regime (squeeze → expansion).
- **RSI** (`rsi`, `rsi_change`): Momentum oscillator [0–100] and its first difference (acceleration).
- **MACD** (`macd`, `macd_signal`, `macd_hist`): Trend-following momentum indicators, normalized by price for cross-ticker comparability.
- **ATR** (`atr_norm`): Average True Range normalized by price to capture relative volatility.
- **ADX** (`adx`): Average Directional Index (normalized to [0, 1]) to measure trend strength.
- **OBV** (`obv_change`): Percentage change in On-Balance Volume to capture volume-driven momentum.
- **Williams %R** (`willr`): Momentum indicator measuring overbought/oversold levels (normalized to [0, 1]).

All features were computed **after** the 2017 train/test split on each ticker independently → no leakage.

## 2. ML Pipeline
- **Imputer**: `SimpleImputer(strategy='mean')`
- **Scaler**: `StandardScaler()`
- **Model**: `HistGradientBoostingClassifier(max_iter=200, max_depth=5, learning_rate=0.1)`
- No dimensionality reduction (PCA) needed — 11 features.

**Hyperparameters** chosen via manual grid search (18 combinations) on walk-forward CV.

## 3. Cross-Validation & Feature Importance
- **Scheme**: Walk-Forward (expanding window) — chosen because blocking was impossible with only 947 training days.
- 10 folds, each with ≥504 training days (2 trading years) + 44-day validation + gap=2 days.
- Visualisation: `results/cross-validation/Time_series_split.png`
- Mean Validation AUC = **0.5185** (±0.0098)
- Train AUC = **0.5726** (±0.0075) → overfit gap (**0.0541**) → slightly higher but acceptable.

### Feature Importance (Permutation-based)
Based on `results/cross-validation/top_10_feature_importance.csv`, the model relies most heavily on price-range and volume indicators:

| Feature      | Importance (Mean ΔAUC) |
|--------------|------------------------|
| `bb_percent` | 0.0138                 |
| `willr`      | 0.0130                 |
| `obv_change` | 0.0117                 |
| `rsi`        | 0.0098                 |
| `macd`       | 0.0094                 |

`bb_percent` and `willr` (Williams %R) are the most influential, suggesting the model is primarily finding signals in overbought/oversold price extremes. `obv_change` is also a significant contributor, indicating that volume momentum is a key secondary signal.

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
| Train  | +0.0280  | 11.8323      | +0.1511      | +0.0014      |
| Test   | -0.1290  | 5.4090       | -1.9169      | -0.0219      |
| S&P 500| +0.6526  | 0.3511       | +0.7134      | +0.6542      |

**Honest Interpretation**:
The strategy **does not outperform** the S&P 500 on the test set (2017–2018).  
While performance improved compared to the baseline (Test PnL -12.9% vs -22.3%), it still loses money while the benchmark gains +65.3%.  
Train performance turned slightly positive (+2.8%), but with a very high drawdown and low Sharpe ratio.

This is expected: even with improved AUC (≈0.52), the edge remains very weak. Pure technical indicators on daily data in a survivor-biased dataset rarely produce tradable alpha without costs, regime detection, or more sophisticated features.

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