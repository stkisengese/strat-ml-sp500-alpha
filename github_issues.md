# GitHub Issues: ML-Driven S&P 500 Trading Strategy

---

## EPIC 1: Project Setup & Repository Structure

---

### Issue #1 — Initialize repository structure and dependencies

**Labels:** `setup` `priority: high`
**Milestone:** Sprint 0

**Description**

Set up the project skeleton exactly as specified so every subsequent script has a consistent place to read from and write to.

**Acceptance Criteria**
- [x] Directory tree matches the spec:
  ```
  project/
  ├── data/
  │   ├── HistoricalPrices.csv
  │   └── all_stocks_5yr.csv
  ├── results/
  │   ├── cross-validation/
  │   ├── selected-model/
  │   └── strategy/
  └── scripts/
      ├── features_engineering.py
      ├── gridsearch.py
      ├── model_selection.py
      ├── create_signal.py
      └── strategy.py
  ```
- [x] `requirements.txt` pins all dependencies (pandas, numpy, scikit-learn, pandas-ta, shap, matplotlib, joblib)
- [x] `README.md` describes how to run each script end-to-end
- [x] Raw data files are placed in `data/` and confirmed loadable

**Learning checkpoint:** Understand the data schema of both CSVs before writing a single line of feature code.

---

### Issue #2 — Exploratory Data Analysis (EDA)

**Labels:** `eda` `data`
**Milestone:** Sprint 0

**Description**

Before any modelling, understand the data's shape, quality, and quirks. Decisions made here propagate through the entire pipeline.

**Tasks**
- [ ] Load `all_stocks_5yr.csv`; check date range, number of unique tickers, missing values per ticker
- [ ] Plot price series for 5 sample tickers — visually confirm non-stationarity
- [ ] Plot log-return series for the same tickers — visually confirm approximate stationarity
- [ ] Check for gaps (missing trading days), duplicate rows, or zero-volume days
- [ ] Confirm the dataset spans the expected range and that 2017 is a clean cut point for train/test
- [ ] Document the survivor bias caveat: the dataset contains only tickers that survived to the collection date

**Learning checkpoint:** Visually seeing the difference between a price series (trending, non-stationary) and its log returns (mean-reverting, stationary) is the intuition behind every modelling choice downstream.

---

## EPIC 2: Feature Engineering — `features_engineering.py`

---

### Issue #3 — Build leakage-free MultiIndex DataFrame

**Labels:** `feature-engineering` `leakage` `priority: critical`
**Milestone:** Sprint 1

**Description**

The entire project's validity rests on this issue. The DataFrame must have a `(date, ticker)` MultiIndex, be sorted chronologically within each ticker, and contain no information from day D+1 or beyond in any row representing day D.

**Tasks**
- [ ] Load `all_stocks_5yr.csv`, parse dates, sort by `(date, Name)`
- [ ] Set MultiIndex to `(date, Name)` — all downstream code will assume this
- [ ] Perform train/test split **before** computing any features:
  ```python
  train = df[df.index.get_level_values("date") < "2017-01-01"]
  test  = df[df.index.get_level_values("date") >= "2017-01-01"]
  ```
- [ ] Write a `compute_features(group)` function that operates on a single ticker's time series using **trailing windows only**
- [ ] Verify no `NaN` leakage: after computing features, confirm that row D contains no close price from D+1

**Leakage audit checklist (must pass before merging):**
- [ ] All rolling windows use `min_periods` and no `center=True`
- [ ] No global statistics (mean, std) computed across the full dataset before splitting
- [ ] No forward-fill that crosses the train/test boundary

**Learning checkpoint:** This issue forces you to internalise the leakage rules discussed in the course — the schema `Features until D 23:59pm → target = sign(return(D+1, D+2))` must be physically present in your DataFrame, not just conceptually understood.

---

### Issue #4 — Implement Bollinger Band features

**Labels:** `feature-engineering` `technical-indicators`
**Milestone:** Sprint 1
**Depends on:** #3

**Description**

Implement Bollinger Band features with correct trailing windows. Raw band levels are not useful as features — derive the stationary, normalised signals.

**Tasks**
- [ ] Compute 20-day SMA and 20-day rolling standard deviation per ticker using `groupby("ticker").transform`
- [ ] Derive `%B = (close - lower_band) / (upper_band - lower_band)` — bounded 0 to 1, measures where price sits within the bands
- [ ] Derive `bandwidth = (upper_band - lower_band) / middle_band` — measures volatility regime; a squeeze (low bandwidth) precedes large moves
- [ ] Use `pandas_ta.bbands` or equivalent library call; verify output matches manual calculation on a sample ticker
- [ ] Confirm both features are stationary (no persistent upward trend)

**Why these features and not raw band levels:** Raw levels inherit the non-stationarity of price. `%B` and `bandwidth` are normalised and stationary by construction — they measure relative position and volatility regime, not absolute price.

---

### Issue #5 — Implement RSI feature

**Labels:** `feature-engineering` `technical-indicators`
**Milestone:** Sprint 1
**Depends on:** #3

**Description**

RSI is a bounded momentum oscillator. Its value already lies in [0, 100] making it naturally well-scaled. The key is ensuring the rolling computation is purely backward-looking.

**Tasks**
- [ ] Compute 14-period RSI per ticker using a trailing window
- [ ] Verify RSI is bounded between 0 and 100 across all tickers
- [ ] Add `rsi_change` = first difference of RSI as an additional feature (captures momentum acceleration)
- [ ] Spot-check: during a strong uptrend, RSI should be consistently high; during a downtrend, consistently low

**Learning checkpoint:** RSI is a mean-reversion signal in sideways markets but a continuation signal in trends — it doesn't have a universal interpretation. The model will learn when to use it through cross-validation, but understanding this regime-dependence explains why it may not always appear in the top feature importances.

---

### Issue #6 — Implement MACD features

**Labels:** `feature-engineering` `technical-indicators`
**Milestone:** Sprint 1
**Depends on:** #3

**Description**

MACD has three components, each capturing a different aspect of trend momentum. All three should be included as separate features.

**Tasks**
- [ ] Compute MACD line (EMA12 − EMA26) per ticker
- [ ] Compute Signal line (EMA9 of MACD line) per ticker
- [ ] Compute Histogram (MACD − Signal) per ticker — this is the most forward-looking component; a shrinking histogram signals decelerating momentum before any line crossover
- [ ] Normalise all three by dividing by the closing price to make them comparable across tickers with different price scales
- [ ] Verify that all three series are stationary (no drift)

---

### Issue #7 — Build and validate the target variable

**Labels:** `feature-engineering` `target` `leakage` `priority: critical`
**Milestone:** Sprint 1
**Depends on:** #3

**Description**

The target must be the **sign of the return from D+1 to D+2, placed on row D**. This shift is the single most common source of leakage in this project.

**Tasks**
- [ ] Compute per-ticker log returns: `log(close_t / close_{t-1})`
- [ ] Shift returns backward by 1 within each ticker group:
  ```python
  df["target"] = df.groupby("Name")["log_return"].transform(lambda x: np.sign(x.shift(-1)))
  ```
- [ ] Confirm the schema with a concrete example:
  - Row for 2016-01-04 (AAPL) should have target = sign(return on 2016-01-06, i.e., close_0106 / close_0105)
- [ ] Drop rows where target is NaN (last day per ticker has no forward return)
- [ ] Check class balance: ideally close to 50/50 between +1 and -1

**Leakage self-test:** If you train a trivial model (e.g., always predict +1) and it achieves accuracy far above 50%, re-examine the target construction — you likely have leakage.

---

### Issue #8 — Assemble and save the final feature matrix

**Labels:** `feature-engineering`
**Milestone:** Sprint 1
**Depends on:** #4 #5 #6 #7

**Description**

Combine all features and target into a single clean DataFrame and persist it for use in all downstream scripts.

**Tasks**
- [ ] Concatenate all feature columns and the target into one DataFrame with MultiIndex `(date, ticker)`
- [ ] Drop rows with NaN in any feature (introduced by rolling window warm-up periods) or in target
- [ ] Confirm the train/test split is still clean after dropping NaN rows
- [ ] Save `X_train`, `y_train`, `X_test`, `y_test` to `data/processed/` as parquet or CSV
- [ ] Print a summary: date range, number of tickers, number of rows, class balance in target

---

## EPIC 3: Cross-Validation — `gridsearch.py`

---

### Issue #9 — Implement date-aware cross-validation splitter

**Labels:** `cross-validation` `priority: high`
**Milestone:** Sprint 2
**Depends on:** #8

**Description**

Standard scikit-learn splitters operate on row indices. Since multiple tickers share the same date, splits must be defined on **unique dates** and then mapped back to rows. Splitting on rows would allow data from the same calendar day to appear in both train and validation — a form of leakage.

**Tasks**
- [ ] Extract `unique_dates` from the training MultiIndex
- [ ] Write a `date_aware_split(unique_dates, n_splits, min_train_days, gap)` generator that yields `(train_dates, val_dates)` pairs
- [ ] Enforce `min_train_days >= 504` (2 trading years)
- [ ] Enforce `gap >= 1` between training end date and validation start date (prevents autocorrelation leakage at the boundary)
- [ ] Write a helper `dates_to_mask(df, dates)` that converts a set of dates to a boolean row mask on the MultiIndex DataFrame
- [ ] Add an assertion that no fold's validation dates overlap with the test set (≥ 2017)

---

### Issue #10 — Implement Blocking Time Series Split

**Labels:** `cross-validation`
**Milestone:** Sprint 2
**Depends on:** #9

**Description**

In a blocking split, the timeline is divided into non-overlapping date blocks. Each fold's training set is one block and its validation set is the immediately following block (with a gap). Folds are independent — training data from fold 1 does not appear in fold 2.

**Tasks**
- [ ] Implement `blocking_time_series_split(unique_dates, n_splits, min_train_days, gap=2)`
- [ ] Verify with a unit test: no two folds share any training dates
- [ ] Verify: each training block contains at least 504 trading days
- [ ] Verify: no validation block bleeds into the test period
- [ ] Save visualisation as `results/cross-validation/blocking_time_series_split.png`

**Visualisation spec:**
- x-axis: calendar date
- y-axis: fold number
- Training periods in blue, validation periods in orange, gap in grey, test set in red
- Each fold on its own horizontal row

**Learning checkpoint:** Because folds are independent, blocking CV gives a less biased model comparison than standard walk-forward — but each fold has less training data. This trade-off should be understood before choosing which CV to use for the grid search.

---

### Issue #11 — Implement Standard Time Series Split (Walk-Forward)

**Labels:** `cross-validation`
**Milestone:** Sprint 2
**Depends on:** #9

**Description**

In walk-forward CV, the training set expands with each fold (expanding window). Each fold trains on all data up to a cutoff, then validates on the next window. This mirrors live deployment most faithfully.

**Tasks**
- [ ] Implement `walk_forward_split(unique_dates, n_splits, min_train_days, gap=2)`
- [ ] Verify with a unit test: fold k's training set is a strict superset of fold k-1's training set
- [ ] Verify: each training set contains at least 504 trading days
- [ ] Save visualisation as `results/cross-validation/Time_series_split.png`

**Note:** Only one CV scheme needs to be submitted. Implement both, compare them in a comment in the code, and commit to one for the grid search. Blocking is recommended when the goal is unbiased model selection; walk-forward when simulating deployment.

---

### Issue #12 — Grid search over pipeline hyperparameters

**Labels:** `cross-validation` `modelling` `priority: high`
**Milestone:** Sprint 2
**Depends on:** #10 #11

**Description**

Run a grid search using the chosen CV scheme. The pipeline must be fit only on each fold's training data — the scaler, imputer, and model all fit inside the fold loop, never on the global dataset.

**Pipeline structure:**
```
SimpleImputer → StandardScaler → [optional PCA] → Classifier
```

**Tasks**
- [ ] Define the pipeline using `sklearn.pipeline.Pipeline`
- [ ] Define `param_grid` covering at minimum:
  - `model__n_estimators`: [100, 200, 300]
  - `model__max_depth`: [3, 5, 7]
  - `model__learning_rate`: [0.05, 0.1] (if GBT)
- [ ] Run grid search using the date-aware CV from #9; do not use `sklearn.GridSearchCV` directly — implement the fold loop manually to enforce date-level splitting
- [ ] For each (fold, hyperparameter set), record AUC, Accuracy, LogLoss on both train and validation partitions
- [ ] Save all results to `results/cross-validation/ml_metrics_train.csv` with double index `(fold, split)`
- [ ] Serialise the best pipeline to `results/selected-model/selected_model.pkl`

**Performance expectation:** AUC meaningfully above 0.5 on validation is a positive signal. If train AUC >> validation AUC, the model is overfitting — reduce complexity or add regularisation.

---

### Issue #13 — Compute and save ML metrics and feature importances

**Labels:** `cross-validation` `feature-importance`
**Milestone:** Sprint 2
**Depends on:** #12

**Description**

Two outputs are required here: a metrics DataFrame and a feature importance DataFrame, both computed fold-by-fold on the train set only.

**Tasks**
- [ ] For each fold, compute on both train and validation:
  - AUC (`roc_auc_score`)
  - Accuracy (`accuracy_score`)
  - Log Loss (`log_loss`)
- [ ] Save as `results/cross-validation/ml_metrics_train.csv` — double index `(fold, split)`, columns = metrics
- [ ] Plot AUC across folds (train vs. validation lines) — save as `results/cross-validation/metric_train.png`
  - x-axis: fold number
  - y-axis: AUC
  - Two lines: train AUC and validation AUC
  - A large and widening gap between train and validation AUC signals overfitting
- [ ] For each fold, extract top 10 feature importances (use `feature_importances_` for tree models or SHAP values)
- [ ] Save as `results/cross-validation/top_10_feature_importance.csv` — index = fold, columns = feature names, values = importance scores

**Learning checkpoint:** Compare feature rankings across folds. If the same features consistently appear in the top 10 across all folds, that is evidence of a genuine, stable signal. If the rankings vary wildly fold-to-fold, the model is fitting noise.

---

## EPIC 4: Model Selection — `model_selection.py`

---

### Issue #14 — Select and validate the final model

**Labels:** `model-selection` `priority: high`
**Milestone:** Sprint 3
**Depends on:** #12 #13

**Description**

Model selection requires analysing the CV metrics, not just picking the highest validation AUC. A model that looks good on average but has high variance across folds is unreliable in deployment.

**Tasks**
- [ ] Load `ml_metrics_train.csv`; compute mean and standard deviation of validation AUC across folds for each candidate pipeline
- [ ] Reject any pipeline where mean(train AUC) >> mean(val AUC) by more than ~0.05 — this is overfitting
- [ ] Prefer the pipeline with the best mean validation AUC AND the lowest fold-to-fold variance
- [ ] Save the selected pipeline to `results/selected-model/selected_model.pkl` (joblib)
- [ ] Save hyperparameters to `results/selected-model/selected_model.txt` in human-readable format

**Decision log:** Add a comment in `model_selection.py` explaining why the selected model was chosen over alternatives. This reasoning should appear in the final `report.md`.

---

## EPIC 5: Signal Generation — `create_signal.py`

---

### Issue #15 — Generate the ML signal via walk-forward prediction

**Labels:** `signal-generation` `priority: critical`
**Milestone:** Sprint 3
**Depends on:** #14

**Description**

The ML signal is the probability output of the classifier, generated by training on fold k and predicting on fold k's validation set — for every fold. This is not the same as training once and predicting everywhere. Each fold gets its own freshly trained model.

**Tasks**
- [ ] Loop over the chosen CV folds:
  ```python
  for fold, (train_dates, val_dates) in enumerate(cv_splits):
      X_tr, y_tr = get_rows(X_train, y_train, train_dates)
      X_val      = get_rows(X_train, None, val_dates)
      
      pipeline.fit(X_tr, y_tr)
      proba = pipeline.predict_proba(X_val)[:, 1]  # P(price goes up)
      
      signal_fold = pd.Series(proba, index=X_val.index, name="ml_signal")
      signal_parts.append(signal_fold)
  ```
- [ ] Concatenate all fold predictions and sort by `(date, ticker)`
- [ ] Verify: every date in the train period (post-warmup) has a signal value for each ticker — no gaps
- [ ] Save to `results/selected-model/ml_signal.csv` with double index `(date, ticker)` and column `ml_signal`

**Critical alignment check:** On day D, `ml_signal` = P(return(D+1, D+2) > 0). This signal will be multiplied by the return from D+1 to D+2 in the backtesting step. Confirm this alignment explicitly with a head/tail printout before proceeding.

---

## EPIC 6: Strategy Backtesting — `strategy.py`

---

### Issue #16 — Implement the backtesting module

**Labels:** `backtesting` `priority: high`
**Milestone:** Sprint 4
**Depends on:** #15

**Description**

The backtesting module converts the ML signal into a position DataFrame and computes daily PnL. The most dangerous mistake — highlighted in the project spec — is applying the signal to the wrong returns. The signal on day D must be multiplied by the return observed between D+1 and D+2.

**Signal-to-strategy conversion (stock picking):**
- Each day, rank all tickers by their ML signal
- Take a long position (buy) on the top-k tickers
- Take a short position (sell) on the bottom-k tickers
- Total absolute investment = $1 per day → weight per side = 1 / (2k)

**Tasks**
- [ ] Implement `signal_to_positions(ml_signal, k=10)` → position DataFrame with MultiIndex `(date, ticker)`, values in {+1/(2k), 0, −1/(2k)}
- [ ] Implement `compute_pnl(positions, returns)`:
  - Align positions from day D with returns from D+1 to D+2
  - Daily PnL = sum over tickers of (position × return) for that day
  - Cumulative PnL = cumsum of daily PnL
- [ ] Verify the temporal alignment with a 5-row worked example before running on the full dataset
- [ ] Compute PnL separately for train and test periods

**Leakage self-test for the strategy:** If cumulative PnL is implausibly smooth or high, re-examine the position-to-return alignment. Real edge is noisy.

---

### Issue #17 — Compute strategy performance metrics

**Labels:** `backtesting` `metrics`
**Milestone:** Sprint 4
**Depends on:** #16

**Description**

Compute the metrics that tell the full story of the strategy — not just whether it made money, but whether the return justified the risk.

**Tasks**
- [ ] **PnL** — total cumulative return on train and test sets
- [ ] **Maximum Drawdown** — largest peak-to-trough decline: `(trough - peak) / peak`
- [ ] **Sharpe Ratio** — `mean(daily_pnl) / std(daily_pnl) * sqrt(252)` — annualised
- [ ] **Calmar Ratio** — `annualised_return / abs(max_drawdown)` — return per unit of worst-case loss
- [ ] Compare all metrics against the S&P 500 benchmark over the same period
- [ ] Save all metrics to `results/strategy/results.csv` with rows = {train, test, sp500} and columns = metrics
- [ ] Interpret the test set metrics honestly: outperforming on train is expected; outperforming on test is the actual signal of strategy quality

---

### Issue #18 — Generate the strategy PnL plot

**Labels:** `backtesting` `visualisation`
**Milestone:** Sprint 4
**Depends on:** #16 #17

**Description**

The PnL plot is the primary visual deliverable. It must show both the strategy and the S&P 500 on the same scale, with a vertical line separating train from test.

**Tasks**
- [ ] Plot cumulative PnL of strategy (y-axis 1) and S&P 500 (y-axis 2) on the **same scale** — if they're on different scales, the comparison is misleading
- [ ] Add a vertical dashed line at `2017-01-01` labelled "Train | Test"
- [ ] Shade the test region lightly to distinguish it visually
- [ ] x-axis: date; y-axis: cumulative PnL in dollars (starting from $0)
- [ ] Add a legend, title, and axis labels
- [ ] Save as `results/strategy/strategy.png` at 150 dpi minimum

**Interpretation guidance to include as a plot annotation or caption:**
- If strategy > S&P 500 in train but not test → overfit
- If strategy > S&P 500 in both → genuine alpha (with appropriate scepticism)
- If strategy < S&P 500 in both → the signal has negative or no value

---

## EPIC 7: Reporting — `report.md`

---

### Issue #19 — Write the final strategy report

**Labels:** `reporting`
**Milestone:** Sprint 5
**Depends on:** all previous issues

**Description**

The report explains every decision made in the pipeline to someone who hasn't seen the code. It should be the single document that tells the full story — what was built, why, and what it achieved.

**Required sections:**

- [ ] **Features** — list all features, explain what market pattern each captures, and justify why they are computed as they are (trailing windows, stationarity)
- [ ] **Pipeline** — document each component:
  - Imputer (strategy and why)
  - Scaler (type and why)
  - Dimensionality reduction (if used)
  - Model (name, final hyperparameters, why selected)
- [ ] **Cross-Validation** — which scheme was chosen (blocking or walk-forward), why, length of train and validation sets per fold, number of folds; include the CV plot
- [ ] **Strategy** — describe the conversion logic (stock-picking / long-short), explain the signal-to-position mapping, confirm the temporal alignment rule
- [ ] **Performance** — embed the PnL plot; report PnL, max drawdown, Sharpe ratio on both train and test; compare to S&P 500; give an honest interpretation of the results
- [ ] **Limitations** — acknowledge survivor bias (only surviving tickers), simplified short-selling assumption, lack of transaction costs, regime sensitivity

---

## EPIC 8: Integration & Quality Assurance

---

### Issue #20 — End-to-end pipeline integration test

**Labels:** `testing` `integration` `priority: high`
**Milestone:** Sprint 5
**Depends on:** all previous issues

**Description**

Run the entire pipeline from raw data to final metrics and verify every output file is produced correctly with no leakage.

**Tasks**
- [ ] Run scripts in order: `features_engineering.py` → `gridsearch.py` → `model_selection.py` → `create_signal.py` → `strategy.py`
- [ ] Verify all output files exist in the correct directories
- [ ] **Leakage smoke test:** Train a model using only the feature values from the day *after* the target date — AUC should collapse to ~0.5. If it doesn't, leakage is present
- [ ] **Temporal alignment smoke test:** Shift the signal forward by one day and re-run PnL computation — PnL should change significantly (confirming signal and returns are aligned correctly in the original code)
- [ ] Confirm the test set was never used during feature computation, scaling, or model training
- [ ] Final check: all output files listed in the repo structure exist and are non-empty

---

### Issue #21 — Code review checklist

**Labels:** `code-quality` `review`
**Milestone:** Sprint 5

**Description**

Before submission, systematically review each script against the key principles from the project.

**Checklist**

**Leakage prevention:**
- [ ] No `StandardScaler` or `SimpleImputer` fit on the full dataset
- [ ] All rolling features use trailing windows only (no `center=True`)
- [ ] Target shift is applied with `shift(-1)` within each ticker group, not globally
- [ ] Signal on day D is multiplied by return from D+1 to D+2, not D to D+1

**Cross-validation correctness:**
- [ ] Splits are on unique dates, not rows
- [ ] No fold's validation set contains dates from the test period (≥ 2017)
- [ ] Each fold's model is trained fresh — no shared state between folds
- [ ] Gap between training end and validation start is at least 1 day

**Stationarity:**
- [ ] Target is `sign(log_return)`, not raw price or raw price level
- [ ] All features are either inherently stationary or explicitly verified as such

**Backtesting correctness:**
- [ ] Total absolute investment per day = $1 across all positions
- [ ] Short positions are correctly signed (negative weight × negative return = positive PnL)
- [ ] PnL benchmark comparison uses the same date range as the strategy

---

## Issue Dependency Map

```
#1 Setup → #2 EDA
              ↓
#3 MultiIndex DF → #4 Bollinger → #8 Final Feature Matrix
              ↓ → #5 RSI ↗
              ↓ → #6 MACD ↗
              ↓ → #7 Target ↗
                              ↓
                        #9 Date-Aware CV Splitter
                        ↓              ↓
                   #10 Blocking   #11 Walk-Forward
                        ↓              ↓
                        #12 Grid Search
                              ↓
                        #13 Metrics & Importances
                              ↓
                        #14 Model Selection
                              ↓
                        #15 Signal Generation
                              ↓
                  #16 Backtesting Module
                  ↓              ↓
            #17 Metrics    #18 PnL Plot
                  ↓              ↓
                  #19 Report
                        ↓
                  #20 Integration Test
                        ↓
                  #21 Code Review
```

## Sprint Plan Summary

| Sprint | Issues | Theme |
|--------|--------|-------|
| 0 | #1, #2 | Setup & EDA |
| 1 | #3, #4, #5, #6, #7, #8 | Feature Engineering |
| 2 | #9, #10, #11, #12, #13 | Cross-Validation & Grid Search |
| 3 | #14, #15 | Model Selection & Signal |
| 4 | #16, #17, #18 | Backtesting |
| 5 | #19, #20, #21 | Report & QA |