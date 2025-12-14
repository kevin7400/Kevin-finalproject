# Architecture

This document describes the high-level architecture and design decisions of the **finance-lstm** project.

The goal of the project is to:

- Download S&P 500 OHLCV data.
- Build 12 technical indicators + 2 targets (next-day return & direction).
- Prepare LSTM-ready sequences.
- Train a stacked LSTM model.
- Train baseline models (Linear, Random Forest, XGBoost).
- Compare all models on a fixed train/test split.

The code is organized as a **simple 4-file structure** under `src/` with a single end-to-end entry point (`main.py`).


## 1. Directory layout

From the project root:

- `src/`
  - `__init__.py` – package marker.
  - `data_loader.py` – **All-in-one module** containing:
    - Configuration constants (ticker, dates, hyperparameters)
    - Raw data download (yfinance) and storage
    - Feature engineering (12 technical indicators)
    - Target creation (next-day return & direction)
    - Preprocessing (train/test split, scaling, LSTM sequence generation)
  - `models.py` – **Model definitions and training**:
    - LSTM model architecture, training, and evaluation
    - Baseline model training helpers
    - Model evaluation utilities
  - `evaluation.py` – **Results and visualization**:
    - Training and evaluating all baseline models
    - Loading LSTM predictions
    - Computing metrics (RMSE, MAE, Accuracy, F1)
    - Generating visualizations (confusion matrices, learning curves)
    - Model comparison and results export
  - `hyperparameter_tuning.py` – **Hyperparameter optimization**:
    - Grid search for baseline models (LinearRegression threshold, RF, XGB)
    - Random search for LSTM hyperparameters
    - Class imbalance handling via balanced sample weights
    - Saving and loading tuned parameters

- `data/`
  - `raw/` – raw OHLCV data downloaded via yfinance.
  - `processed/` – engineered features + targets CSV.
  - `lstm/` – LSTM-ready arrays + scaler + LSTM models + predictions.
  - `tuning/` – saved hyperparameter tuning results (`best_params.json`).

- `results/` – model comparison CSV and visualization plots.

- `tests/` – pytest suite focusing on pure functions and light-weight scenarios.

- `main.py` – user-facing entry point that orchestrates the complete pipeline.


## 2. Data flow overview

The system is intentionally linear and modular. At a high level:

1. **Download raw data** (`src.data_loader.download_and_save_raw_data()`)
   - Uses configuration constants: `TICKER`, `START_DATE`, `END_DATE` from `src.data_loader`
   - Writes a CSV in `data/raw/` (e.g. `sp500_2018_2024.csv`).

2. **Feature engineering & targets** (`src.data_loader.build_and_save_feature_target_dataset()`)
   - Reads the raw CSV from `data/raw/`.
   - Computes 12 indicators:
     - RSI(14), MACD & histogram, Bollinger lower band & %B, SMA(50), EMA(20), OBV,
       min–max normalized close and volume, lagged log return, ATR(14).
   - Defines targets:
     - `next_day_return` = simple % return of next close vs current close.
     - `next_day_direction` = 1 if `next_day_return > 0`, else 0.
   - Drops warmup rows / last row with no next-day close.
   - Saves `data/processed/sp500_<START>_<END>_features_targets.csv`.

3. **Preprocessing for LSTM** (`src.data_loader.prepare_lstm_data()`)
   - Loads processed CSV.
   - Splits by date:
     - Train: `index <= TRAIN_END`.
     - Test:  `index >= TEST_START`.
   - Applies outlier handling to returns using `clip_returns_train_only()` helper:
     - Calculates mean (μ) and standard deviation (σ) from training data only.
     - Clips extreme returns to μ ± 3σ bounds for both train and test.
     - This helper is shared by both `prepare_lstm_data()` and `prepare_tuning_data()` for consistency.
   - Scales 12 features using `MinMaxScaler` (fit on train, apply on test).
   - Builds sequences:
     - For each time `t`, uses `LOOKBACK` consecutive days `[t-LOOKBACK+1, ..., t]`
       to predict the target at `t` (which itself already encodes "next-day").
   - Saves:
     - `X_train_seq.npy`, `y_train_reg_seq.npy`, `y_train_cls_seq.npy`
     - `X_test_seq.npy`,  `y_test_reg_seq.npy`,  `y_test_cls_seq.npy`
     - `feature_scaler.joblib` in `data/lstm/`.

4. **Hyperparameter tuning** (optional, `src.hyperparameter_tuning.tune_all_models()`)
   - Uses train/val/test split (train: 2018-2021, val: 2022, test: 2023-2024).
   - Supports three LSTM modes via `mode` parameter: `regressor`, `classifier`, `multitask`.
   - Tunes each model:
     - LinearRegression: Grid search over classification thresholds.
     - RandomForest: Grid search over n_estimators, max_depth, min_samples_split, + threshold.
     - XGBoost: Grid search over n_estimators, learning_rate, max_depth, gamma, + threshold.
     - LSTM (regressor): Random search over units, dropout, learning_rate, batch_size, loss, + threshold.
     - LSTM (classifier): Random search over units, dropout, learning_rate, batch_size, + threshold.
     - LSTM (multitask): Random search over units, dropout, learning_rate, batch_size, alpha_return, + threshold.
   - Uses balanced sample weights for RF, XGB, and LSTM classifier/multitask to handle class imbalance.
   - Uses `find_best_threshold()` to tune optimal direction prediction threshold for each model.
   - Saves best parameters to `data/tuning/best_params.json`.

5. **LSTM training & prediction** (`src.models.train_and_evaluate_lstm()`)
   - Loads sequences from `data/lstm/`.
   - Builds a stacked LSTM:
     - Input → `LSTM(units1, return_sequences=True)` → Dropout
     - → `LSTM(units2)` → Dropout → `Dense(1, linear)`
   - Hyperparameters:
     - Controlled by `LSTM_CONFIG` dict in `src.data_loader` (units, dropout, batch size, epochs, learning rate).
   - Trains only on `next_day_return` (regression).
   - At evaluation:
     - Predicts continuous returns on the test set.
     - Derives direction label by `pred > 0` → 1, else 0.
   - Saves:
     - Best model: `lstm_model_best.keras`.
     - Final model: `lstm_model_final.keras`.
     - Predicted returns / directions: `y_pred_reg_lstm.npy`, `y_pred_dir_lstm.npy`.

6. **Baselines & model comparison** (`src.evaluation.evaluate_all_models()`)
   - Reloads processed CSV.
   - Applies same date-based split.
   - Scales features via a new `MinMaxScaler` (this is independent from the LSTM scaler).
   - Trains 3 regression baselines on `next_day_return`:
     - `LinearRegression`
     - `RandomForestRegressor`
     - `XGBRegressor`
   - Aligns evaluation window:
     - LSTM test sequences drop the first `LOOKBACK - 1` days.
     - Baselines predictions are also truncated at the front so all models
       are evaluated on the same effective test set.
   - Loads LSTM predictions from `data/lstm/`.
   - Computes metrics for each model:
     - Regression: RMSE, MAE.
     - Direction: Accuracy, F1 (from sign of predicted return).
   - Generates visualizations:
     - Confusion matrices for all models
     - Learning curves for LSTM
   - Saves a comparison table to `results/model_comparison.csv`.


## 3. Design principles & decisions

### 3.1. Configuration-centric design

All configuration constants live at the top of `src/data_loader.py`:

- Ticker symbol and date ranges.
- Train/test boundaries.
- Lookback window.
- LSTM hyperparameters.

The rest of the code accesses configuration via imports from `src.data_loader`, rather than hard-coding constants. This makes it easy to:

- Re-run the pipeline on different time windows.
- Switch to another index/ticker.
- Tune model hyperparameters without editing multiple modules.


### 3.2. Time-based, non-leaky splits

The project explicitly enforces:

- **Time-based splitting**, not random splitting.
- Train period is strictly earlier than test period (no shuffling).
- Scaling is done by fitting `MinMaxScaler` only on the training slice and
  transforming the test slice, avoiding target leakage.

This matches realistic backtesting constraints (you cannot see the future when training).


### 3.3. Sequence alignment and fair comparison

The LSTM uses windows of length `LOOKBACK`. As a result:

- The first `LOOKBACK - 1` test rows cannot form a full sequence.
- LSTM predictions exist only for `len(test) - (LOOKBACK - 1)` timestamps.

To ensure **fair comparison**:

- Baseline predictions are computed on the full test set.
- The first `LOOKBACK - 1` baseline predictions are dropped so LSTM and baselines are evaluated on the same aligned subset.

This is handled centrally in `src.evaluation.evaluate_all_models()`.


### 3.4. Regression-first, direction-from-sign

The specification requires:

- Predicting next-day percentage return.
- Deriving direction as Up/Down from the sign of the predicted return.

Design choice:

- Only one regression model per algorithm (LSTM + baselines).
- No separate classifier is trained.
- Direction is consistently defined as:
  - 1 if predicted return > 0.
  - 0 otherwise.

This keeps the architecture simpler and exactly matches the project requirements.


### 3.5. Class imbalance handling

Stock market data often exhibits class imbalance (e.g., more up days than down days). Without proper handling, models can "game" metrics by always predicting the majority class.

Design choice:

- RandomForest and XGBoost use balanced sample weights during training.
- Sample weights are computed using `sklearn.utils.class_weight.compute_sample_weight('balanced', y_train_cls)`.
- This forces models to treat both classes equally, preventing artificially high F1 scores.
- The LSTM doesn't need this as it's trained on regression (not classification).

This ensures fair and meaningful model comparisons.


### 3.6. Simplified 4-file structure

The project uses a **simplified 4-file structure** instead of a complex package hierarchy:

- `src/data_loader.py` – All data-related operations (download, features, preprocessing, configuration)
- `src/models.py` – All model definitions and training (LSTM and baseline helpers)
- `src/evaluation.py` – All evaluation, metrics, and visualization
- `src/hyperparameter_tuning.py` – Hyperparameter optimization for all models

This structure:

- Reduces cognitive overhead (only 4 files to navigate)
- Keeps related functionality together
- Maintains clear separation of concerns (data, models, evaluation, tuning)
- Makes the codebase easier to understand and modify
- Follows common educational/tutorial patterns for ML projects


### 3.7. Multi-task Learning Architecture

The project supports three LSTM modes for direction prediction:

| Mode | Description | Output |
|------|-------------|--------|
| `regressor` | Predicts returns, derives direction from sign | Returns + Direction |
| `classifier` | Predicts direction directly via sigmoid | Direction only |
| `multitask` | Shared trunk with dual output heads | Returns + Direction |

**Multitask LSTM Architecture:**

```
Input (lookback, n_features)
         │
    LSTM(units1, return_sequences=True)
         │
    Dropout(rate)
         │
    LSTM(units2)
         │
    Dropout(rate)
         │
    +----+----+
    │         │
Dense(1)   Dense(1)
linear     sigmoid
    │         │
return_out  dir_out
```

The multitask model uses:
- **Shared trunk**: Both heads share the LSTM representation
- **Separate heads**: Linear for regression, sigmoid for classification
- **Loss weighting**: Configurable `alpha_return` to balance losses (default 0.5)
- **Class weights**: Sample weights applied to direction head for imbalance handling

This architecture enables the model to learn representations useful for both tasks simultaneously.


### 3.8. Threshold Tuning for Direction Prediction

All models now include optimal threshold tuning using the `find_best_threshold()` helper:

```python
def find_best_threshold(
    y_true_cls: np.ndarray,
    y_pred_continuous: np.ndarray,
    metric: str = "f1",  # or "balanced_acc"
    n_quantiles: int = 100,
) -> Tuple[float, float]:
```

**How it works:**
1. Generate thresholds from quantiles of the prediction distribution
2. Include canonical thresholds (0.0 for returns, 0.5 for probabilities)
3. Search for threshold that maximizes F1 or balanced accuracy
4. Return best threshold and corresponding score

**Application by model type:**
- **Regressors** (LSTM, baselines): Threshold applied to raw return predictions
- **Classifiers** (LSTM classifier): Threshold applied to sigmoid probabilities
- **Multitask**: Threshold applied to direction head probabilities

This approach adapts to the actual prediction distribution rather than assuming a fixed threshold.


### 3.9. Clear separation of concerns

Each module has a narrow responsibility:

- `src/data_loader.py` – IO with external API (yfinance), feature engineering, preprocessing, configuration
- `src/models.py` – modeling logic (LSTM architecture and training)
- `src/evaluation.py` – baseline training, metric computation, visualization, and result aggregation
- `main.py` – orchestration and user-friendly logging

This separation makes it easier to:

- Unit test individual steps with synthetic data.
- Swap or extend components (e.g., add new features, new models, or different scalers).
- Keep side effects (file IO, downloads) localized.


### 3.10. Python package with proper imports

The project is configured as a Python package:

- Package code lives in `src/`.
- `src/__init__.py` makes it importable as a package.
- `pyproject.toml` configures package metadata and dependencies.
- pytest is configured to add the project root to Python path via `pythonpath = ["."]`

Advantages:

- Clean imports: `import src.data_loader`, `import src.models`, `import src.evaluation`
- Tests import the modules exactly as they will be used.
- No import confusion between local modules and installed dependencies.


### 3.11. Artifacts as first-class outputs

Intermediate artifacts are saved to disk at each stage:

- Raw CSV (`data/raw`).
- Processed features/targets (`data/processed`).
- LSTM-ready arrays + scaler (`data/lstm`).
- Trained models and predictions (`data/lstm`).
- Tuned hyperparameters (`data/tuning`).
- Model comparison table and visualizations (`results/`).

This design enables:

- Re-running only parts of the pipeline (e.g., just retrain the LSTM).
- Offline analysis and plotting from CSV/NPY files without re-running the full pipeline.
- Auditing the exact data that fed each model.


### 3.12. Testability

Tests are written to:

- Use small synthetic DataFrames / arrays to avoid heavy downloads.
- Patch filesystem paths and configuration to work in temporary directories.
- Exercise key behaviors:
  - Feature and target definitions.
  - Train/test splits.
  - Sequence construction.
  - LSTM model shape and basic training loop.
  - Baseline training and metric calculation.
  - Evaluation and alignment logic.
  - Pipeline import and orchestration entry points.

The result is a test suite that achieves high coverage without depending on the live yfinance service or long-running training jobs.


## 4. Extension points

The architecture is designed to be extendable:

- **New indicators**:
  - Can be added in `src.data_loader.add_technical_indicators()` and then included
    in the feature list used by preprocessing and evaluation.

- **Different targets**:
  - New targets (e.g., multi-day returns, volatility) can be computed alongside
    `next_day_return` / `next_day_direction` and saved in the processed CSV.

- **Additional models**:
  - New baselines or deep models can be added to `src.models` and integrated into
    `src.evaluation.evaluate_all_models()`.

- **Alternative splits or frequencies**:
  - The time-split logic can be adjusted in configuration constants and `train_test_split_by_date`,
    e.g., to use another horizon or to experiment with rolling windows.


---

This architecture tries to balance:

- Clarity and traceability of each step.
- Strict, non-leaky ML practices for time series.
- Ease of configuration and experimentation.
- Testability and reproducibility.
- Fair model comparison through class imbalance handling.
- Simplicity through consolidation (4 files instead of 10+).
