# LSTM S&P 500 Forecasting Project

This project builds an end-to-end pipeline to forecast next-day percentage returns and direction (Up/Down) for the S&P 500 using:

- 12 technical indicators
- A stacked LSTM model (TensorFlow / Keras)
- Baseline models (Linear Regression, Random Forest, XGBoost)
- A time-based train/test split (default: 2018–2022 train, 2023–2024 test)

The code is structured as a Python package with a `src/` layout:

- Package: `finance_lstm/` (under `src/`)
- Data folders: `data/raw/`, `data/processed/`, `data/lstm/`, `data/results/`

---

## 1. Setup (once)

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate

# Install the project in editable mode (also installs dependencies)
pip install -e .
````

Recommended Python version: **3.10+** (the code has been tested with Python 3.9+).

If needed, you can also install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

(but `pip install -e .` is enough, as it installs the package and its dependencies).

---

## 2. Configuration (`src/finance_lstm/config.py`)

All main project settings are centralized in:

* `src/finance_lstm/config.py`

The rest of the modules read from this file and **do not** hard-code dates or hyperparameters.

Typical contents of `config.py`:

```python
# Ticker to download (S&P 500 index)
TICKER = "^GSPC"

# Raw download date range
START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"

# Train / test split boundaries
TRAIN_END  = "2022-12-31"
TEST_START = "2023-01-01"

# LSTM lookback window (number of past days)
LOOKBACK = 64

# LSTM hyperparameters
LSTM_CONFIG = {
    "units1": 64,
    "units2": 32,
    "dropout": 0.2,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-3,
}
```

These values are used by:

* `finance_lstm.download_data` for ticker and download date range
* `finance_lstm.features` for building indicators and targets
* `finance_lstm.preprocessing` for the train/test split and lookback
* `finance_lstm.models.lstm` for the LSTM architecture and training
* `finance_lstm.evaluation` for aligning baseline evaluation
* `finance_lstm.pipeline` to orchestrate everything

If you change any of these in `config.py`, the whole pipeline will adapt automatically on the next run.

> Note: if you significantly change `START_DATE` / `END_DATE` after you already generated data, you may want to delete existing files under `data/raw/`, `data/processed/`, and `data/lstm/` so the pipeline can rebuild everything using the new configuration.

---

## 3. Run the full pipeline

Once the virtual environment is activated:

```bash
source .venv/bin/activate
python run_pipeline.py
```

This will:

1. **Download raw data** with `yfinance` (if the raw CSV does not exist), using:

   * `config.TICKER`
   * `config.START_DATE`
   * `config.END_DATE`

   and save to `data/raw/sp500_YYYY_YYYY.csv`.

2. **Build features & targets** (12 technical indicators + next-day return & direction) based on the raw data and save them to:

   * `data/processed/sp500_YYYY_YYYY_features_targets.csv`

3. **Preprocess data for LSTM** via `finance_lstm.preprocessing`:

   * Time-based split using `config.TRAIN_END` and `config.TEST_START`
   * MinMax scaling of the 12 indicators
   * Creation of 3D sequences `[samples, config.LOOKBACK, 12]` for the LSTM
   * Arrays and scaler stored under `data/lstm/`:

     * `X_train_seq.npy`, `y_train_reg_seq.npy`, `y_train_cls_seq.npy`
     * `X_test_seq.npy`, `y_test_reg_seq.npy`, `y_test_cls_seq.npy`
     * `feature_scaler.joblib`

4. **Train the LSTM model** via `finance_lstm.models.lstm`:

   * Stacked LSTM layers with sizes/dropout from `config.LSTM_CONFIG`
   * `Dense(1, linear)` output for next-day percentage return
   * Early stopping and learning-rate reduction on plateau
   * Saves models and test predictions under `data/lstm/`:

     * `lstm_model_best.keras`
     * `lstm_model_final.keras`
     * `y_pred_reg_lstm.npy` (continuous % returns)
     * `y_pred_dir_lstm.npy` (Up/Down labels from predicted sign)

5. **Train baseline models and evaluate all models** via `finance_lstm.evaluation`:

   * Baselines (trained on flat features):

     * Linear Regression
     * Random Forest Regressor
     * XGBoost Regressor
   * LSTM (using the saved predictions)
   * Metrics on the test set:

     * **RMSE**, **MAE** (magnitude of next-day return)
     * **Accuracy**, **F1** (direction, using sign of predicted return)
   * Results are printed and saved to:

     * `data/results/model_comparison.csv`

---

## 4. Running each step manually (optional)

If you prefer to run each step individually (for debugging or experimentation), you can use the following commands from the project root, **after** activating the virtual environment:

```bash
source .venv/bin/activate
```

### 4.1. Download data

```bash
python -m finance_lstm.download_data
```

* Uses `config.TICKER`, `config.START_DATE`, and `config.END_DATE`.
* Downloads daily OHLCV data (e.g. S&P 500 via `^GSPC`) using `yfinance`.
* Saves to:

  * `data/raw/sp500_YYYY_YYYY.csv` (exact name depends on config).

### 4.2. Build features & targets

```bash
python -m finance_lstm.features
```

* Loads the raw CSV from `data/raw/`.

* Computes 12 technical indicators:

  * RSI (14)
  * MACD line + MACD histogram
  * Bollinger Bands (lower band + %B)
  * SMA_50
  * EMA_20
  * OBV
  * Normalized close price
  * Normalized daily volume
  * Lagged log return (t-1)
  * ATR (14)

* Computes two targets:

  * `next_day_return` (simple percentage return, in %)
  * `next_day_direction`:

    * `1` if `next_day_return > 0`
    * `0` otherwise

* Drops warm-up rows with NaNs and the last row (no next-day target).

* Saves to:

  * `data/processed/sp500_YYYY_YYYY_features_targets.csv`

### 4.3. Preprocess for LSTM

```bash
python -m finance_lstm.preprocessing
```

* Loads the processed dataset from `data/processed/`.

* Splits by date using `config.TRAIN_END` and `config.TEST_START`:

  * Train: up to `TRAIN_END` (inclusive)
  * Test: from `TEST_START` (inclusive)

* Scales the 12 indicators with `MinMaxScaler` (fit on train, apply to test).

* Builds LSTM sequences with a lookback window of `config.LOOKBACK` trading days:

  * `X_train_seq` shape: `(n_train_sequences, LOOKBACK, 12)`
  * `X_test_seq` shape:  `(n_test_sequences,  LOOKBACK, 12)`

* Creates aligned regression and classification targets at the **last day in the window**:

  * `y_train_reg_seq`, `y_train_cls_seq`
  * `y_test_reg_seq`, `y_test_cls_seq`

* Saves arrays and the scaler to:

  * `data/lstm/X_train_seq.npy`
  * `data/lstm/y_train_reg_seq.npy`
  * `data/lstm/y_train_cls_seq.npy`
  * `data/lstm/X_test_seq.npy`
  * `data/lstm/y_test_reg_seq.npy`
  * `data/lstm/y_test_cls_seq.npy`
  * `data/lstm/feature_scaler.joblib`

### 4.4. Train the LSTM model

```bash
python -m finance_lstm.models.lstm
```

* Loads the LSTM-ready arrays from `data/lstm/`.

* Builds a stacked LSTM model using `config.LSTM_CONFIG`:

  * `LSTM(units1, return_sequences=True)` → `Dropout(dropout)`
  * `LSTM(units2)` → `Dropout(dropout)`
  * `Dense(1, activation="linear")` (regression output)

* Trains to predict `next_day_return` (in %) with:

  * `batch_size = config.LSTM_CONFIG["batch_size"]`
  * `epochs = config.LSTM_CONFIG["epochs"]`
  * `learning_rate = config.LSTM_CONFIG["learning_rate"]`

* Uses EarlyStopping + ReduceLROnPlateau + ModelCheckpoint.

* After training:

  * Saves the best model to `data/lstm/lstm_model_best.keras`
  * Saves the final model to `data/lstm/lstm_model_final.keras`
  * Predicts test returns → `data/lstm/y_pred_reg_lstm.npy`
  * Derives test directions (`> 0` → Up) → `data/lstm/y_pred_dir_lstm.npy`

### 4.5. Train baseline models and evaluate all models

```bash
python -m finance_lstm.evaluation
```

* Loads the processed dataset and applies the same time-based train/test split using `config.TRAIN_END` and `config.TEST_START`.

* Scales features with `MinMaxScaler` (fit on train, transform test).

* Trains three baselines on `next_day_return`:

  * `LinearRegression`
  * `RandomForestRegressor`
  * `XGBRegressor`

* To compare fairly with LSTM, it evaluates baselines on the **same subset** of the test set as the LSTM sequences (i.e. it drops the first `LOOKBACK - 1` test samples).

* Loads LSTM test predictions from `data/lstm/`.

* Evaluates all models (baselines + LSTM) using:

  * **RMSE**, **MAE** (regression magnitude)
  * **Accuracy**, **F1** (direction classification)

* Prints a comparison table such as:

  * `model, rmse, mae, accuracy, f1`

* Saves the comparison table to:

  * `data/results/model_comparison.csv`

This CSV can be used directly in the project report to show that the LSTM outperforms the baselines (lower RMSE/MAE and higher Accuracy/F1, if everything is configured correctly).

## 5. Tests

A small test suite is provided to validate the core logic (feature engineering, preprocessing, models, evaluation, and pipeline imports).

### 5.1. Install test dependencies

From the project root, after creating/activating the virtualenv:

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
```

If you prefer installing only the minimal dependencies for testing:

```bash
pip install pytest pytest-cov
```

### 5.2. Run all tests

From the project root:

```bash
pytest
```

This will run all tests under the `tests/` directory.

### 5.3. Run tests with coverage

To measure coverage for the `finance_lstm` package:

```bash
pytest --cov=finance_lstm --cov-report=term-missing
```

This prints a line-by-line summary to the terminal, including which lines are not covered.

You can also generate an HTML coverage report:

```bash
pytest --cov=finance_lstm --cov-report=html
```

This will create an `htmlcov/` folder. Open `htmlcov/index.html` in your browser to inspect coverage interactively.

### 5.4. Running a specific test module

For quick iteration, you can run a single test file, for example:

```bash
pytest tests/test_features.py
pytest tests/test_preprocessing.py
pytest tests/test_lstm_model.py
```

This is useful when you are modifying only one part of the pipeline and want fast feedback.

## 6. Code style & linting

This project uses **Black** for formatting and **Flake8** for linting (PEP 8 compliance).

### 6.1. Install dev dependencies

If you haven't already:

```bash
pip install -r requirements-dev.txt
```

### 6.2. Run Black (auto-formatter)

Formats the source code, tests, and the pipeline entrypoint:

```bash
black src tests run_pipeline.py
```

Black reads its configuration from `pyproject.toml` (line length, exclusions, etc.).

### 6.3. Run Flake8 (linter)

Checks for style issues, unused imports, etc.:

```bash
flake8 src tests run_pipeline.py
```

Flake8 is configured via `.flake8` in the project root. The goal is for **both Black and Flake8 to run cleanly** before committing or submitting the project.
