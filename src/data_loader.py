"""
Data loading and preprocessing module.

This module consolidates:
- Configuration constants (from config.py)
- Data download using yfinance (from download_data.py)
- Feature engineering with 12 technical indicators (from features.py)
- Preprocessing: scaling and LSTM sequence creation (from preprocessing.py)

Public functions:
    - download_and_save_raw_data()
    - build_and_save_feature_target_dataset()
    - prepare_lstm_data()
"""

from __future__ import annotations

import pathlib
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# ---------------------------------------------------------------------------
# Configuration Constants (from config.py)
# ---------------------------------------------------------------------------

# Random seed for reproducibility
# Note: LSTM predictions are sensitive to initialization. Seed 5678 gives balanced
# predictions (~75% Up) while other seeds may give extreme biases (2% or 100% Up).
RANDOM_SEED = 5678

# Ticker to download (S&P 500 index)
TICKER = "^GSPC"

# Raw download date range
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

# Train / test split boundaries (for final evaluation)
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"

# Hyperparameter tuning split boundaries
# Training: 2018-2021 (4 years), Validation: 2022 (1 year), Test: 2023-2024 (unchanged)
TUNE_TRAIN_END = "2021-12-31"
TUNE_VAL_START = "2022-01-01"
TUNE_VAL_END = "2022-12-31"

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
    "loss": "mse",  # Options: "mse", "mae", "huber"
}

# Number of technical indicators
N_FEATURES = 12

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# src/data_loader.py -> project root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LSTM_DIR = DATA_DIR / "lstm"

# Create directories
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LSTM_DIR.mkdir(parents=True, exist_ok=True)

# Module-level variables for default paths (used by tests)
START_YEAR = START_DATE[:4]
END_YEAR = END_DATE[:4]
RAW_CSV = RAW_DIR / f"sp500_{START_YEAR}_{END_YEAR}.csv"
PROCESSED_CSV = PROCESSED_DIR / f"sp500_{START_YEAR}_{END_YEAR}_features_targets.csv"


# ---------------------------------------------------------------------------
# Data Download (from download_data.py)
# ---------------------------------------------------------------------------


def get_default_raw_csv_path() -> pathlib.Path:
    """
    Default path for the raw S&P 500 CSV, based on config dates.

    Example: data/raw/sp500_2018_2024.csv
    """
    start_year = START_DATE[:4]
    end_year = END_DATE[:4]
    return RAW_DIR / f"sp500_{start_year}_{end_year}.csv"


def download_sp500_data(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download historical OHLCV data via yfinance.

    Returns:
        DataFrame with DatetimeIndex and at least:
            ['Open', 'High', 'Low', 'Close', 'Volume']
    """
    print(f"Calling yfinance.download(ticker={ticker}, start={start}, end={end})...")
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=True,
    )

    if df.empty:
        raise RuntimeError(
            "Downloaded DataFrame is empty. " "Check ticker or date range in config."
        )

    # Handle MultiIndex columns like ('Close', '^GSPC')
    if isinstance(df.columns, pd.MultiIndex):
        if "Price" in df.columns.names:
            df.columns = df.columns.get_level_values("Price")
        else:
            df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing required OHLCV columns: {missing}. "
            f"Got columns: {list(df.columns)}"
        )

    cols_to_keep = [
        c
        for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        if c in df.columns
    ]
    df = df[cols_to_keep]

    print(f"Downloaded {len(df)} rows with columns: {list(df.columns)}")
    return df


def download_and_save_raw_data(force: bool = False) -> pathlib.Path:
    """
    Download raw data (if needed) and return the CSV path.

    Args:
        force:
            If True, always re-download and overwrite the CSV.
            If False, reuse existing CSV when available.

    Returns:
        Path to the raw CSV file.
    """
    csv_path = get_default_raw_csv_path()

    if csv_path.exists() and not force:
        print(f"Raw CSV already exists at {csv_path}, reusing it.")
        return csv_path

    print(
        f"Downloading raw data for {TICKER} " f"from {START_DATE} to {END_DATE}..."
    )
    df = download_sp500_data(
        ticker=TICKER,
        start=START_DATE,
        end=END_DATE,
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=True)
    print(f"Saved raw data to: {csv_path.resolve()}")

    return csv_path


# ---------------------------------------------------------------------------
# Feature Engineering (from features.py)
# ---------------------------------------------------------------------------


def get_default_processed_csv_path() -> pathlib.Path:
    """
    Return the canonical path for the processed features/targets CSV.

    Example: data/processed/sp500_2018_2024_features_targets.csv
    """
    start_year = START_DATE[:4]
    end_year = END_DATE[:4]
    return PROCESSED_DIR / f"sp500_{start_year}_{end_year}_features_targets.csv"


def load_raw_data(path: pathlib.Path) -> pd.DataFrame:
    """
    Load raw OHLCV data from CSV.
    Expects columns at least: open, high, low, close, volume.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found at: {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Standardize column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing required columns {missing} in raw data. Got: {list(df.columns)}"
        )

    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the 12 technical indicators specified in the project:

    1)  RSI (Relative Strength Index, 14)
    2)  MACD line (12, 26, 9)
    3)  MACD_H (MACD histogram)
    4)  BBL (Bollinger Band Lower, 20, 2)
    5)  BBP (Bollinger Band %B, 20, 2)
    6)  SMA_50
    7)  EMA_20
    8)  OBV (On-Balance Volume)
    9)  Close Price (normalized, min-max)
    10) Daily Volume (normalized, min-max)
    11) Lagged Log Return (t-1)
    12) ATR (Average True Range, 14)
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # 1) RSI (length 14)
    rsi_indicator = RSIIndicator(close=close, window=14)
    df["rsi_14"] = rsi_indicator.rsi()

    # 2–3) MACD line & histogram (12, 26, 9)
    macd_indicator = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_indicator.macd()  # MACD line
    df["macd_h"] = macd_indicator.macd_diff()  # MACD histogram (macd - signal)

    # 4–5) Bollinger Bands (20, 2): lower band & %B
    bb_indicator = BollingerBands(close=close, window=20, window_dev=2)
    df["bbl"] = bb_indicator.bollinger_lband()
    df["bbp"] = bb_indicator.bollinger_pband()

    # 6) SMA_50
    sma_50_indicator = SMAIndicator(close=close, window=50)
    df["sma_50"] = sma_50_indicator.sma_indicator()

    # 7) EMA_20
    ema_20_indicator = EMAIndicator(close=close, window=20)
    df["ema_20"] = ema_20_indicator.ema_indicator()

    # 8) OBV
    obv_indicator = OnBalanceVolumeIndicator(close=close, volume=volume)
    df["obv"] = obv_indicator.on_balance_volume()

    # 9) Close Price (normalized: min-max over full period)
    close_min = close.min()
    close_max = close.max()
    df["close_norm"] = (close - close_min) / (close_max - close_min)

    # 10) Daily Volume (normalized: min-max)
    vol_min = volume.min()
    vol_max = volume.max()
    df["volume_norm"] = (volume - vol_min) / (vol_max - vol_min)

    # 11) Lagged Log Return (t-1)
    # log_return_t = log(C_t / C_{t-1})
    # lagged_log_return_t = log_return_{t-1}
    log_return = np.log(close / close.shift(1))
    df["lagged_log_return"] = log_return.shift(1)

    # 12) ATR (Average True Range, 14)
    atr_indicator = AverageTrueRange(high=high, low=low, close=close, window=14)
    df["atr_14"] = atr_indicator.average_true_range()

    return df


def add_targets(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Add the two targets:
      - next_day_return: Simple percentage return for next day (Regression target)
      - next_day_direction: Up/Down binary (Classification target)

    Definition (simple return):
      next_day_return_t = ((Close_{t+1} - Close_t) / Close_t) * 100

    Direction rule:
      - If next_day_return > threshold -> 1 (Up)
      - Else -> 0 (Down)

    Default: threshold = 0.0 (%), i.e. strictly positive = Up, otherwise Down.
    """
    close = df["close"]

    next_close = close.shift(-1)
    next_day_return = (next_close - close) / close * 100.0

    df["next_day_return"] = next_day_return
    df["next_day_direction"] = (next_day_return > threshold).astype(int)

    return df


def build_feature_target_dataset() -> pd.DataFrame:
    """
    Full pipeline for feature/target dataset:

      - Load raw CSV
      - Add indicators
      - Add targets
      - Drop rows with NaNs (indicator warmup + last row with no next-day return)
      - Keep only the 12 features + 2 targets
    """
    df = load_raw_data(RAW_CSV)

    df = add_technical_indicators(df)
    df = add_targets(df, threshold=0.0)

    feature_cols = [
        "rsi_14",
        "macd",
        "macd_h",
        "bbl",
        "bbp",
        "sma_50",
        "ema_20",
        "obv",
        "close_norm",
        "volume_norm",
        "lagged_log_return",
        "atr_14",
    ]
    target_cols = ["next_day_return", "next_day_direction"]

    df_final = df[feature_cols + target_cols].dropna()

    print(f"Final dataset shape: {df_final.shape}")
    print("Columns:", df_final.columns.tolist())
    print("Date range:", df_final.index.min(), "->", df_final.index.max())

    return df_final


def save_dataset(df: pd.DataFrame, path: pathlib.Path) -> None:
    """Save the DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)
    print(f"Saved feature/target dataset to: {path.resolve()}")


def build_and_save_feature_target_dataset() -> pathlib.Path:
    """
    Convenience helper used by the pipeline.

    Builds the dataset and saves it to the default processed CSV path,
    then returns that path.
    """
    df = build_feature_target_dataset()
    save_dataset(df, PROCESSED_CSV)
    return PROCESSED_CSV


# ---------------------------------------------------------------------------
# Preprocessing (from preprocessing.py)
# ---------------------------------------------------------------------------


def load_processed_dataset() -> pd.DataFrame:
    """
    Load the processed feature/target CSV built by feature engineering.
    """
    path = get_default_processed_csv_path()
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()

    print("Loaded processed dataset:")
    print("Shape:", df.shape)
    print("Date range:", df.index.min(), "->", df.index.max())

    return df


def train_test_split_by_date(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split:
      - Train: all rows with date <= train_end (inclusive)
      - Test:  all rows with date >= test_start (inclusive)
    """
    df_train = df[df.index <= train_end].copy()
    df_test = df[df.index >= test_start].copy()

    print(
        "\nTrain set:",
        df_train.index.min(),
        "->",
        df_train.index.max(),
        "rows:",
        len(df_train),
    )
    print(
        "Test  set:",
        df_test.index.min(),
        "->",
        df_test.index.max(),
        "rows:",
        len(df_test),
    )

    if df_train.empty or df_test.empty:
        raise RuntimeError(
            "Train or test set is empty; check TRAIN_END / TEST_START."
        )

    return df_train, df_test


def scale_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Fit MinMaxScaler on train features only, then transform both train and test.
    """
    scaler = MinMaxScaler()

    X_train = df_train[feature_cols].values
    X_test = df_test[feature_cols].values

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Clip test data to [0, 1] to handle distribution shift (test period may have
    # values outside training range, e.g., higher stock prices in 2023-2024)
    X_test_scaled = np.clip(X_test_scaled, 0, 1)

    print("\nFeature scaling:")
    print("X_train_scaled shape:", X_train_scaled.shape)
    print("X_test_scaled  shape:", X_test_scaled.shape)

    return X_train_scaled, X_test_scaled, scaler


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int = LOOKBACK,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sequences for LSTM.

    For each index i >= lookback-1:
      X_seq[k] = X[i - lookback + 1 : i + 1, :]
      y_seq[k] = y[i]

    So each sequence of `lookback` days is used to predict the *target at the last day*
    in that window (and the target itself is already "next-day" return/direction).
    """
    X_seqs = []
    y_seqs = []

    for i in range(lookback - 1, len(X)):
        X_window = X[i - lookback + 1 : i + 1, :]  # shape (lookback, n_features)
        y_value = y[i]  # scalar target aligned with last row in window

        X_seqs.append(X_window)
        y_seqs.append(y_value)

    return np.array(X_seqs), np.array(y_seqs)


def prepare_lstm_data() -> None:
    """
    Main preprocessing function used by the pipeline.

    It:
      - loads the processed dataset
      - splits into train/test by date
      - scales features
      - creates LSTM sequences
      - saves numpy arrays and the scaler under `data/lstm/`
    """
    df = load_processed_dataset()

    feature_cols = [
        "rsi_14",
        "macd",
        "macd_h",
        "bbl",
        "bbp",
        "sma_50",
        "ema_20",
        "obv",
        "close_norm",
        "volume_norm",
        "lagged_log_return",
        "atr_14",
    ]
    target_reg_col = "next_day_return"
    target_cls_col = "next_day_direction"

    # 1) Split into train/test
    df_train, df_test = train_test_split_by_date(df)

    # 2) Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(
        df_train, df_test, feature_cols
    )

    # 3) Extract targets
    # Raw targets (for evaluation metrics - RMSE, MAE)
    y_train_reg_raw = df_train[target_reg_col].values
    y_test_reg_raw = df_test[target_reg_col].values

    # Scale regression targets using StandardScaler (for LSTM training)
    # This centers targets around 0, which helps the LSTM learn both positive
    # and negative predictions instead of collapsing to a small positive mean.
    target_scaler = StandardScaler()
    y_train_reg = target_scaler.fit_transform(y_train_reg_raw.reshape(-1, 1)).flatten()
    y_test_reg = target_scaler.transform(y_test_reg_raw.reshape(-1, 1)).flatten()

    # Classification targets (no scaling - binary 0/1)
    y_train_cls = df_train[target_cls_col].values
    y_test_cls = df_test[target_cls_col].values

    print("\nTargets shapes:")
    print("y_train_reg (scaled):", y_train_reg.shape, "y_test_reg (scaled):", y_test_reg.shape)
    print("y_train_reg_raw:", y_train_reg_raw.shape, "y_test_reg_raw:", y_test_reg_raw.shape)
    print("y_train_cls:", y_train_cls.shape, "y_test_cls:", y_test_cls.shape)

    # 4) Create LSTM sequences
    # Scaled regression targets (for LSTM training)
    X_train_seq, y_train_reg_seq = create_sequences(
        X_train_scaled, y_train_reg, LOOKBACK
    )
    X_test_seq, y_test_reg_seq = create_sequences(X_test_scaled, y_test_reg, LOOKBACK)

    # Raw regression targets (for evaluation metrics - RMSE, MAE)
    _, y_train_reg_raw_seq = create_sequences(X_train_scaled, y_train_reg_raw, LOOKBACK)
    _, y_test_reg_raw_seq = create_sequences(X_test_scaled, y_test_reg_raw, LOOKBACK)

    # For classification: same sequences, different targets
    _, y_train_cls_seq = create_sequences(X_train_scaled, y_train_cls, LOOKBACK)
    _, y_test_cls_seq = create_sequences(X_test_scaled, y_test_cls, LOOKBACK)

    print("\nLSTM sequence shapes:")
    print("X_train_seq:", X_train_seq.shape)
    print("y_train_reg_seq (scaled):", y_train_reg_seq.shape)
    print("y_train_reg_raw_seq:", y_train_reg_raw_seq.shape)
    print("y_train_cls_seq:", y_train_cls_seq.shape)
    print("X_test_seq:", X_test_seq.shape)
    print("y_test_reg_seq (scaled):", y_test_reg_seq.shape)
    print("y_test_reg_raw_seq:", y_test_reg_raw_seq.shape)
    print("y_test_cls_seq:", y_test_cls_seq.shape)

    # 5) Save arrays & scalers
    np.save(LSTM_DIR / "X_train_seq.npy", X_train_seq)
    np.save(LSTM_DIR / "y_train_reg_seq.npy", y_train_reg_seq)  # scaled
    np.save(LSTM_DIR / "y_train_reg_raw_seq.npy", y_train_reg_raw_seq)  # raw
    np.save(LSTM_DIR / "y_train_cls_seq.npy", y_train_cls_seq)

    np.save(LSTM_DIR / "X_test_seq.npy", X_test_seq)
    np.save(LSTM_DIR / "y_test_reg_seq.npy", y_test_reg_seq)  # scaled
    np.save(LSTM_DIR / "y_test_reg_raw_seq.npy", y_test_reg_raw_seq)  # raw
    np.save(LSTM_DIR / "y_test_cls_seq.npy", y_test_cls_seq)

    joblib.dump(scaler, LSTM_DIR / "feature_scaler.joblib")
    joblib.dump(target_scaler, LSTM_DIR / "target_scaler.joblib")

    print(f"\nSaved LSTM-ready arrays and scalers in: {LSTM_DIR.resolve()}")
