"""
Feature engineering: build 12 technical indicators + 2 targets.

This module:
- Reads the raw S&P 500 CSV from `data/raw/`.
- Adds the required technical indicators (RSI, MACD, Bollinger, etc.).
- Adds targets:
    * next_day_return       (regression target, simple % return)
    * next_day_direction    (classification target, Up/Down)
- Saves the processed dataset under `data/processed/`.

Public helpers:
    - get_default_processed_csv_path()
    - build_and_save_feature_target_dataset()
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

from . import config
from .download_data import get_default_raw_csv_path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# src/finance_lstm/features.py -> src -> project root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = config.START_DATE[:4]
END_YEAR = config.END_DATE[:4]

RAW_CSV = get_default_raw_csv_path()
PROCESSED_CSV = PROCESSED_DIR / f"sp500_{START_YEAR}_{END_YEAR}_features_targets.csv"


def get_default_processed_csv_path() -> pathlib.Path:
    """
    Return the canonical path for the processed features/targets CSV.

    Example: data/processed/sp500_2018_2024_features_targets.csv
    """
    return PROCESSED_CSV


# ---------------------------------------------------------------------------
# Core feature engineering
# ---------------------------------------------------------------------------


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

    Definition (simple return, as agreed):
      next_day_return_t = ((Close_{t+1} - Close_t) / Close_t) * 100

    Direction rule (our choice, documented):
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


def main() -> None:
    """Manual entry point:

    python -m finance_lstm.features
    """
    build_and_save_feature_target_dataset()


if __name__ == "__main__":
    main()
