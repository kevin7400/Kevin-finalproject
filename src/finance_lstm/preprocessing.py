"""
Preprocessing to create LSTM-ready 3D sequences.

- Reads the processed features/targets CSV.
- Splits into train/test by date (config).
- Scales features with MinMaxScaler (fit on train only).
- Builds sequences of shape (samples, lookback, features).
- Saves numpy arrays + scaler in `data/lstm/`.

Public helper:
    - prepare_lstm_data()
"""

from __future__ import annotations

import pathlib
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from . import config
from .features import get_default_processed_csv_path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# src/finance_lstm/preprocessing.py -> src -> project root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LSTM_DIR = DATA_DIR / "lstm"
LSTM_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = config.LOOKBACK
N_FEATURES = 12  # number of indicators/inputs


def load_processed_dataset() -> pd.DataFrame:
    """
    Load the processed feature/target CSV built by `features.py`.
    """
    path = get_default_processed_csv_path()
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()

    print("Loaded processed dataset:")
    print("Shape:", df.shape)
    print("Date range:", df.index.min(), "->", df.index.max())

    return df


def train_test_split_by_date(
    df: pd.DataFrame,
    train_end: str = config.TRAIN_END,
    test_start: str = config.TEST_START,
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
            "Train or test set is empty; check TRAIN_END / TEST_START in config.py."
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

    # 3) Extract targets (no scaling)
    y_train_reg = df_train[target_reg_col].values
    y_test_reg = df_test[target_reg_col].values
    y_train_cls = df_train[target_cls_col].values
    y_test_cls = df_test[target_cls_col].values

    print("\nTargets shapes:")
    print("y_train_reg:", y_train_reg.shape, "y_test_reg:", y_test_reg.shape)
    print("y_train_cls:", y_train_cls.shape, "y_test_cls:", y_test_cls.shape)

    # 4) Create LSTM sequences
    X_train_seq, y_train_reg_seq = create_sequences(
        X_train_scaled, y_train_reg, LOOKBACK
    )
    X_test_seq, y_test_reg_seq = create_sequences(X_test_scaled, y_test_reg, LOOKBACK)

    # For classification: same sequences, different targets
    _, y_train_cls_seq = create_sequences(X_train_scaled, y_train_cls, LOOKBACK)
    _, y_test_cls_seq = create_sequences(X_test_scaled, y_test_cls, LOOKBACK)

    print("\nLSTM sequence shapes:")
    print("X_train_seq:", X_train_seq.shape)
    print("y_train_reg_seq:", y_train_reg_seq.shape)
    print("y_train_cls_seq:", y_train_cls_seq.shape)
    print("X_test_seq:", X_test_seq.shape)
    print("y_test_reg_seq:", y_test_reg_seq.shape)
    print("y_test_cls_seq:", y_test_cls_seq.shape)

    # 5) Save arrays & scaler
    np.save(LSTM_DIR / "X_train_seq.npy", X_train_seq)
    np.save(LSTM_DIR / "y_train_reg_seq.npy", y_train_reg_seq)
    np.save(LSTM_DIR / "y_train_cls_seq.npy", y_train_cls_seq)

    np.save(LSTM_DIR / "X_test_seq.npy", X_test_seq)
    np.save(LSTM_DIR / "y_test_reg_seq.npy", y_test_reg_seq)
    np.save(LSTM_DIR / "y_test_cls_seq.npy", y_test_cls_seq)

    joblib.dump(scaler, LSTM_DIR / "feature_scaler.joblib")

    print(f"\nSaved LSTM-ready arrays and scaler in: {LSTM_DIR.resolve()}")


def main() -> None:
    """
    Manual entry point:

        python -m finance_lstm.preprocessing
    """
    prepare_lstm_data()


if __name__ == "__main__":
    main()
