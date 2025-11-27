"""
LSTM model: build, train, and evaluate.

- Uses preprocessed sequences from `data/lstm/` created by `preprocessing.py`.
- Predicts next-day percentage return (regression).
- Derives direction (Up/Down) from the sign of the predicted return.

Public helper:
    - train_and_evaluate_lstm()
"""

from __future__ import annotations

import pathlib
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from .. import config
from ..visualization import plot_learning_curves

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# src/finance_lstm/models/lstm.py -> src -> project root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
LSTM_DIR = DATA_DIR / "lstm"
LSTM_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_lstm_data() -> Tuple[np.ndarray, ...]:
    """
    Load preprocessed LSTM-ready arrays for regression and classification.

    Expected files in `data/lstm/`:
        - X_train_seq.npy
        - y_train_reg_seq.npy
        - y_train_cls_seq.npy
        - X_test_seq.npy
        - y_test_reg_seq.npy
        - y_test_cls_seq.npy
    """
    X_train = np.load(LSTM_DIR / "X_train_seq.npy")
    y_train_reg = np.load(LSTM_DIR / "y_train_reg_seq.npy")
    y_train_cls = np.load(LSTM_DIR / "y_train_cls_seq.npy")

    X_test = np.load(LSTM_DIR / "X_test_seq.npy")
    y_test_reg = np.load(LSTM_DIR / "y_test_reg_seq.npy")
    y_test_cls = np.load(LSTM_DIR / "y_test_cls_seq.npy")

    # Cast to float32 for TF
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    y_train_reg = y_train_reg.astype("float32")
    y_test_reg = y_test_reg.astype("float32")

    print("Loaded LSTM data:")
    print("X_train:", X_train.shape)
    print("y_train_reg:", y_train_reg.shape)
    print("y_train_cls:", y_train_cls.shape)
    print("X_test:", X_test.shape)
    print("y_test_reg:", y_test_reg.shape)
    print("y_test_cls:", y_test_cls.shape)

    return X_train, y_train_reg, y_train_cls, X_test, y_test_reg, y_test_cls


# ---------------------------------------------------------------------------
# Model definition & training
# ---------------------------------------------------------------------------


def build_lstm_model(lookback: int, n_features: int) -> tf.keras.Model:
    """
    Build a Sequential model with stacked LSTM layers and a Dense(1) linear output
    for regression (next-day percentage return).

    Architecture (configurable via config.LSTM_CONFIG):
      - Input(shape=(lookback, n_features))
      - LSTM(units1, return_sequences=True)
      - Dropout(dropout)
      - LSTM(units2)
      - Dropout(dropout)
      - Dense(1, activation='linear')
    """
    units1 = config.LSTM_CONFIG["units1"]
    units2 = config.LSTM_CONFIG["units2"]
    dropout_rate = config.LSTM_CONFIG["dropout"]
    learning_rate = config.LSTM_CONFIG["learning_rate"]

    model = Sequential(
        [
            Input(shape=(lookback, n_features)),
            LSTM(units1, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units2),
            Dropout(dropout_rate),
            Dense(1, activation="linear"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",  # predicting return magnitude (regression)
        metrics=["mae"],
    )

    model.summary()
    return model


def train_lstm_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train_reg: np.ndarray,
    val_split: float = 0.2,
):
    """
    Train the LSTM model with early stopping and learning rate reduction.
    """
    batch_size = config.LSTM_CONFIG["batch_size"]
    epochs = config.LSTM_CONFIG["epochs"]

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(LSTM_DIR / "lstm_model_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train_reg,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=False,  # respect temporal order
    )

    return history


# ---------------------------------------------------------------------------
# Evaluation & prediction saving
# ---------------------------------------------------------------------------


def evaluate_and_save_predictions(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test_reg: np.ndarray,
    y_test_cls: np.ndarray,
) -> tuple[float, float]:
    """
    Evaluate regression performance and save predictions.

    IMPORTANT: For direction (classification), we **do not** train a separate
    classifier. Instead, we:

      1) Predict the next-day percentage return (continuous)
      2) Direction = 1 if predicted_return > 0, else 0

    This matches the project requirement:
      "If the predicted percentage return is positive, then Up; else Down."
    """
    # Regression metrics (MSE, MAE)
    test_loss, test_mae = model.evaluate(X_test, y_test_reg, verbose=1)
    print(f"\nTest MSE (loss): {test_loss:.4f}")
    print(f"Test MAE:        {test_mae:.4f}")

    # Predict returns
    y_pred_reg = model.predict(X_test, verbose=0).flatten()

    # Direction derived from sign
    y_pred_dir = (y_pred_reg > 0.0).astype(int)

    # Save predictions for later comparisons (baselines vs LSTM)
    np.save(LSTM_DIR / "y_pred_reg_lstm.npy", y_pred_reg)
    np.save(LSTM_DIR / "y_pred_dir_lstm.npy", y_pred_dir)

    print("\nSample of predicted vs true returns (first 5):")
    for i in range(min(5, len(y_test_reg))):
        print(
            f"Pred: {y_pred_reg[i]:7.3f} %, "
            f"True: {y_test_reg[i]:7.3f} %, "
            f"Pred_dir: {y_pred_dir[i]}, "
            f"True_dir: {int(y_test_cls[i])}"
        )

    print("\nSaved LSTM predictions to:", LSTM_DIR.resolve())
    return float(test_loss), float(test_mae)


# ---------------------------------------------------------------------------
# Public entry point for pipeline
# ---------------------------------------------------------------------------


def train_and_evaluate_lstm() -> tuple[float, float, tf.keras.callbacks.History]:
    """
    Full LSTM workflow:

      - Set random seeds for reproducibility
      - Load preprocessed LSTM data from `data/lstm/`
      - Build model based on config + data shape
      - Train model
      - Generate and save learning curve visualizations
      - Save final model
      - Evaluate on test set and save predictions

    Returns
    -------
    tuple[float, float, tf.keras.callbacks.History]
        - Test MSE (Mean Squared Error)
        - Test MAE (Mean Absolute Error)
        - Training history object (for potential further analysis)
    """
    # Set random seeds for reproducibility
    import random

    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)

    (
        X_train,
        y_train_reg,
        y_train_cls,  # unused but loaded for completeness
        X_test,
        y_test_reg,
        y_test_cls,
    ) = load_lstm_data()

    lookback = X_train.shape[1]
    n_features = X_train.shape[2]

    model = build_lstm_model(lookback=lookback, n_features=n_features)

    history = train_lstm_model(model, X_train, y_train_reg)

    # Generate and save learning curves
    plot_learning_curves(history)

    # Save final model (best one already stored via ModelCheckpoint)
    final_path = LSTM_DIR / "lstm_model_final.keras"
    model.save(str(final_path))
    print(f"\nSaved final LSTM model to: {final_path.resolve()}")

    test_mse, test_mae = evaluate_and_save_predictions(
        model, X_test, y_test_reg, y_test_cls
    )
    # Evaluate and store predictions
    return test_mse, test_mae, history


def main() -> None:
    """
    Manual entry point:

        python -m finance_lstm.models.lstm
    """
    train_and_evaluate_lstm()


if __name__ == "__main__":
    main()
