"""
LSTM and baseline models.

This module consolidates:
- LSTM model architecture, training, and evaluation (from models/lstm.py)
- Baseline models: Linear Regression, Random Forest, XGBoost (from models/baselines.py)

Public functions:
    - train_and_evaluate_lstm()              # Regression approach (predict returns)
    - train_and_evaluate_lstm_classifier()   # Classification approach (predict direction)
    - train_baseline_models()
    - evaluate_regression_and_direction()
"""

from __future__ import annotations

import pathlib
from typing import Tuple

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBRegressor

# Import config from data_loader
from src.data_loader import RANDOM_SEED, LSTM_CONFIG

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# src/models.py -> project root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LSTM_DIR = DATA_DIR / "lstm"
LSTM_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# LSTM: Data loading
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
# LSTM: Model definition & training
# ---------------------------------------------------------------------------


def build_lstm_model(
    lookback: int,
    n_features: int,
    config: dict = None,
) -> tf.keras.Model:
    """
    Build a Sequential model with stacked LSTM layers and a Dense(1) linear output
    for regression (next-day percentage return).

    Architecture (configurable via config or LSTM_CONFIG):
      - Input(shape=(lookback, n_features))
      - LSTM(units1, return_sequences=True)
      - Dropout(dropout)
      - LSTM(units2)
      - Dropout(dropout)
      - Dense(1, activation='linear')

    Args:
        lookback: Number of timesteps in input sequence.
        n_features: Number of features per timestep.
        config: Optional dict with keys 'units1', 'units2', 'dropout', 'learning_rate'.
                If None, uses LSTM_CONFIG defaults.

    Returns:
        Compiled Keras model.
    """
    # Merge with defaults
    cfg = LSTM_CONFIG.copy()
    if config:
        cfg.update(config)

    units1 = cfg["units1"]
    units2 = cfg["units2"]
    dropout_rate = cfg["dropout"]
    learning_rate = cfg["learning_rate"]
    loss_fn = cfg.get("loss", "mse")  # Default to MSE if not specified

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
        loss=loss_fn,  # configurable: "mse", "mae", "huber"
        metrics=["mae"],
    )

    model.summary()
    return model


def build_lstm_classifier(
    lookback: int,
    n_features: int,
    config: dict = None,
) -> tf.keras.Model:
    """
    Build a Sequential LSTM model for binary classification (Up/Down direction).

    This is an alternative to the regression approach. Instead of predicting
    returns and deriving direction from sign, this model directly predicts
    the probability of an "Up" day.

    Architecture:
      - Input(shape=(lookback, n_features))
      - LSTM(units1, return_sequences=True)
      - Dropout(dropout)
      - LSTM(units2)
      - Dropout(dropout)
      - Dense(1, activation='sigmoid')  # Binary classification output

    Args:
        lookback: Number of timesteps in input sequence.
        n_features: Number of features per timestep.
        config: Optional dict with keys 'units1', 'units2', 'dropout', 'learning_rate'.
                If None, uses LSTM_CONFIG defaults.

    Returns:
        Compiled Keras model with binary_crossentropy loss.
    """
    # Merge with defaults
    cfg = LSTM_CONFIG.copy()
    if config:
        cfg.update(config)

    units1 = cfg["units1"]
    units2 = cfg["units2"]
    dropout_rate = cfg["dropout"]
    learning_rate = cfg["learning_rate"]

    model = Sequential(
        [
            Input(shape=(lookback, n_features)),
            LSTM(units1, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units2),
            Dropout(dropout_rate),
            Dense(1, activation="sigmoid"),  # Sigmoid for binary classification
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",  # Classification loss
        metrics=["accuracy"],
    )

    model.summary()
    return model


def build_lstm_multitask(
    lookback: int,
    n_features: int,
    config: dict = None,
) -> tf.keras.Model:
    """
    Build a multitask LSTM with shared trunk and two output heads.

    This model learns both return magnitude (regression) and direction
    (classification) jointly using a shared representation.

    Architecture:
        Input -> LSTM(units1) -> Dropout -> LSTM(units2) -> Dropout
                            |
                    +-------+-------+
                    |               |
                Dense(1)        Dense(1)
                linear          sigmoid
                return_out      dir_out

    Args:
        lookback: Number of timesteps in input sequence.
        n_features: Number of features per timestep.
        config: Optional dict with keys 'units1', 'units2', 'dropout', 'learning_rate',
                'alpha_return' (loss weight for return head, default 0.5).

    Returns:
        Compiled Keras model with two outputs.
    """
    # Merge with defaults
    cfg = LSTM_CONFIG.copy()
    if config:
        cfg.update(config)

    units1 = cfg["units1"]
    units2 = cfg["units2"]
    dropout_rate = cfg["dropout"]
    learning_rate = cfg["learning_rate"]
    alpha_return = cfg.get("alpha_return", 0.5)  # Loss weight for return head

    # Shared trunk using Functional API
    inputs = Input(shape=(lookback, n_features), name="input")
    x = LSTM(units1, return_sequences=True, name="lstm1")(inputs)
    x = Dropout(dropout_rate, name="dropout1")(x)
    x = LSTM(units2, name="lstm2")(x)
    x = Dropout(dropout_rate, name="dropout2")(x)

    # Regression head (returns)
    return_out = Dense(1, activation="linear", name="return_out")(x)

    # Classification head (direction)
    dir_out = Dense(1, activation="sigmoid", name="dir_out")(x)

    model = Model(inputs=inputs, outputs=[return_out, dir_out], name="lstm_multitask")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "return_out": "mse",
            "dir_out": "binary_crossentropy",
        },
        loss_weights={
            "return_out": alpha_return,
            "dir_out": 1.0 - alpha_return,
        },
        metrics={
            "return_out": ["mae"],
            "dir_out": ["accuracy"],
        },
    )

    model.summary()
    return model


def train_lstm_classifier(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train_cls: np.ndarray,
    val_split: float = 0.2,
    use_class_weights: bool = True,
):
    """
    Train the LSTM classifier with early stopping, learning rate reduction,
    and optional class weights for handling imbalanced datasets.

    Args:
        model: Compiled Keras classifier model.
        X_train: Training features.
        y_train_cls: Training classification labels (0/1).
        val_split: Validation split ratio.
        use_class_weights: If True, compute balanced class weights.

    Returns:
        Training history object.
    """
    batch_size = LSTM_CONFIG["batch_size"]
    epochs = LSTM_CONFIG["epochs"]

    # Compute class weights for imbalanced data
    class_weight_dict = None
    if use_class_weights:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.array([0, 1])
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train_cls.astype(int))
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        print(f"\nUsing class weights: Down={class_weights[0]:.3f}, Up={class_weights[1]:.3f}")

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
            filepath=str(LSTM_DIR / "lstm_classifier_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train_cls,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
        shuffle=False,  # respect temporal order
    )

    return history


def train_lstm_multitask(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train_reg: np.ndarray,
    y_train_cls: np.ndarray,
    config: dict = None,
    val_split: float = 0.2,
    use_class_weights: bool = True,
) -> tf.keras.callbacks.History:
    """
    Train the multitask LSTM with balanced sample weights for direction head.

    Args:
        model: Compiled multitask Keras model with 'return_out' and 'dir_out' heads.
        X_train: Training features.
        y_train_reg: Training regression targets (scaled).
        y_train_cls: Training classification targets (0/1).
        config: Optional config dict with 'batch_size', 'epochs'.
        val_split: Validation split ratio.
        use_class_weights: If True, compute balanced sample weights for direction head.

    Returns:
        Training history object.
    """
    # Merge with defaults
    cfg = LSTM_CONFIG.copy()
    if config:
        cfg.update(config)

    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]

    # Compute sample weights for direction head (class imbalance)
    # Note: We use list-style outputs to avoid Keras compatibility issues
    # with dict-style sample_weight. Sample weight is applied uniformly
    # to both outputs since Keras doesn't support per-output weights with lists.
    sample_weight = None
    if use_class_weights:
        dir_weights = compute_sample_weight('balanced', y_train_cls.astype(int))
        sample_weight = dir_weights  # Applied to combined loss
        print(f"\nUsing sample weights for direction head:")
        print(f"  Down weight: {dir_weights[y_train_cls == 0].mean():.3f}")
        print(f"  Up weight: {dir_weights[y_train_cls == 1].mean():.3f}")

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
            filepath=str(LSTM_DIR / "lstm_multitask_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Use list-style outputs for compatibility
    history = model.fit(
        X_train,
        [y_train_reg, y_train_cls],  # List format instead of dict
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        sample_weight=sample_weight,
        verbose=1,
        shuffle=False,  # respect temporal order
    )

    return history


def train_lstm_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train_reg: np.ndarray,
    val_split: float = 0.2,
):
    """
    Train the LSTM model with early stopping and learning rate reduction.
    """
    batch_size = LSTM_CONFIG["batch_size"]
    epochs = LSTM_CONFIG["epochs"]

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
# LSTM: Evaluation & prediction saving
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

    Note: The LSTM is trained on scaled targets (StandardScaler), so we must
    inverse transform predictions before deriving direction and computing
    metrics on raw percentage returns.
    """
    # Load target scaler for inverse transform
    target_scaler = joblib.load(LSTM_DIR / "target_scaler.joblib")

    # Load raw test targets for proper RMSE/MAE calculation
    y_test_reg_raw = np.load(LSTM_DIR / "y_test_reg_raw_seq.npy")

    # Model loss on scaled targets (for consistency with training)
    test_loss_scaled, test_mae_scaled = model.evaluate(X_test, y_test_reg, verbose=1)
    print(f"\nTest MSE (scaled): {test_loss_scaled:.4f}")
    print(f"Test MAE (scaled): {test_mae_scaled:.4f}")

    # Predict scaled returns
    y_pred_reg_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse transform to original scale (raw percentage returns)
    y_pred_reg = target_scaler.inverse_transform(
        y_pred_reg_scaled.reshape(-1, 1)
    ).flatten()

    # Compute metrics on raw scale
    test_mse = float(mean_squared_error(y_test_reg_raw, y_pred_reg))
    test_mae = float(mean_absolute_error(y_test_reg_raw, y_pred_reg))
    print(f"\nTest MSE (raw %): {test_mse:.4f}")
    print(f"Test MAE (raw %): {test_mae:.4f}")

    # Direction derived from sign of raw predictions
    y_pred_dir = (y_pred_reg > 0.0).astype(int)

    # Save predictions (in raw scale) for later comparisons (baselines vs LSTM)
    np.save(LSTM_DIR / "y_pred_reg_lstm.npy", y_pred_reg)
    np.save(LSTM_DIR / "y_pred_dir_lstm.npy", y_pred_dir)

    print("\nSample of predicted vs true returns (first 5):")
    for i in range(min(5, len(y_test_reg_raw))):
        print(
            f"Pred: {y_pred_reg[i]:7.3f} %, "
            f"True: {y_test_reg_raw[i]:7.3f} %, "
            f"Pred_dir: {y_pred_dir[i]}, "
            f"True_dir: {int(y_test_cls[i])}"
        )

    print("\nSaved LSTM predictions to:", LSTM_DIR.resolve())
    return test_mse, test_mae


def evaluate_and_save_classifier_predictions(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test_cls: np.ndarray,
    y_test_reg_raw: np.ndarray = None,
    threshold: float = None,
) -> dict:
    """
    Evaluate LSTM classifier and save predictions.

    For the classifier approach, we:
      1) Predict probability of Up (sigmoid output)
      2) Direction = 1 if probability > threshold, else 0

    Args:
        model: Trained classifier model.
        X_test: Test features.
        y_test_cls: True direction labels (0/1).
        y_test_reg_raw: Optional raw returns for RMSE/MAE calculation.
        threshold: Classification threshold. If None, tries to load from
                   tuned params or defaults to 0.5.

    Returns:
        Dict with accuracy, f1, precision, recall metrics.
    """
    # Load tuned threshold if not provided
    if threshold is None:
        try:
            from src.hyperparameter_tuning import load_best_params
            params = load_best_params()
            threshold = params.get('LSTMClassifier', {}).get('best_threshold', 0.5)
            print(f"Loaded tuned threshold for classifier: {threshold}")
        except (FileNotFoundError, KeyError):
            threshold = 0.5
            print(f"Using default threshold: {threshold}")

    # Predict probabilities
    y_pred_proba = model.predict(X_test, verbose=0).flatten()

    # Apply threshold for direction
    y_pred_dir = (y_pred_proba > threshold).astype(int)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test_cls, verbose=1)
    print(f"\nTest Loss (binary_crossentropy): {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Compute classification metrics
    acc = float(accuracy_score(y_test_cls, y_pred_dir))
    f1 = float(f1_score(y_test_cls, y_pred_dir))
    precision = float(precision_score(y_test_cls, y_pred_dir, zero_division=0))
    recall = float(recall_score(y_test_cls, y_pred_dir, zero_division=0))

    print(f"\nClassification Metrics (threshold={threshold}):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")

    # Prediction distribution
    n_pred_up = y_pred_dir.sum()
    n_pred_down = len(y_pred_dir) - n_pred_up
    print(f"\nPrediction distribution: Up={n_pred_up} ({100*n_pred_up/len(y_pred_dir):.1f}%), "
          f"Down={n_pred_down} ({100*n_pred_down/len(y_pred_dir):.1f}%)")

    # Actual distribution
    n_actual_up = y_test_cls.sum()
    n_actual_down = len(y_test_cls) - n_actual_up
    print(f"Actual distribution:     Up={n_actual_up} ({100*n_actual_up/len(y_test_cls):.1f}%), "
          f"Down={n_actual_down} ({100*n_actual_down/len(y_test_cls):.1f}%)")

    # Save predictions
    np.save(LSTM_DIR / "y_pred_proba_lstm_classifier.npy", y_pred_proba)
    np.save(LSTM_DIR / "y_pred_dir_lstm_classifier.npy", y_pred_dir)

    # Also save as standard LSTM prediction files for evaluation.py compatibility
    np.save(LSTM_DIR / "y_pred_dir_lstm.npy", y_pred_dir)

    # For RMSE/MAE, we can use probability as a proxy for return direction strength
    # Or load raw returns if provided
    if y_test_reg_raw is not None:
        # Map predictions to pseudo-returns for magnitude comparison
        # This is for compatibility - classifier doesn't predict magnitude
        mean_up = y_test_reg_raw[y_test_cls == 1].mean() if (y_test_cls == 1).any() else 0.5
        mean_down = y_test_reg_raw[y_test_cls == 0].mean() if (y_test_cls == 0).any() else -0.5
        y_pred_reg = np.where(y_pred_dir == 1, mean_up, mean_down)
        np.save(LSTM_DIR / "y_pred_reg_lstm.npy", y_pred_reg)

    print("\nSample predictions (first 5):")
    for i in range(min(5, len(y_test_cls))):
        print(
            f"Prob: {y_pred_proba[i]:.3f}, "
            f"Pred_dir: {y_pred_dir[i]}, "
            f"True_dir: {int(y_test_cls[i])}"
        )

    print("\nSaved LSTM classifier predictions to:", LSTM_DIR.resolve())

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'threshold': threshold,
    }


def evaluate_and_save_multitask_predictions(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test_reg_raw: np.ndarray,
    y_test_cls: np.ndarray,
    threshold: float = None,
) -> dict:
    """
    Evaluate multitask LSTM and save predictions for both outputs.

    For the multitask model, we:
      1) Get regression predictions from return_out head
      2) Get direction probabilities from dir_out head
      3) Apply threshold to direction probabilities
      4) Compute metrics for both tasks

    Args:
        model: Trained multitask model with 'return_out' and 'dir_out' heads.
        X_test: Test features.
        y_test_reg_raw: True raw returns (for RMSE/MAE).
        y_test_cls: True direction labels (0/1).
        threshold: Classification threshold for direction head.
                   If None, tries to load from tuned params or defaults to 0.5.

    Returns:
        Dict with rmse, mae, accuracy, f1, precision, recall metrics.
    """
    # Load tuned threshold if not provided
    if threshold is None:
        try:
            from src.hyperparameter_tuning import load_best_params
            params = load_best_params()
            threshold = params.get('LSTMMultiTask', {}).get('best_threshold', 0.5)
            print(f"Loaded tuned threshold for multitask: {threshold}")
        except (FileNotFoundError, KeyError):
            threshold = 0.5
            print(f"Using default threshold: {threshold}")

    # Get predictions from both heads
    predictions = model.predict(X_test, verbose=0)
    y_pred_reg_scaled = predictions[0].flatten()
    y_pred_proba = predictions[1].flatten()

    # Inverse transform regression predictions to raw scale
    target_scaler = joblib.load(LSTM_DIR / "target_scaler.joblib")
    y_pred_reg_raw = target_scaler.inverse_transform(
        y_pred_reg_scaled.reshape(-1, 1)
    ).flatten()

    # Apply threshold to direction predictions
    y_pred_dir = (y_pred_proba > threshold).astype(int)

    # Compute regression metrics
    rmse = float(np.sqrt(mean_squared_error(y_test_reg_raw, y_pred_reg_raw)))
    mae = float(mean_absolute_error(y_test_reg_raw, y_pred_reg_raw))

    # Compute classification metrics
    acc = float(accuracy_score(y_test_cls, y_pred_dir))
    f1 = float(f1_score(y_test_cls, y_pred_dir))
    precision = float(precision_score(y_test_cls, y_pred_dir, zero_division=0))
    recall = float(recall_score(y_test_cls, y_pred_dir, zero_division=0))

    print("\n" + "="*60)
    print("MULTITASK LSTM EVALUATION")
    print("="*60)
    print(f"\nRegression Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"\nClassification Metrics (threshold={threshold}):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")

    # Prediction distribution
    n_pred_up = y_pred_dir.sum()
    n_pred_down = len(y_pred_dir) - n_pred_up
    print(f"\nPrediction distribution: Up={n_pred_up} ({100*n_pred_up/len(y_pred_dir):.1f}%), "
          f"Down={n_pred_down} ({100*n_pred_down/len(y_pred_dir):.1f}%)")

    # Actual distribution
    n_actual_up = int(y_test_cls.sum())
    n_actual_down = len(y_test_cls) - n_actual_up
    print(f"Actual distribution:     Up={n_actual_up} ({100*n_actual_up/len(y_test_cls):.1f}%), "
          f"Down={n_actual_down} ({100*n_actual_down/len(y_test_cls):.1f}%)")

    # Save predictions for evaluation.py compatibility
    np.save(LSTM_DIR / "y_pred_reg_lstm.npy", y_pred_reg_raw)
    np.save(LSTM_DIR / "y_pred_dir_lstm.npy", y_pred_dir)
    np.save(LSTM_DIR / "y_pred_proba_lstm_multitask.npy", y_pred_proba)

    print("\nSample predictions (first 5):")
    for i in range(min(5, len(y_test_cls))):
        print(
            f"Pred_return: {y_pred_reg_raw[i]:+.3f}%, "
            f"Pred_prob: {y_pred_proba[i]:.3f}, "
            f"Pred_dir: {y_pred_dir[i]}, "
            f"True_return: {y_test_reg_raw[i]:+.3f}%, "
            f"True_dir: {int(y_test_cls[i])}"
        )

    print("\nSaved LSTM multitask predictions to:", LSTM_DIR.resolve())

    return {
        'rmse': rmse,
        'mae': mae,
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'threshold': threshold,
    }


# ---------------------------------------------------------------------------
# LSTM: Public entry point
# ---------------------------------------------------------------------------


def train_and_evaluate_lstm_multitask(config: dict = None) -> tuple[dict, tf.keras.callbacks.History]:
    """
    Full multitask LSTM workflow:

      - Set random seeds for reproducibility
      - Load preprocessed LSTM data from `data/lstm/`
      - Build multitask model with shared trunk and two heads
      - Train model with sample weights for class imbalance
      - Save final model
      - Evaluate on test set and save predictions

    This is the MULTITASK approach that jointly learns return magnitude
    (regression) and direction (classification) using a shared representation.

    Args:
        config: Optional dict with LSTM hyperparameters (units1, units2, dropout,
                learning_rate, alpha_return). If None, uses LSTM_CONFIG defaults.

    Returns:
        tuple[dict, tf.keras.callbacks.History]
            - Dict with rmse, mae, accuracy, f1, precision, recall metrics
            - Training history object
    """
    # Set random seeds for reproducibility
    import random

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    (
        X_train,
        y_train_reg,
        y_train_cls,
        X_test,
        y_test_reg,
        y_test_cls,
    ) = load_lstm_data()

    # Load raw returns for evaluation
    y_test_reg_raw = np.load(LSTM_DIR / "y_test_reg_raw_seq.npy")

    lookback = X_train.shape[1]
    n_features = X_train.shape[2]

    print("\n" + "="*60)
    print("LSTM MULTITASK MODE")
    print("Training to predict both returns (regression) and direction (classification)")
    print("="*60 + "\n")

    model = build_lstm_multitask(lookback=lookback, n_features=n_features, config=config)

    history = train_lstm_multitask(model, X_train, y_train_reg, y_train_cls, config=config)

    # Save final model
    final_path = LSTM_DIR / "lstm_multitask_final.keras"
    model.save(str(final_path))
    print(f"\nSaved final LSTM multitask model to: {final_path.resolve()}")

    metrics = evaluate_and_save_multitask_predictions(
        model, X_test, y_test_reg_raw, y_test_cls
    )

    return metrics, history


def train_and_evaluate_lstm_classifier(config: dict = None) -> tuple[dict, tf.keras.callbacks.History]:
    """
    Full LSTM classifier workflow:

      - Set random seeds for reproducibility
      - Load preprocessed LSTM data from `data/lstm/`
      - Build classifier model based on config + data shape
      - Train model with class weights
      - Save final model
      - Evaluate on test set and save predictions

    This is the CLASSIFICATION approach that directly predicts Up/Down direction
    instead of regressing returns and deriving direction from sign.

    Args:
        config: Optional dict with LSTM hyperparameters (units1, units2, dropout,
                learning_rate). If None, uses LSTM_CONFIG defaults.

    Returns:
        tuple[dict, tf.keras.callbacks.History]
            - Dict with accuracy, f1, precision, recall metrics
            - Training history object
    """
    # Set random seeds for reproducibility
    import random

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    (
        X_train,
        y_train_reg,  # unused for classifier
        y_train_cls,
        X_test,
        y_test_reg,  # unused for classifier
        y_test_cls,
    ) = load_lstm_data()

    # Load raw returns for optional magnitude metrics
    y_test_reg_raw = np.load(LSTM_DIR / "y_test_reg_raw_seq.npy")

    lookback = X_train.shape[1]
    n_features = X_train.shape[2]

    print("\n" + "="*60)
    print("LSTM CLASSIFIER MODE")
    print("Training to directly predict Up/Down direction")
    print("="*60 + "\n")

    model = build_lstm_classifier(lookback=lookback, n_features=n_features, config=config)

    history = train_lstm_classifier(model, X_train, y_train_cls)

    # Save final model
    final_path = LSTM_DIR / "lstm_classifier_final.keras"
    model.save(str(final_path))
    print(f"\nSaved final LSTM classifier to: {final_path.resolve()}")

    metrics = evaluate_and_save_classifier_predictions(
        model, X_test, y_test_cls, y_test_reg_raw
    )

    return metrics, history


def train_and_evaluate_lstm(config: dict = None) -> tuple[float, float, tf.keras.callbacks.History]:
    """
    Full LSTM workflow:

      - Set random seeds for reproducibility
      - Load preprocessed LSTM data from `data/lstm/`
      - Build model based on config + data shape
      - Train model
      - Save final model
      - Evaluate on test set and save predictions

    Args:
        config: Optional dict with LSTM hyperparameters (units1, units2, dropout,
                learning_rate). If None, uses LSTM_CONFIG defaults.

    Returns
    -------
    tuple[float, float, tf.keras.callbacks.History]
        - Test MSE (Mean Squared Error)
        - Test MAE (Mean Absolute Error)
        - Training history object (for potential further analysis)
    """
    # Set random seeds for reproducibility
    import random

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

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

    model = build_lstm_model(lookback=lookback, n_features=n_features, config=config)

    history = train_lstm_model(model, X_train, y_train_reg)

    # Save final model (best one already stored via ModelCheckpoint)
    final_path = LSTM_DIR / "lstm_model_final.keras"
    model.save(str(final_path))
    print(f"\nSaved final LSTM model to: {final_path.resolve()}")

    test_mse, test_mae = evaluate_and_save_predictions(
        model, X_test, y_test_reg, y_test_cls
    )
    # Evaluate and store predictions
    return test_mse, test_mae, history


# ---------------------------------------------------------------------------
# Baseline Models (from baselines.py)
# ---------------------------------------------------------------------------


def train_baseline_models(
    X_train: np.ndarray,
    y_train_reg: np.ndarray,
    y_train_cls: np.ndarray = None,
    rf_params: dict = None,
    xgb_params: dict = None,
    use_sample_weights: bool = True,
):
    """
    Train 3 regression baselines:
      - LinearRegression
      - RandomForestRegressor
      - XGBRegressor

    All are trained to predict next_day_return (regression).
    Direction metrics will be based on the sign of the predicted return.

    Args:
        X_train: Training features.
        y_train_reg: Training regression targets.
        y_train_cls: Training classification targets (0/1 direction).
                     Used to compute sample weights for class balancing.
        rf_params: Optional dict of RandomForest hyperparameters.
        xgb_params: Optional dict of XGBoost hyperparameters.
        use_sample_weights: If True and y_train_cls provided, use balanced
                           sample weights to prevent class imbalance exploitation.

    Returns:
        Dict mapping model name to fitted model.
    """
    models = {}

    # Compute sample weights if class labels provided
    # This prevents models from gaming F1 by always predicting the majority class
    if use_sample_weights and y_train_cls is not None:
        sample_weight = compute_sample_weight('balanced', y_train_cls)
        print(f"Using balanced sample weights (Down weight: {sample_weight[y_train_cls == 0].mean():.3f}, "
              f"Up weight: {sample_weight[y_train_cls == 1].mean():.3f})")
    else:
        sample_weight = None

    # Linear baseline (no sample weights - too simple to benefit)
    lin = LinearRegression()
    lin.fit(X_train, y_train_reg)
    models["LinearRegression"] = lin

    # Random Forest baseline with sample weights
    rf_defaults = {
        'n_estimators': 300,
        'max_depth': None,
        'min_samples_split': 2,
        'random_state': 42,
        'n_jobs': -1,
    }
    if rf_params:
        rf_defaults.update(rf_params)
    rf = RandomForestRegressor(**rf_defaults)
    rf.fit(X_train, y_train_reg, sample_weight=sample_weight)
    models["RandomForest"] = rf

    # XGBoost baseline with sample weights
    xgb_defaults = {
        'n_estimators': 500,
        'max_depth': 4,
        'learning_rate': 0.05,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1,
    }
    if xgb_params:
        xgb_defaults.update(xgb_params)
    xgb = XGBRegressor(**xgb_defaults)
    xgb.fit(X_train, y_train_reg, sample_weight=sample_weight)
    models["XGBoost"] = xgb

    return models


def evaluate_regression_and_direction(
    y_true_reg: np.ndarray,
    y_true_cls: np.ndarray,
    y_pred_reg: np.ndarray,
):
    """
    Compute RMSE, MAE for regression & Accuracy, F1, Precision, Recall for direction, using:
      - y_pred_reg (continuous %) for magnitude
      - sign(y_pred_reg) as classification (Up if > 0, else Down)
    """
    rmse = float(np.sqrt(mean_squared_error(y_true_reg, y_pred_reg)))
    mae = float(mean_absolute_error(y_true_reg, y_pred_reg))

    y_pred_dir = (y_pred_reg > 0.0).astype(int)

    acc = float(accuracy_score(y_true_cls, y_pred_dir))
    f1 = float(f1_score(y_true_cls, y_pred_dir))

    from sklearn.metrics import precision_score, recall_score
    precision = float(precision_score(y_true_cls, y_pred_dir, zero_division=0))
    recall = float(recall_score(y_true_cls, y_pred_dir, zero_division=0))

    return rmse, mae, acc, f1, precision, recall
