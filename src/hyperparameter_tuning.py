"""
Hyperparameter tuning module for stock prediction models.

This module provides grid search and random search for tuning:
- LinearRegression: classification threshold optimization
- RandomForest: n_estimators, max_depth, min_samples_split
- XGBoost: n_estimators, learning_rate, max_depth, gamma
- LSTM: units, dropout, learning_rate, batch_size

Data split for tuning:
- Training: 2018-01-01 to 2021-12-31 (4 years)
- Validation: 2022-01-01 to 2022-12-31 (1 year)
- Test: 2023-01-01 to 2024-12-31 (unchanged)

All hyperparameters are selected based on F1-score on the validation set.
"""

import json
import pathlib
from itertools import product
from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    balanced_accuracy_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBRegressor
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from src.data_loader import (
    RANDOM_SEED,
    LOOKBACK,
    N_FEATURES,
    START_DATE,
    TUNE_TRAIN_END,
    TUNE_VAL_START,
    TUNE_VAL_END,
    TEST_START,
    END_DATE,
    clip_returns_train_only,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TUNING_DIR = DATA_DIR / "tuning"
TUNING_DIR.mkdir(parents=True, exist_ok=True)
BEST_PARAMS_PATH = TUNING_DIR / "best_params.json"


# ---------------------------------------------------------------------------
# Hyperparameter Grids
# ---------------------------------------------------------------------------

# LinearRegression: threshold values to test
LR_THRESHOLD_GRID = np.linspace(-0.5, 0.5, 101)  # -0.5 to +0.5 in 0.01 steps

# RandomForest hyperparameters
RF_PARAM_GRID = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
}

# XGBoost hyperparameters
XGB_PARAM_GRID = {
    'n_estimators': [100, 300, 500, 700],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 8],
    'gamma': [0, 0.1, 0.2, 0.5],
}

# LSTM hyperparameters
LSTM_PARAM_GRID = {
    'units1': [32, 64, 128],
    'units2': [16, 32, 64],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
    'batch_size': [16, 32, 64],
    'loss': ['mse', 'mae', 'huber'],  # Loss function comparison
}

# Number of random samples for LSTM tuning
LSTM_N_SAMPLES = 50


# ---------------------------------------------------------------------------
# Threshold Tuning Helper
# ---------------------------------------------------------------------------


def find_best_threshold(
    y_true_cls: np.ndarray,
    y_pred_continuous: np.ndarray,
    metric: str = "f1",
    n_quantiles: int = 100,
) -> Tuple[float, float]:
    """
    Find optimal threshold using quantiles of prediction distribution.

    Instead of a fixed grid, this scans thresholds based on the actual
    prediction distribution, which adapts to model behavior.

    Args:
        y_true_cls: True binary labels (0=Down, 1=Up)
        y_pred_continuous: Continuous predictions (raw returns or probabilities)
        metric: "f1" for F1 score or "balanced_acc" for balanced accuracy
        n_quantiles: Number of quantile-based thresholds to try (default: 100)

    Returns:
        Tuple of (best_threshold, best_score)
    """
    # Generate thresholds from prediction quantiles (5th to 95th percentile)
    thresholds = np.percentile(y_pred_continuous, np.linspace(5, 95, n_quantiles))

    # Also include 0.0 and 0.5 as common thresholds
    thresholds = np.unique(np.concatenate([thresholds, [0.0, 0.5]]))

    best_threshold = 0.0
    best_score = 0.0

    for thresh in thresholds:
        y_pred_dir = (y_pred_continuous > thresh).astype(int)

        # Skip if all predictions are the same class
        if len(np.unique(y_pred_dir)) == 1:
            continue

        if metric == "f1":
            score = f1_score(y_true_cls, y_pred_dir, zero_division=0)
        elif metric == "balanced_acc":
            score = balanced_accuracy_score(y_true_cls, y_pred_dir)
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'f1' or 'balanced_acc'.")

        if score > best_score:
            best_score = score
            best_threshold = float(thresh)

    return best_threshold, best_score


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

def prepare_tuning_data() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,  # X_train, y_train_reg (scaled), y_train_cls
    np.ndarray, np.ndarray, np.ndarray,  # X_val, y_val_reg (scaled), y_val_cls
    np.ndarray, np.ndarray, np.ndarray,  # X_test, y_test_reg (scaled), y_test_cls
    Any,  # feature_scaler
    Any,  # target_scaler
    np.ndarray, np.ndarray, np.ndarray,  # y_train_reg_raw, y_val_reg_raw, y_test_reg_raw
]:
    """
    Prepare train/validation/test splits for hyperparameter tuning.

    Split boundaries:
        - Train: 2018-01-01 to 2021-12-31
        - Val:   2022-01-01 to 2022-12-31
        - Test:  2023-01-01 to 2024-12-31

    Returns:
        Tuple containing scaled features and targets for train/val/test sets,
        plus the fitted scalers and raw regression targets.

        For LSTM: Use scaled targets for training, raw targets + inverse transform for F1.
        For baselines: Use raw targets directly.
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from src.data_loader import build_and_save_feature_target_dataset, get_default_processed_csv_path

    # Load or build processed dataset
    csv_path = get_default_processed_csv_path()
    if not csv_path.exists():
        build_and_save_feature_target_dataset()

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()

    # Split by date
    df_train = df[df.index <= TUNE_TRAIN_END]
    df_val = df[(df.index >= TUNE_VAL_START) & (df.index <= TUNE_VAL_END)]
    df_test = df[df.index >= TEST_START]

    print(f"Tuning data splits:")
    print(f"  Train: {df_train.index.min().date()} to {df_train.index.max().date()} ({len(df_train)} rows)")
    print(f"  Val:   {df_val.index.min().date()} to {df_val.index.max().date()} ({len(df_val)} rows)")
    print(f"  Test:  {df_test.index.min().date()} to {df_test.index.max().date()} ({len(df_test)} rows)")

    # Separate features and targets
    target_reg_col = "next_day_return"
    target_cls_col = "next_day_direction"
    feature_cols = [c for c in df.columns if c not in [target_reg_col, target_cls_col]]

    X_train = df_train[feature_cols].values
    X_val = df_val[feature_cols].values
    X_test = df_test[feature_cols].values

    # Raw regression targets (for baselines and F1 calculation)
    y_train_reg_raw = df_train[target_reg_col].values
    y_val_reg_raw = df_val[target_reg_col].values
    y_test_reg_raw = df_test[target_reg_col].values

    # Clip outliers at μ ± 3σ (using training data only to avoid leakage)
    # This is consistent with prepare_lstm_data() in data_loader.py
    y_train_reg_raw, y_val_reg_raw, y_test_reg_raw = clip_returns_train_only(
        y_train_reg_raw, y_val_raw=y_val_reg_raw, y_test_raw=y_test_reg_raw, n_sigma=3.0
    )

    # Scale regression targets for LSTM (StandardScaler centers around 0)
    target_scaler = StandardScaler()
    y_train_reg = target_scaler.fit_transform(y_train_reg_raw.reshape(-1, 1)).flatten()
    y_val_reg = target_scaler.transform(y_val_reg_raw.reshape(-1, 1)).flatten()
    y_test_reg = target_scaler.transform(y_test_reg_raw.reshape(-1, 1)).flatten()

    y_train_cls = df_train[target_cls_col].values
    y_val_cls = df_val[target_cls_col].values
    y_test_cls = df_test[target_cls_col].values

    # Scale features (fit on train only)
    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    X_test_scaled = feature_scaler.transform(X_test)

    # Clip to [0, 1] to handle distribution shift
    X_val_scaled = np.clip(X_val_scaled, 0, 1)
    X_test_scaled = np.clip(X_test_scaled, 0, 1)

    return (
        X_train_scaled, y_train_reg, y_train_cls,
        X_val_scaled, y_val_reg, y_val_cls,
        X_test_scaled, y_test_reg, y_test_cls,
        feature_scaler,
        target_scaler,
        y_train_reg_raw, y_val_reg_raw, y_test_reg_raw,
    )


def create_sequences_for_tuning(
    X: np.ndarray,
    y_reg: np.ndarray,
    y_cls: np.ndarray,
    lookback: int = LOOKBACK,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sequences for LSTM from flat arrays."""
    n_samples = len(X) - lookback + 1
    n_features = X.shape[1]

    X_seq = np.zeros((n_samples, lookback, n_features))
    y_reg_seq = np.zeros(n_samples)
    y_cls_seq = np.zeros(n_samples)

    for i in range(n_samples):
        X_seq[i] = X[i:i + lookback]
        y_reg_seq[i] = y_reg[i + lookback - 1]
        y_cls_seq[i] = y_cls[i + lookback - 1]

    return X_seq, y_reg_seq, y_cls_seq


# ---------------------------------------------------------------------------
# F1 Score Helper
# ---------------------------------------------------------------------------

def compute_f1_from_predictions(
    y_true_cls: np.ndarray,
    y_pred_reg: np.ndarray,
    threshold: float = 0.0
) -> float:
    """
    Compute F1 score from regression predictions.

    Direction rule: pred_direction = 1 if y_pred_reg > threshold else 0
    """
    y_pred_dir = (y_pred_reg > threshold).astype(int)
    return float(f1_score(y_true_cls, y_pred_dir, zero_division=0))


# ---------------------------------------------------------------------------
# LinearRegression Threshold Tuning
# ---------------------------------------------------------------------------

def tune_linear_regression_threshold(
    X_train: np.ndarray,
    y_train_reg: np.ndarray,
    X_val: np.ndarray,
    y_val_reg: np.ndarray,
    y_val_cls: np.ndarray,
) -> Dict[str, Any]:
    """
    Tune the classification threshold for LinearRegression.

    LinearRegression has no hyperparameters, but we can optimize
    the threshold used to convert predicted returns to Up/Down.

    Returns:
        dict with 'best_threshold', 'best_f1', 'search_results'
    """
    print("\n--- Tuning LinearRegression threshold ---")

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train_reg)

    # Predict on validation
    y_pred_val = model.predict(X_val)

    best_f1 = -1.0
    best_threshold = 0.0
    results = []

    for threshold in tqdm(LR_THRESHOLD_GRID, desc="LR threshold"):
        f1 = compute_f1_from_predictions(y_val_cls, y_pred_val, threshold)
        results.append({'threshold': float(threshold), 'val_f1': f1})

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    print(f"Best threshold: {best_threshold:.4f}, Val F1: {best_f1:.4f}")

    return {
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'search_results': results,
    }


# ---------------------------------------------------------------------------
# RandomForest Tuning
# ---------------------------------------------------------------------------

def tune_random_forest(
    X_train: np.ndarray,
    y_train_reg: np.ndarray,
    X_val: np.ndarray,
    y_val_reg: np.ndarray,
    y_val_cls: np.ndarray,
    y_train_cls: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Grid search for RandomForest hyperparameters.

    Args:
        y_train_cls: Training classification targets for computing sample weights.
                     Prevents model from exploiting class imbalance.

    Returns:
        dict with 'best_params', 'best_f1', 'search_results'
    """
    print("\n--- Tuning RandomForest ---")

    # Compute sample weights to prevent class imbalance exploitation
    if y_train_cls is not None:
        sample_weight = compute_sample_weight('balanced', y_train_cls)
        print(f"Using balanced sample weights")
    else:
        sample_weight = None

    # Generate all combinations
    param_names = list(RF_PARAM_GRID.keys())
    param_values = list(RF_PARAM_GRID.values())
    combinations = list(product(*param_values))

    best_f1 = -1.0
    best_params = None
    results = []

    for combo in tqdm(combinations, desc="RF grid search"):
        params = dict(zip(param_names, combo))

        # Build and train model with sample weights
        model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(X_train, y_train_reg, sample_weight=sample_weight)

        # Evaluate on validation
        y_pred_val = model.predict(X_val)
        f1 = compute_f1_from_predictions(y_val_cls, y_pred_val)

        results.append({'params': params, 'val_f1': f1})

        if f1 > best_f1:
            best_f1 = f1
            best_params = params.copy()

    print(f"Best params: {best_params}, Val F1 (threshold=0): {best_f1:.4f}")

    # Retrain with best params and tune threshold
    print("Tuning threshold for RandomForest...")
    best_model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    best_model.fit(X_train, y_train_reg, sample_weight=sample_weight)
    y_pred_val = best_model.predict(X_val)

    best_threshold = 0.0
    best_f1_with_threshold = best_f1
    for threshold in LR_THRESHOLD_GRID:
        f1 = compute_f1_from_predictions(y_val_cls, y_pred_val, threshold)
        if f1 > best_f1_with_threshold:
            best_f1_with_threshold = f1
            best_threshold = float(threshold)

    print(f"Best threshold: {best_threshold:.4f}, Val F1: {best_f1_with_threshold:.4f}")

    return {
        'best_params': best_params,
        'best_threshold': best_threshold,
        'best_f1': best_f1_with_threshold,
        'search_results': results,
    }


# ---------------------------------------------------------------------------
# XGBoost Tuning
# ---------------------------------------------------------------------------

def tune_xgboost(
    X_train: np.ndarray,
    y_train_reg: np.ndarray,
    X_val: np.ndarray,
    y_val_reg: np.ndarray,
    y_val_cls: np.ndarray,
    y_train_cls: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Grid search for XGBoost hyperparameters.

    Args:
        y_train_cls: Training classification targets for computing sample weights.
                     Prevents model from exploiting class imbalance.

    Returns:
        dict with 'best_params', 'best_f1', 'search_results'
    """
    print("\n--- Tuning XGBoost ---")

    # Compute sample weights to prevent class imbalance exploitation
    if y_train_cls is not None:
        sample_weight = compute_sample_weight('balanced', y_train_cls)
        print(f"Using balanced sample weights")
    else:
        sample_weight = None

    # Generate all combinations
    param_names = list(XGB_PARAM_GRID.keys())
    param_values = list(XGB_PARAM_GRID.values())
    combinations = list(product(*param_values))

    best_f1 = -1.0
    best_params = None
    results = []

    for combo in tqdm(combinations, desc="XGB grid search"):
        params = dict(zip(param_names, combo))

        # Build and train model with sample weights
        model = XGBRegressor(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            gamma=params['gamma'],
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train, y_train_reg, sample_weight=sample_weight)

        # Evaluate on validation
        y_pred_val = model.predict(X_val)
        f1 = compute_f1_from_predictions(y_val_cls, y_pred_val)

        results.append({'params': params, 'val_f1': f1})

        if f1 > best_f1:
            best_f1 = f1
            best_params = params.copy()

    print(f"Best params: {best_params}, Val F1 (threshold=0): {best_f1:.4f}")

    # Retrain with best params and tune threshold
    print("Tuning threshold for XGBoost...")
    best_model = XGBRegressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        gamma=best_params['gamma'],
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    best_model.fit(X_train, y_train_reg, sample_weight=sample_weight)
    y_pred_val = best_model.predict(X_val)

    best_threshold = 0.0
    best_f1_with_threshold = best_f1
    for threshold in LR_THRESHOLD_GRID:
        f1 = compute_f1_from_predictions(y_val_cls, y_pred_val, threshold)
        if f1 > best_f1_with_threshold:
            best_f1_with_threshold = f1
            best_threshold = float(threshold)

    print(f"Best threshold: {best_threshold:.4f}, Val F1: {best_f1_with_threshold:.4f}")

    return {
        'best_params': best_params,
        'best_threshold': best_threshold,
        'best_f1': best_f1_with_threshold,
        'search_results': results,
    }


# ---------------------------------------------------------------------------
# LSTM Tuning
# ---------------------------------------------------------------------------

def build_lstm_with_params(
    lookback: int,
    n_features: int,
    params: Dict[str, Any],
) -> tf.keras.Model:
    """Build LSTM model with specified hyperparameters."""
    model = Sequential([
        Input(shape=(lookback, n_features)),
        LSTM(params['units1'], return_sequences=True),
        Dropout(params['dropout']),
        LSTM(params['units2']),
        Dropout(params['dropout']),
        Dense(1, activation='linear'),
    ])

    # Get loss function (default to mse for backward compatibility)
    loss_fn = params.get('loss', 'mse')

    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
        loss=loss_fn,
        metrics=['mae'],
    )
    return model


def tune_lstm(
    X_train_seq: np.ndarray,
    y_train_reg_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_reg_seq: np.ndarray,
    y_val_cls_seq: np.ndarray,
    target_scaler: Any,
    n_samples: int = LSTM_N_SAMPLES,
) -> Dict[str, Any]:
    """
    Random search for LSTM hyperparameters.

    Uses random sampling from the parameter grid for efficiency.

    Note: LSTM is trained on scaled targets. For F1 calculation, predictions
    are inverse transformed to raw scale, then direction is derived from sign.

    Args:
        X_train_seq: Training sequences (scaled features).
        y_train_reg_seq: Training regression targets (scaled).
        X_val_seq: Validation sequences (scaled features).
        y_val_reg_seq: Validation regression targets (scaled).
        y_val_cls_seq: Validation classification targets (0/1).
        target_scaler: Fitted StandardScaler for inverse transform.
        n_samples: Number of random combinations to try.

    Returns:
        dict with 'best_params', 'best_f1', 'search_results'
    """
    print(f"\n--- Tuning LSTM (random search, {n_samples} samples) ---")

    # Generate all possible combinations
    param_names = list(LSTM_PARAM_GRID.keys())
    param_values = list(LSTM_PARAM_GRID.values())
    all_combinations = list(product(*param_values))

    # Random sample
    np.random.seed(RANDOM_SEED)
    if n_samples < len(all_combinations):
        indices = np.random.choice(len(all_combinations), n_samples, replace=False)
        sampled_combinations = [all_combinations[i] for i in indices]
    else:
        sampled_combinations = all_combinations

    lookback = X_train_seq.shape[1]
    n_features = X_train_seq.shape[2]

    best_f1 = -1.0
    best_params = None
    results = []

    for combo in tqdm(sampled_combinations, desc="LSTM random search"):
        params = dict(zip(param_names, combo))
        params['epochs'] = 50  # Max epochs (early stopping will cut short)

        try:
            # Build model
            model = build_lstm_with_params(lookback, n_features, params)

            # Train with early stopping
            history = model.fit(
                X_train_seq, y_train_reg_seq,
                validation_data=(X_val_seq, y_val_reg_seq),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True,
                        verbose=0,
                    )
                ],
                verbose=0,
            )

            # Evaluate F1 on validation
            # Predictions are in scaled space, so inverse transform before computing F1
            y_pred_val_scaled = model.predict(X_val_seq, verbose=0).flatten()
            y_pred_val_raw = target_scaler.inverse_transform(
                y_pred_val_scaled.reshape(-1, 1)
            ).flatten()

            # Compute F1 using raw predictions (direction = sign of raw return)
            y_pred_dir = (y_pred_val_raw > 0.0).astype(int)
            f1 = float(f1_score(y_val_cls_seq, y_pred_dir))

            results.append({'params': params, 'val_f1': f1})

            if f1 > best_f1:
                best_f1 = f1
                best_params = params.copy()

        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue

        finally:
            # Clear session to prevent memory buildup
            tf.keras.backend.clear_session()

    print(f"Best params: {best_params}, Val F1: {best_f1:.4f}")

    # Threshold tuning: retrain best model and find optimal threshold
    best_threshold = 0.0  # Default threshold
    best_threshold_f1 = best_f1

    if best_params is not None:
        print("\n--- Tuning threshold for best LSTM model ---")
        try:
            # Retrain best model
            model = build_lstm_with_params(lookback, n_features, best_params)
            model.fit(
                X_train_seq, y_train_reg_seq,
                validation_data=(X_val_seq, y_val_reg_seq),
                epochs=best_params.get('epochs', 50),
                batch_size=best_params['batch_size'],
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True,
                        verbose=0,
                    )
                ],
                verbose=0,
            )

            # Get raw predictions on validation
            y_pred_val_scaled = model.predict(X_val_seq, verbose=0).flatten()
            y_pred_val_raw = target_scaler.inverse_transform(
                y_pred_val_scaled.reshape(-1, 1)
            ).flatten()

            # Find best threshold using validation predictions
            best_threshold, best_threshold_f1 = find_best_threshold(
                y_val_cls_seq, y_pred_val_raw, metric="f1"
            )
            print(f"Tuned threshold: {best_threshold:.4f}, F1: {best_threshold_f1:.4f}")

        except Exception as e:
            print(f"Warning: Threshold tuning failed: {e}")
        finally:
            tf.keras.backend.clear_session()

    return {
        'best_params': best_params,
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'best_threshold_f1': best_threshold_f1,
        'search_results': results,
    }


def build_lstm_classifier_with_params(
    lookback: int,
    n_features: int,
    params: Dict[str, Any],
) -> tf.keras.Model:
    """Build LSTM classifier model with specified hyperparameters."""
    model = Sequential([
        Input(shape=(lookback, n_features)),
        LSTM(params['units1'], return_sequences=True),
        Dropout(params['dropout']),
        LSTM(params['units2']),
        Dropout(params['dropout']),
        Dense(1, activation='sigmoid'),  # Sigmoid for binary classification
    ])

    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
        loss='binary_crossentropy',  # Classification loss
        metrics=['accuracy'],
    )
    return model


# LSTM Classifier hyperparameter grid (no 'loss' - always binary_crossentropy)
LSTM_CLASSIFIER_PARAM_GRID = {
    'units1': [32, 64, 128],
    'units2': [16, 32, 64],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
    'batch_size': [16, 32, 64],
}


def tune_lstm_classifier(
    X_train_seq: np.ndarray,
    y_train_cls_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_cls_seq: np.ndarray,
    n_samples: int = LSTM_N_SAMPLES,
) -> Dict[str, Any]:
    """
    Random search for LSTM classifier hyperparameters.

    Uses random sampling from the parameter grid for efficiency.
    Optimizes directly for F1 score on binary classification task.

    Args:
        X_train_seq: Training sequences (scaled features).
        y_train_cls_seq: Training classification targets (0/1).
        X_val_seq: Validation sequences (scaled features).
        y_val_cls_seq: Validation classification targets (0/1).
        n_samples: Number of random combinations to try.

    Returns:
        dict with 'best_params', 'best_f1', 'best_threshold', 'search_results'
    """
    from sklearn.utils.class_weight import compute_class_weight

    print(f"\n--- Tuning LSTM Classifier (random search, {n_samples} samples) ---")

    # Generate all possible combinations
    param_names = list(LSTM_CLASSIFIER_PARAM_GRID.keys())
    param_values = list(LSTM_CLASSIFIER_PARAM_GRID.values())
    all_combinations = list(product(*param_values))

    # Random sample
    np.random.seed(RANDOM_SEED)
    if n_samples < len(all_combinations):
        indices = np.random.choice(len(all_combinations), n_samples, replace=False)
        sampled_combinations = [all_combinations[i] for i in indices]
    else:
        sampled_combinations = all_combinations

    lookback = X_train_seq.shape[1]
    n_features = X_train_seq.shape[2]

    # Compute class weights
    classes = np.array([0, 1])
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_cls_seq.astype(int))
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"Using class weights: Down={class_weights[0]:.3f}, Up={class_weights[1]:.3f}")

    best_f1 = -1.0
    best_params = None
    best_threshold = 0.5
    results = []

    for combo in tqdm(sampled_combinations, desc="LSTM Classifier random search"):
        params = dict(zip(param_names, combo))
        params['epochs'] = 50  # Max epochs (early stopping will cut short)

        try:
            # Build classifier model
            model = build_lstm_classifier_with_params(lookback, n_features, params)

            # Train with early stopping and class weights
            history = model.fit(
                X_train_seq, y_train_cls_seq,
                validation_data=(X_val_seq, y_val_cls_seq),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                class_weight=class_weight_dict,
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True,
                        verbose=0,
                    )
                ],
                verbose=0,
            )

            # Get predicted probabilities
            y_pred_proba = model.predict(X_val_seq, verbose=0).flatten()

            # Find best threshold
            thresholds = np.linspace(0.3, 0.7, 41)  # 0.3 to 0.7 in 0.01 steps
            best_thresh_f1 = -1.0
            best_thresh = 0.5

            for thresh in thresholds:
                y_pred_dir = (y_pred_proba > thresh).astype(int)
                f1 = float(f1_score(y_val_cls_seq, y_pred_dir))
                if f1 > best_thresh_f1:
                    best_thresh_f1 = f1
                    best_thresh = thresh

            results.append({
                'params': params,
                'val_f1': best_thresh_f1,
                'threshold': best_thresh,
            })

            if best_thresh_f1 > best_f1:
                best_f1 = best_thresh_f1
                best_params = params.copy()
                best_threshold = best_thresh

        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue

        finally:
            # Clear session to prevent memory buildup
            tf.keras.backend.clear_session()

    print(f"Best params: {best_params}, threshold: {best_threshold:.3f}, Val F1: {best_f1:.4f}")

    return {
        'best_params': best_params,
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'search_results': results,
    }


# LSTM Multitask hyperparameter grid
LSTM_MULTITASK_PARAM_GRID = {
    'units1': [32, 64, 128],
    'units2': [16, 32, 64],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
    'batch_size': [16, 32, 64],
    'alpha_return': [0.3, 0.5, 0.7],  # Loss weight for return head
}


def build_lstm_multitask_with_params(
    lookback: int,
    n_features: int,
    params: Dict[str, Any],
) -> tf.keras.Model:
    """Build LSTM multitask model with specified hyperparameters."""
    from tensorflow.keras.layers import Input as KerasInput
    from tensorflow.keras.models import Model

    alpha_return = params.get('alpha_return', 0.5)

    # Shared trunk using Functional API
    inputs = KerasInput(shape=(lookback, n_features), name="input")
    x = LSTM(params['units1'], return_sequences=True, name="lstm1")(inputs)
    x = Dropout(params['dropout'], name="dropout1")(x)
    x = LSTM(params['units2'], name="lstm2")(x)
    x = Dropout(params['dropout'], name="dropout2")(x)

    # Regression head (returns)
    return_out = Dense(1, activation="linear", name="return_out")(x)

    # Classification head (direction)
    dir_out = Dense(1, activation="sigmoid", name="dir_out")(x)

    model = Model(inputs=inputs, outputs=[return_out, dir_out], name="lstm_multitask")

    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
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
    return model


def tune_lstm_multitask(
    X_train_seq: np.ndarray,
    y_train_reg_seq: np.ndarray,
    y_train_cls_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_reg_seq: np.ndarray,
    y_val_cls_seq: np.ndarray,
    target_scaler: Any,
    n_samples: int = LSTM_N_SAMPLES,
) -> Dict[str, Any]:
    """
    Random search for LSTM multitask hyperparameters.

    Optimizes for combined performance: regression (RMSE) and direction (F1).

    Args:
        X_train_seq: Training sequences (scaled features).
        y_train_reg_seq: Training regression targets (scaled).
        y_train_cls_seq: Training classification targets (0/1).
        X_val_seq: Validation sequences (scaled features).
        y_val_reg_seq: Validation regression targets (scaled).
        y_val_cls_seq: Validation classification targets (0/1).
        target_scaler: Fitted StandardScaler for inverse transform.
        n_samples: Number of random combinations to try.

    Returns:
        dict with 'best_params', 'best_f1', 'best_threshold', 'search_results'
    """
    print(f"\n--- Tuning LSTM Multitask (random search, {n_samples} samples) ---")

    # Generate all possible combinations
    param_names = list(LSTM_MULTITASK_PARAM_GRID.keys())
    param_values = list(LSTM_MULTITASK_PARAM_GRID.values())
    all_combinations = list(product(*param_values))

    # Random sample
    np.random.seed(RANDOM_SEED)
    if n_samples < len(all_combinations):
        indices = np.random.choice(len(all_combinations), n_samples, replace=False)
        sampled_combinations = [all_combinations[i] for i in indices]
    else:
        sampled_combinations = all_combinations

    lookback = X_train_seq.shape[1]
    n_features = X_train_seq.shape[2]

    # Note: We don't use sample_weight during tuning to avoid Keras compatibility
    # issues with dict-style outputs. The final training will use sample_weight.
    # Tuning focuses on finding optimal hyperparameters using F1 on validation set.

    best_f1 = -1.0
    best_params = None
    best_threshold = 0.5
    results = []

    for combo in tqdm(sampled_combinations, desc="LSTM Multitask random search"):
        params = dict(zip(param_names, combo))
        params['epochs'] = 50  # Max epochs (early stopping will cut short)

        try:
            # Build multitask model
            model = build_lstm_multitask_with_params(lookback, n_features, params)

            # Train with early stopping (no sample_weight to avoid Keras dict issues)
            history = model.fit(
                X_train_seq,
                [y_train_reg_seq, y_train_cls_seq],  # List format for outputs
                validation_split=0.1,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True,
                        verbose=0,
                    )
                ],
                verbose=0,
                shuffle=False,  # Respect temporal order
            )

            # Get predictions from both heads
            y_pred_reg_scaled, y_pred_proba = model.predict(X_val_seq, verbose=0)
            y_pred_proba = y_pred_proba.flatten()

            # Find best threshold for direction head
            best_thresh, best_thresh_f1 = find_best_threshold(
                y_val_cls_seq, y_pred_proba, metric="f1"
            )

            results.append({
                'params': params,
                'val_f1': best_thresh_f1,
                'threshold': best_thresh,
            })

            if best_thresh_f1 > best_f1:
                best_f1 = best_thresh_f1
                best_params = params.copy()
                best_threshold = best_thresh

        except Exception as e:
            import traceback
            print(f"Error with params {params}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        finally:
            # Clear session to prevent memory buildup
            tf.keras.backend.clear_session()

    print(f"Best params: {best_params}, threshold: {best_threshold:.3f}, Val F1: {best_f1:.4f}")

    return {
        'best_params': best_params,
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'search_results': results,
    }


# ---------------------------------------------------------------------------
# Save/Load Best Parameters
# ---------------------------------------------------------------------------

def save_best_params(results: Dict[str, Any]) -> None:
    """Save best hyperparameters to JSON."""
    # Convert to serializable format
    serializable = {}
    for model_name, data in results.items():
        if model_name == 'LinearRegression':
            serializable[model_name] = {
                'best_threshold': data['best_threshold'],
                'best_f1': float(data['best_f1']),
            }
        else:
            # Skip if best_params is None (all trials failed)
            if data.get('best_params') is None:
                print(f"Warning: Skipping {model_name} - no valid params found (all trials failed)")
                continue

            # Convert numpy types to Python natives
            params = {}
            for k, v in data['best_params'].items():
                if isinstance(v, (np.integer, np.floating)):
                    params[k] = float(v) if isinstance(v, np.floating) else int(v)
                elif v is None:
                    params[k] = None
                else:
                    params[k] = v

            serializable[model_name] = {
                'best_params': params,
                'best_threshold': data.get('best_threshold', 0.0),
                'best_f1': float(data['best_f1']),
            }

    with open(BEST_PARAMS_PATH, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\nSaved best parameters to: {BEST_PARAMS_PATH}")


def load_best_params() -> Dict[str, Any]:
    """Load previously saved best hyperparameters."""
    if not BEST_PARAMS_PATH.exists():
        raise FileNotFoundError(f"No saved parameters found at {BEST_PARAMS_PATH}")

    with open(BEST_PARAMS_PATH, 'r') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main Tuning Entry Point
# ---------------------------------------------------------------------------

def tune_all_models(save_results: bool = True, mode: str = "regressor") -> Dict[str, Any]:
    """
    Run hyperparameter tuning for all models.

    Process:
        1. Prepare train/val/test data with tuning splits
        2. Tune each model on validation set (optimize F1)
        3. Save best hyperparameters
        4. Return summary of best configurations

    Args:
        save_results: Whether to save results to JSON file.
        mode: LSTM training mode - "regressor", "classifier", or "multitask".

    Returns:
        dict mapping model_name -> tuning results
    """
    mode_display = {
        "regressor": "REGRESSOR (MSE/MAE/Huber)",
        "classifier": "CLASSIFIER (binary_crossentropy)",
        "multitask": "MULTITASK (shared trunk, dual heads)",
    }
    print("=" * 60)
    print("HYPERPARAMETER TUNING")
    print(f"LSTM Mode: {mode_display.get(mode, mode)}")
    print("=" * 60)

    # Prepare data
    # Note: y_train_reg, y_val_reg, y_test_reg are SCALED (for LSTM)
    # y_train_reg_raw, y_val_reg_raw, y_test_reg_raw are RAW (for baselines and F1)
    (
        X_train, y_train_reg, y_train_cls,
        X_val, y_val_reg, y_val_cls,
        X_test, y_test_reg, y_test_cls,
        feature_scaler,
        target_scaler,
        y_train_reg_raw, y_val_reg_raw, y_test_reg_raw,
    ) = prepare_tuning_data()

    results = {}

    # 1. Tune LinearRegression threshold (use RAW targets)
    lr_results = tune_linear_regression_threshold(
        X_train, y_train_reg_raw, X_val, y_val_reg_raw, y_val_cls
    )
    results['LinearRegression'] = lr_results

    # 2. Tune RandomForest (use RAW targets, with sample weights)
    rf_results = tune_random_forest(
        X_train, y_train_reg_raw, X_val, y_val_reg_raw, y_val_cls,
        y_train_cls=y_train_cls,
    )
    results['RandomForest'] = rf_results

    # 3. Tune XGBoost (use RAW targets, with sample weights)
    xgb_results = tune_xgboost(
        X_train, y_train_reg_raw, X_val, y_val_reg_raw, y_val_cls,
        y_train_cls=y_train_cls,
    )
    results['XGBoost'] = xgb_results

    # 4. Tune LSTM (needs sequential data)
    print("\nPreparing LSTM sequences...")
    X_train_seq, y_train_reg_seq, y_train_cls_seq = create_sequences_for_tuning(
        X_train, y_train_reg, y_train_cls  # scaled targets
    )
    X_val_seq, y_val_reg_seq, y_val_cls_seq = create_sequences_for_tuning(
        X_val, y_val_reg, y_val_cls  # scaled targets
    )
    print(f"  X_train_seq: {X_train_seq.shape}, X_val_seq: {X_val_seq.shape}")

    if mode == "classifier":
        # Tune LSTM as classifier
        lstm_results = tune_lstm_classifier(
            X_train_seq, y_train_cls_seq,
            X_val_seq, y_val_cls_seq,
        )
        results['LSTMClassifier'] = lstm_results
    elif mode == "multitask":
        # Tune LSTM as multitask (both regression and classification)
        lstm_results = tune_lstm_multitask(
            X_train_seq, y_train_reg_seq, y_train_cls_seq,
            X_val_seq, y_val_reg_seq, y_val_cls_seq,
            target_scaler,
        )
        results['LSTMMultiTask'] = lstm_results
    else:
        # Tune LSTM as regressor (default)
        lstm_results = tune_lstm(
            X_train_seq, y_train_reg_seq,
            X_val_seq, y_val_reg_seq, y_val_cls_seq,
            target_scaler,  # Pass scaler for inverse transform during F1 calculation
        )
        results['LSTM'] = lstm_results

    # Save results
    if save_results:
        save_best_params(results)

    # Print summary
    print("\n" + "=" * 60)
    print("TUNING SUMMARY")
    print("=" * 60)
    for model_name, data in results.items():
        if model_name == 'LinearRegression':
            print(f"{model_name}: threshold={data['best_threshold']:.4f}, Val F1={data['best_f1']:.4f}")
        elif model_name in ('LSTMClassifier', 'LSTMMultiTask', 'LSTM'):
            threshold_str = f", threshold={data['best_threshold']:.4f}" if 'best_threshold' in data else ""
            print(f"{model_name}: {data['best_params']}{threshold_str}, Val F1={data['best_f1']:.4f}")
        else:
            print(f"{model_name}: {data['best_params']}, Val F1={data['best_f1']:.4f}")

    return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tune_all_models()
