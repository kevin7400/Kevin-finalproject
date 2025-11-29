"""
Model evaluation and visualization.

This module consolidates:
- Model evaluation logic: train baselines, compute metrics, compare models (from evaluation.py)
- Visualization: learning curves and confusion matrices (from visualization.py)

Public functions:
    - plot_learning_curves()
    - plot_confusion_matrix()
    - plot_all_confusion_matrices()
    - evaluate_all_models()
"""

from __future__ import annotations

import pathlib
from typing import Tuple, Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import MinMaxScaler

# Import from data_loader
from src.data_loader import (
    TRAIN_END,
    TEST_START,
    START_DATE,
    END_DATE,
    LOOKBACK,
    get_default_processed_csv_path,
)

# Import from models
from src.models import train_baseline_models

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# src/evaluation.py -> project root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LSTM_DIR = DATA_DIR / "lstm"
RESULTS_DIR = PROJECT_ROOT / "results"  # Changed from data/results/ to results/

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Visualization Functions (from visualization.py)
# ---------------------------------------------------------------------------


def plot_learning_curves(
    history: Any,
    save_path: pathlib.Path | None = None,
) -> None:
    """
    Plot LSTM training and validation loss/MAE over epochs.

    Creates a figure with two subplots:
    - Left: MSE (Mean Squared Error) loss
    - Right: MAE (Mean Absolute Error)

    Both plots show training and validation curves for comparison.

    Parameters
    ----------
    history : Any
        Keras History object from model.fit() containing training metrics.
        Must have keys: 'loss', 'val_loss', 'mae', 'val_mae'.
    save_path : pathlib.Path | None, optional
        Where to save the plot. Defaults to results/learning_curves.png.

    Returns
    -------
    None
        The plot is saved to disk.

    Raises
    ------
    ValueError
        If history is None, missing required attributes, or has missing keys.

    Examples
    --------
    >>> history = model.fit(X_train, y_train, validation_split=0.2, epochs=50)
    >>> plot_learning_curves(history)
    Learning curves saved to: .../results/learning_curves.png
    """
    if history is None or not hasattr(history, "history"):
        raise ValueError("Invalid history object: must have 'history' attribute")

    required_keys = ["loss", "val_loss", "mae", "val_mae"]
    for key in required_keys:
        if key not in history.history:
            raise ValueError(f"History missing required key: {key}")

    if save_path is None:
        save_path = RESULTS_DIR / "learning_curves.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot MSE (loss)
    epochs = range(1, len(history.history["loss"]) + 1)
    ax1.plot(
        epochs,
        history.history["loss"],
        "b-",
        label="Training Loss (MSE)",
        linewidth=2,
    )
    ax1.plot(
        epochs,
        history.history["val_loss"],
        "r-",
        label="Validation Loss (MSE)",
        linewidth=2,
    )
    ax1.set_title("Model Loss (MSE) Over Epochs", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("MSE", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot MAE
    ax2.plot(
        epochs, history.history["mae"], "b-", label="Training MAE", linewidth=2
    )
    ax2.plot(
        epochs,
        history.history["val_mae"],
        "r-",
        label="Validation MAE",
        linewidth=2,
    )
    ax2.set_title("Model MAE Over Epochs", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("MAE", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nLearning curves saved to: {save_path.resolve()}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: pathlib.Path | None = None,
) -> None:
    """
    Generate and save a confusion matrix heatmap for direction classification.

    Parameters
    ----------
    y_true : np.ndarray
        True direction labels (0 = Down, 1 = Up).
    y_pred : np.ndarray
        Predicted direction labels (0 = Down, 1 = Up).
    model_name : str
        Name of the model (for title and filename).
    save_path : pathlib.Path | None, optional
        Where to save the plot. Defaults to results/confusion_matrix_{model_name}.png.

    Returns
    -------
    None
        The plot is saved to disk.

    Raises
    ------
    ValueError
        If y_true and y_pred have different shapes.

    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> y_pred = np.array([0, 1, 0, 0, 1])
    >>> plot_confusion_matrix(y_true, y_pred, "LSTM")
    Confusion matrix for LSTM saved to: .../confusion_matrix_lstm.png
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    if save_path is None:
        safe_name = model_name.lower().replace(" ", "_")
        save_path = RESULTS_DIR / f"confusion_matrix_{safe_name}.png"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        square=True,
        cbar_kws={"label": "Count"},
        ax=ax,
        xticklabels=["Down (0)", "Up (1)"],
        yticklabels=["Down (0)", "Up (1)"],
    )

    ax.set_title(
        f"Confusion Matrix: {model_name}\nDirection Classification",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(
        f"Confusion matrix for {model_name} saved to: {save_path.resolve()}"
    )


def plot_all_confusion_matrices(
    models_dict: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Generate confusion matrices for multiple models.

    Convenience function to batch-generate confusion matrices for all models
    being compared.

    Parameters
    ----------
    models_dict : dict[str, tuple[np.ndarray, np.ndarray]]
        Dictionary with model names as keys and (y_true, y_pred) tuples as values.

    Returns
    -------
    None
        All plots are saved to disk.

    Examples
    --------
    >>> models_dict = {
    ...     "LSTM": (y_test_true, y_test_pred_lstm),
    ...     "Random Forest": (y_test_true, y_test_pred_rf),
    ... }
    >>> plot_all_confusion_matrices(models_dict)
    Confusion matrix for LSTM saved to: ...
    Confusion matrix for Random Forest saved to: ...
    """
    for model_name, (y_true, y_pred) in models_dict.items():
        plot_confusion_matrix(y_true, y_pred, model_name)


# ---------------------------------------------------------------------------
# Evaluation Functions (from evaluation.py)
# ---------------------------------------------------------------------------


def load_processed_dataset(
    csv_path: pathlib.Path | None = None,
) -> pd.DataFrame:
    """Load the processed features/targets dataset from CSV.

    Args:
        csv_path: Path to the processed CSV. If None, uses the default from config.

    Returns:
        Time-indexed pandas DataFrame sorted by date.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If the dataset is empty.
    """
    path = csv_path or get_default_processed_csv_path()
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at: {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if df.empty:
        raise ValueError(f"Processed dataset at {path} is empty.")

    print(
        "Processed dataset loaded:",
        df.shape,
        df.index.min(),
        "->",
        df.index.max(),
    )
    return df


def train_test_split_by_date(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time series DataFrame into train and test sets by date.

    Args:
        df: Full time-indexed DataFrame.
        train_end: Last date (inclusive) assigned to training.
        test_start: First date (inclusive) assigned to test.

    Returns:
        A tuple (df_train, df_test).

    Raises:
        RuntimeError: If either the train or test set is empty.
    """
    df_train = df[df.index <= train_end].copy()
    df_test = df[df.index >= test_start].copy()

    print(
        "Train period:",
        df_train.index.min(),
        "->",
        df_train.index.max(),
        "| rows:",
        len(df_train),
    )
    print(
        "Test  period:",
        df_test.index.min(),
        "->",
        df_test.index.max(),
        "| rows:",
        len(df_test),
    )

    if df_train.empty or df_test.empty:
        raise RuntimeError(
            "Train or test set is empty; check date bounds in config."
        )

    return df_train, df_test


def prepare_features_and_targets(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Prepare scaled features and targets for baseline models.

    Features (12 indicators):
        rsi_14, macd, macd_h, bbl, bbp, sma_50, ema_20,
        obv, close_norm, volume_norm, lagged_log_return, atr_14

    Targets:
        - next_day_return (regression)
        - next_day_direction (classification)

    Args:
        df_train: Training subset of the processed dataset.
        df_test: Test subset of the processed dataset.

    Returns:
        A 6-tuple:
            (
                X_train_scaled,
                X_test_scaled,
                y_train_reg,
                y_test_reg,
                y_train_cls,
                y_test_cls
            )
    """
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

    X_train = df_train[feature_cols].values
    X_test = df_test[feature_cols].values

    y_train_reg = df_train[target_reg_col].values
    y_test_reg = df_test[target_reg_col].values

    y_train_cls = df_train[target_cls_col].values
    y_test_cls = df_test[target_cls_col].values

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(
        "X_train_scaled:",
        X_train_scaled.shape,
        "| X_test_scaled:",
        X_test_scaled.shape,
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train_reg,
        y_test_reg,
        y_train_cls,
        y_test_cls,
    )


def evaluate_regression_and_direction(
    y_true_reg: np.ndarray,
    y_true_cls: np.ndarray,
    y_pred_reg: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Compute RMSE/MAE for regression and Accuracy/F1 for direction.

    Direction is derived from the sign of y_pred_reg:
        - y_pred_dir = 1 if y_pred_reg > 0, else 0

    Args:
        y_true_reg: True regression targets (percentage returns).
        y_true_cls: True direction labels (0/1).
        y_pred_reg: Predicted regression values.

    Returns:
        A tuple (rmse, mae, accuracy, f1).
    """
    rmse = float(np.sqrt(mean_squared_error(y_true_reg, y_pred_reg)))
    mae = float(mean_absolute_error(y_true_reg, y_pred_reg))

    y_pred_dir = (y_pred_reg > 0.0).astype(int)

    acc = float(accuracy_score(y_true_cls, y_pred_dir))
    f1 = float(f1_score(y_true_cls, y_pred_dir))

    return rmse, mae, acc, f1


def load_lstm_predictions(
    lstm_dir: pathlib.Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load LSTM test targets and predictions from disk.

    Expects the following files in data/lstm/ (or lstm_dir):
        - y_test_reg_seq.npy
        - y_test_cls_seq.npy
        - y_pred_reg_lstm.npy
        - y_pred_dir_lstm.npy

    Returns:
        A 4-tuple (y_test_reg_seq, y_test_cls_seq, y_pred_reg_lstm, y_pred_dir_lstm).

    Raises:
        FileNotFoundError: If any required file is missing.
        AssertionError: If the shapes do not match.
    """
    dir_path = lstm_dir or LSTM_DIR

    files = {
        "y_test_reg_seq": dir_path / "y_test_reg_seq.npy",
        "y_test_cls_seq": dir_path / "y_test_cls_seq.npy",
        "y_pred_reg_lstm": dir_path / "y_pred_reg_lstm.npy",
        "y_pred_dir_lstm": dir_path / "y_pred_dir_lstm.npy",
    }

    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Required LSTM output not found: {path}")

    y_test_reg_seq = np.load(files["y_test_reg_seq"])
    y_test_cls_seq = np.load(files["y_test_cls_seq"])
    y_pred_reg_lstm = np.load(files["y_pred_reg_lstm"])
    y_pred_dir_lstm = np.load(files["y_pred_dir_lstm"])

    print("\nLSTM arrays loaded:")
    print("  y_test_reg_seq:", y_test_reg_seq.shape)
    print("  y_test_cls_seq:", y_test_cls_seq.shape)
    print("  y_pred_reg_lstm:", y_pred_reg_lstm.shape)
    print("  y_pred_dir_lstm:", y_pred_dir_lstm.shape)

    assert y_test_reg_seq.shape == y_pred_reg_lstm.shape
    assert y_test_cls_seq.shape == y_pred_dir_lstm.shape

    return y_test_reg_seq, y_test_cls_seq, y_pred_reg_lstm, y_pred_dir_lstm


def evaluate_all_models() -> pd.DataFrame:
    """Train baselines, evaluate them and the LSTM, and save comparison table.

    Steps:
        1. Load processed dataset and split into train/test by date.
        2. Prepare scaled features and targets for baselines.
        3. Train LinearRegression, RandomForestRegressor, and XGBRegressor.
        4. Align test subset with LSTM's effective test window (drop first LOOKBACK-1).
        5. Compute RMSE/MAE/Accuracy/F1 for each baseline.
        6. Load LSTM predictions and compute the same metrics.
        7. Save the comparison as results/model_comparison.csv.

    Returns:
        A pandas DataFrame with one row per model and metrics columns.
    """
    df = load_processed_dataset()
    df_train, df_test = train_test_split_by_date(df)

    (
        X_train_scaled,
        X_test_scaled,
        y_train_reg,
        y_test_reg,
        y_train_cls,
        y_test_cls,
    ) = prepare_features_and_targets(df_train, df_test)

    # Train baselines
    models = train_baseline_models(X_train_scaled, y_train_reg)

    # Align baseline evaluation with LSTM test sequences:
    # LSTM test length = len(df_test) - (LOOKBACK - 1)
    # => drop first LOOKBACK-1 test samples for fair comparison.
    y_test_reg_eval = y_test_reg[LOOKBACK - 1 :]
    y_test_cls_eval = y_test_cls[LOOKBACK - 1 :]

    print(
        "\nTest eval target shapes for baselines:",
        y_test_reg_eval.shape,
        y_test_cls_eval.shape,
    )

    rows = []

    # Evaluate each baseline
    for name, model in models.items():
        y_pred_reg_full = model.predict(X_test_scaled)
        y_pred_reg_eval = y_pred_reg_full[LOOKBACK - 1 :]

        rmse, mae, acc, f1 = evaluate_regression_and_direction(
            y_test_reg_eval, y_test_cls_eval, y_pred_reg_eval
        )

        rows.append(
            {
                "model": name,
                "rmse": rmse,
                "mae": mae,
                "accuracy": acc,
                "f1": f1,
            }
        )

    # Evaluate LSTM using its own saved predictions
    (
        y_test_reg_seq,
        y_test_cls_seq,
        y_pred_reg_lstm,
        y_pred_dir_lstm,
    ) = load_lstm_predictions()

    rmse_lstm = float(np.sqrt(mean_squared_error(y_test_reg_seq, y_pred_reg_lstm)))
    mae_lstm = float(mean_absolute_error(y_test_reg_seq, y_pred_reg_lstm))
    acc_lstm = float(accuracy_score(y_test_cls_seq, y_pred_dir_lstm))
    f1_lstm = float(f1_score(y_test_cls_seq, y_pred_dir_lstm))

    rows.append(
        {
            "model": "LSTM",
            "rmse": rmse_lstm,
            "mae": mae_lstm,
            "accuracy": acc_lstm,
            "f1": f1_lstm,
        }
    )

    # Build comparison table
    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(by="rmse")

    print(
        "\n=== Model Comparison on Test Set "
        f"({START_DATE[:4]}–{TRAIN_END[:4]} train, "
        f"{TEST_START[:4]}–{END_DATE[:4]} test) ==="
    )
    print(results_df.to_string(index=False))

    out_path = RESULTS_DIR / "model_comparison.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved results table to: {out_path.resolve()}")

    # Generate confusion matrices for all models
    print("\n=== Generating Confusion Matrices ===")
    confusion_data = {}

    # Add baselines
    for name, model in models.items():
        y_pred_reg_full = model.predict(X_test_scaled)
        y_pred_reg_eval = y_pred_reg_full[LOOKBACK - 1 :]
        y_pred_dir_eval = (y_pred_reg_eval > 0.0).astype(int)
        confusion_data[name] = (y_test_cls_eval, y_pred_dir_eval)

    # Add LSTM
    confusion_data["LSTM"] = (y_test_cls_seq, y_pred_dir_lstm)

    # Generate all confusion matrix plots
    plot_all_confusion_matrices(confusion_data)

    return results_df


if __name__ == "__main__":
    # Allow running this module directly for debugging.
    evaluate_all_models()
