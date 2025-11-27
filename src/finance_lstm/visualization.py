"""
Visualization utilities for model training and evaluation.

This module provides functions to generate and save visualizations for:
- Learning curves (loss/MAE over epochs) for LSTM training
- Confusion matrices for classification performance

All plots are saved to the data/results/ directory.
"""

from __future__ import annotations

import pathlib
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Project root detection
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
        Where to save the plot. Defaults to data/results/learning_curves.png.

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
    Learning curves saved to: .../data/results/learning_curves.png
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
        Where to save the plot. Defaults to data/results/confusion_matrix_{model_name}.png.

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
