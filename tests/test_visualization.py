"""
Tests for the visualization module.

Tests cover:
- Learning curve plot generation
- Confusion matrix plot generation
- Batch confusion matrix generation
- Error handling for invalid inputs
"""

import numpy as np
import pytest
from pathlib import Path

from finance_lstm.visualization import (
    plot_learning_curves,
    plot_confusion_matrix,
    plot_all_confusion_matrices,
)


class MockHistory:
    """Mock Keras History object for testing."""

    def __init__(self):
        self.history = {
            "loss": [0.5, 0.4, 0.3, 0.25, 0.2],
            "val_loss": [0.6, 0.5, 0.4, 0.35, 0.3],
            "mae": [0.3, 0.25, 0.2, 0.18, 0.15],
            "val_mae": [0.35, 0.3, 0.25, 0.22, 0.2],
        }


def test_plot_learning_curves_creates_file(tmp_path):
    """Test that learning curves plot is created."""
    history = MockHistory()
    save_path = tmp_path / "test_learning_curves.png"

    plot_learning_curves(history, save_path=save_path)

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_learning_curves_with_invalid_history():
    """Test that plot_learning_curves raises error for invalid history."""
    with pytest.raises(ValueError, match="Invalid history object"):
        plot_learning_curves(None)


def test_plot_learning_curves_with_missing_keys(tmp_path):
    """Test that plot_learning_curves raises error for missing keys."""
    class IncompleteHistory:
        def __init__(self):
            self.history = {"loss": [0.5, 0.4]}

    history = IncompleteHistory()
    save_path = tmp_path / "test.png"

    with pytest.raises(ValueError, match="History missing required key"):
        plot_learning_curves(history, save_path=save_path)


def test_plot_confusion_matrix_creates_file(tmp_path):
    """Test that confusion matrix plot is created."""
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 1])
    save_path = tmp_path / "test_cm.png"

    plot_confusion_matrix(y_true, y_pred, "TestModel", save_path=save_path)

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_confusion_matrix_with_mismatched_shapes():
    """Test that plot_confusion_matrix raises error for mismatched shapes."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1])  # Different length

    with pytest.raises(ValueError, match="Shape mismatch"):
        plot_confusion_matrix(y_true, y_pred, "Test")


def test_plot_all_confusion_matrices_creates_multiple_files(tmp_path):
    """Test that multiple confusion matrices are created."""
    # Patch RESULTS_DIR to tmp_path
    from finance_lstm import visualization

    original_results_dir = visualization.RESULTS_DIR
    visualization.RESULTS_DIR = tmp_path

    try:
        models_dict = {
            "Model1": (
                np.array([0, 1, 0, 1, 1, 0]),
                np.array([0, 1, 1, 1, 0, 0]),
            ),
            "Model2": (
                np.array([1, 0, 1, 0, 1, 1]),
                np.array([1, 1, 1, 0, 1, 0]),
            ),
        }

        plot_all_confusion_matrices(models_dict)

        assert (tmp_path / "confusion_matrix_model1.png").exists()
        assert (tmp_path / "confusion_matrix_model2.png").exists()
    finally:
        visualization.RESULTS_DIR = original_results_dir


def test_plot_confusion_matrix_filename_sanitization(tmp_path):
    """Test that model names with spaces are sanitized in filenames."""
    from finance_lstm import visualization

    original_results_dir = visualization.RESULTS_DIR
    visualization.RESULTS_DIR = tmp_path

    try:
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])

        plot_confusion_matrix(y_true, y_pred, "Random Forest")

        # Should create file with underscores instead of spaces
        assert (tmp_path / "confusion_matrix_random_forest.png").exists()
    finally:
        visualization.RESULTS_DIR = original_results_dir
