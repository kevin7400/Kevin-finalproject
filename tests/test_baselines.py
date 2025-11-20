import numpy as np
import pytest

from finance_lstm.models.baselines import evaluate_regression_and_direction


def test_evaluate_regression_and_direction_metrics():
    """
    For small synthetic data, check RMSE, MAE, Accuracy and F1.

    y_true_reg: [0.0,  1.0, -1.0]
    y_pred_reg: [0.1,  0.5, -0.3]

    y_true_cls (true direction labels): [0, 1, 0]
    y_pred_dir (from sign of y_pred_reg): [1, 1, 0]
    """

    y_true_reg = np.array([0.0, 1.0, -1.0], dtype=float)
    y_true_cls = np.array([0, 1, 0], dtype=int)
    y_pred_reg = np.array([0.1, 0.5, -0.3], dtype=float)

    rmse, mae, acc, f1 = evaluate_regression_and_direction(
        y_true_reg=y_true_reg,
        y_true_cls=y_true_cls,
        y_pred_reg=y_pred_reg,
    )

    # ------------------------------------------------------------------
    # Regression metrics (computed by hand)
    # diffs = y_true - y_pred = [-0.1, 0.5, -0.7]
    # squared = [0.01, 0.25, 0.49] -> MSE = (0.01 + 0.25 + 0.49) / 3 = 0.25
    # RMSE = sqrt(0.25) = 0.5
    # MAE  = (0.1 + 0.5 + 0.7) / 3 = 1.3 / 3 ≈ 0.4333
    # ------------------------------------------------------------------
    assert rmse == pytest.approx(0.5, rel=1e-3)
    assert mae == pytest.approx(1.3 / 3.0, rel=1e-3)

    # ------------------------------------------------------------------
    # Classification metrics (direction from sign(y_pred_reg))
    # y_true_cls  = [0, 1, 0]
    # y_pred_dir  = [1, 1, 0]
    # Correct at indices 1 and 2 -> Accuracy = 2 / 3
    #
    # Positive class = 1:
    #   TP = 1 (idx 1)
    #   FP = 1 (idx 0)
    #   FN = 0
    # Precision = TP / (TP + FP) = 1 / 2
    # Recall    = TP / (TP + FN) = 1 / 1 = 1
    # F1        = 2 * P * R / (P + R) = 2 * 0.5 * 1 / (0.5 + 1) = 2/3
    # ------------------------------------------------------------------
    assert acc == pytest.approx(2.0 / 3.0, rel=1e-3)
    assert f1 == pytest.approx(2.0 / 3.0, rel=1e-3)
