import numpy as np
import pandas as pd
import pytest

import src.evaluation as evaluation


# --------------------------------------------------------------------------------------
# load_processed_dataset
# --------------------------------------------------------------------------------------


def test_load_processed_dataset_success(tmp_path):
    """
    load_processed_dataset should:
    - read the CSV
    - sort by index
    - return a non-empty DataFrame
    """
    # Create a small unsorted CSV
    dates = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"])
    df = pd.DataFrame({"value": [3, 1, 2]}, index=dates)
    csv_path = tmp_path / "processed.csv"
    df.to_csv(csv_path)

    loaded = evaluation.load_processed_dataset(csv_path=csv_path)

    assert not loaded.empty
    # DataFrame should be sorted by index
    assert list(loaded.index) == sorted(dates)


def test_load_processed_dataset_missing_file(tmp_path):
    """
    If the CSV path does not exist,
    load_processed_dataset should raise FileNotFoundError.
    """
    csv_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        evaluation.load_processed_dataset(csv_path=csv_path)


def test_load_processed_dataset_empty_file(tmp_path):
    """
    If the CSV exists but is empty, load_processed_dataset should raise ValueError.
    """
    csv_path = tmp_path / "empty.csv"
    # Write an empty DataFrame
    pd.DataFrame().to_csv(csv_path)

    with pytest.raises(ValueError):
        evaluation.load_processed_dataset(csv_path=csv_path)


# --------------------------------------------------------------------------------------
# train_test_split_by_date
# --------------------------------------------------------------------------------------


def _make_simple_df_for_split():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.DataFrame({"value": range(5)}, index=dates)


def test_train_test_split_by_date_ok():
    df = _make_simple_df_for_split()

    train_end = "2020-01-03"
    test_start = "2020-01-04"

    df_train, df_test = evaluation.train_test_split_by_date(
        df, train_end=train_end, test_start=test_start
    )

    # Train: 1,2,3 Jan -> 3 rows
    assert len(df_train) == 3
    assert df_train.index.min().strftime("%Y-%m-%d") == "2020-01-01"
    assert df_train.index.max().strftime("%Y-%m-%d") == "2020-01-03"

    # Test: 4,5 Jan -> 2 rows
    assert len(df_test) == 2
    assert df_test.index.min().strftime("%Y-%m-%d") == "2020-01-04"
    assert df_test.index.max().strftime("%Y-%m-%d") == "2020-01-05"


def test_train_test_split_by_date_raises_if_empty():
    df = _make_simple_df_for_split()

    # These bounds exclude all rows from both train and test
    train_end = "2019-12-31"
    test_start = "2030-01-01"

    with pytest.raises(RuntimeError):
        evaluation.train_test_split_by_date(
            df, train_end=train_end, test_start=test_start
        )


# --------------------------------------------------------------------------------------
# prepare_features_and_targets
# --------------------------------------------------------------------------------------


def _make_df_with_features_and_targets():
    """
    Build a tiny DataFrame with the exact columns
    expected by prepare_features_and_targets.
    Two rows is enough to check shapes & basic scaling.
    """
    dates = pd.date_range("2020-01-01", periods=4, freq="D")

    data = {
        "rsi_14": [30.0, 40.0, 50.0, 60.0],
        "macd": [0.1, 0.2, 0.3, 0.4],
        "macd_h": [0.01, 0.02, 0.03, 0.04],
        "bbl": [100.0, 101.0, 102.0, 103.0],
        "bbp": [0.2, 0.4, 0.6, 0.8],
        "sma_50": [2000.0, 2010.0, 2020.0, 2030.0],
        "ema_20": [1990.0, 2000.0, 2010.0, 2020.0],
        "obv": [1000.0, 2000.0, 3000.0, 4000.0],
        "close_norm": [0.1, 0.2, 0.3, 0.4],
        "volume_norm": [0.5, 0.6, 0.7, 0.8],
        "lagged_log_return": [0.0, 0.01, 0.02, 0.03],
        "atr_14": [1.0, 1.1, 1.2, 1.3],
        "next_day_return": [0.5, -0.2, 0.1, -0.3],
        "next_day_direction": [1, 0, 1, 0],
    }

    return pd.DataFrame(data, index=dates)


def test_prepare_features_and_targets_shapes():
    df = _make_df_with_features_and_targets()
    # Simple split: first 2 rows train, last 2 rows test
    df_train = df.iloc[:2].copy()
    df_test = df.iloc[2:].copy()

    (
        X_train_scaled,
        X_test_scaled,
        y_train_reg,
        y_test_reg,
        y_train_cls,
        y_test_cls,
    ) = evaluation.prepare_features_and_targets(df_train, df_test)

    assert X_train_scaled.shape == (2, 12)
    assert X_test_scaled.shape == (2, 12)
    assert y_train_reg.shape == (2,)
    assert y_test_reg.shape == (2,)
    assert y_train_cls.shape == (2,)
    assert y_test_cls.shape == (2,)

    # Check that scaling produced values in [0, 1] for train
    assert np.all(X_train_scaled >= 0.0)
    assert np.all(X_train_scaled <= 1.0)


# --------------------------------------------------------------------------------------
# evaluate_regression_and_direction
# --------------------------------------------------------------------------------------


def test_evaluate_regression_and_direction_values():
    """
    Small numeric check for RMSE/MAE/Accuracy/F1.

    y_true_reg: [0.0,  1.0, -1.0]
    y_pred_reg: [0.1,  0.5, -0.3]
    y_true_cls: [0, 1, 0]
    """
    y_true_reg = np.array([0.0, 1.0, -1.0], dtype=float)
    y_pred_reg = np.array([0.1, 0.5, -0.3], dtype=float)
    y_true_cls = np.array([0, 1, 0], dtype=int)

    rmse, mae, acc, f1, precision, recall = evaluation.evaluate_regression_and_direction(
        y_true_reg=y_true_reg,
        y_true_cls=y_true_cls,
        y_pred_reg=y_pred_reg,
    )

    # diffs = [-0.1, 0.5, -0.7]
    # squared = [0.01, 0.25, 0.49] -> MSE = 0.25 -> RMSE = 0.5
    # MAE = (0.1 + 0.5 + 0.7) / 3 = 1.3 / 3
    assert rmse == pytest.approx(0.5, rel=1e-3)
    assert mae == pytest.approx(1.3 / 3.0, rel=1e-3)

    # y_pred_dir = [1, 1, 0]; y_true_cls = [0, 1, 0]
    # correct = 2/3
    # F1 for class 1: precision=1/2, recall=1 -> F1 = 2/3
    assert acc == pytest.approx(2.0 / 3.0, rel=1e-3)
    assert f1 == pytest.approx(2.0 / 3.0, rel=1e-3)
    assert precision == pytest.approx(0.5, rel=1e-3)
    assert recall == pytest.approx(1.0, rel=1e-3)


# --------------------------------------------------------------------------------------
# load_lstm_predictions
# --------------------------------------------------------------------------------------


def test_load_lstm_predictions_success(tmp_path):
    """
    load_lstm_predictions should load all five .npy files,
    check shapes, and return them.
    """
    dir_path = tmp_path

    # Scaled targets (for reference)
    y_test_reg_seq = np.array([0.1, -0.2, 0.3], dtype=float)
    # Raw targets (for RMSE/MAE metrics)
    y_test_reg_raw_seq = np.array([0.5, -1.0, 1.5], dtype=float)
    y_test_cls_seq = np.array([1, 0, 1], dtype=int)
    # Predictions (already inverse transformed to raw scale)
    y_pred_reg_lstm = np.array([0.4, -0.8, 1.2], dtype=float)
    y_pred_dir_lstm = np.array([1, 0, 1], dtype=int)

    np.save(dir_path / "y_test_reg_seq.npy", y_test_reg_seq)
    np.save(dir_path / "y_test_reg_raw_seq.npy", y_test_reg_raw_seq)
    np.save(dir_path / "y_test_cls_seq.npy", y_test_cls_seq)
    np.save(dir_path / "y_pred_reg_lstm.npy", y_pred_reg_lstm)
    np.save(dir_path / "y_pred_dir_lstm.npy", y_pred_dir_lstm)

    (
        y_reg_raw_loaded,
        y_cls_loaded,
        y_pred_reg_loaded,
        y_pred_dir_loaded,
        y_reg_scaled_loaded,
    ) = evaluation.load_lstm_predictions(lstm_dir=dir_path)

    assert np.array_equal(y_reg_raw_loaded, y_test_reg_raw_seq)
    assert np.array_equal(y_cls_loaded, y_test_cls_seq)
    assert np.array_equal(y_pred_reg_loaded, y_pred_reg_lstm)
    assert np.array_equal(y_pred_dir_loaded, y_pred_dir_lstm)
    assert np.array_equal(y_reg_scaled_loaded, y_test_reg_seq)


def test_load_lstm_predictions_missing_files(tmp_path):
    """
    If one or more LSTM output files are missing,
    load_lstm_predictions should raise FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        evaluation.load_lstm_predictions(lstm_dir=tmp_path)
