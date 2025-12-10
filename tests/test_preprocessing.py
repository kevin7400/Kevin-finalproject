import pathlib

import numpy as np
import pandas as pd
import pytest

import src.data_loader as data_loader


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _make_processed_df(
    start_date: str = "2020-01-01",
    periods: int = 10,
) -> pd.DataFrame:
    """
    Create a small synthetic processed dataset with:
      - 12 feature columns
      - 2 target columns
      - daily DatetimeIndex
    """
    dates = pd.date_range(start=start_date, periods=periods, freq="D")

    # 12 features
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

    data = {}
    for i, col in enumerate(feature_cols):
        # simple pattern so everything is non-constant
        data[col] = np.linspace(i, i + periods - 1, periods, dtype="float64")

    # regression target
    data["next_day_return"] = np.linspace(0.0, 0.9, periods, dtype="float64")
    # classification target 0/1 alternating
    data["next_day_direction"] = np.array(
        [0, 1] * (periods // 2) + ([0] if periods % 2 else []), dtype="int64"
    )

    df = pd.DataFrame(data, index=dates)
    return df


# --------------------------------------------------------------------------------------
# train_test_split_by_date
# --------------------------------------------------------------------------------------


def test_train_test_split_by_date_happy_path():
    """
    train_test_split_by_date should split DataFrame by the given boundaries.
    """
    df = _make_processed_df(start_date="2020-01-01", periods=10)

    train_end = "2020-01-05"
    test_start = "2020-01-06"

    df_train, df_test = data_loader.train_test_split_by_date(
        df, train_end=train_end, test_start=test_start
    )

    assert len(df_train) == 5
    assert len(df_test) == 5
    assert str(df_train.index.min().date()) == "2020-01-01"
    assert str(df_train.index.max().date()) == "2020-01-05"
    assert str(df_test.index.min().date()) == "2020-01-06"
    assert str(df_test.index.max().date()) == "2020-01-10"


def test_train_test_split_by_date_raises_if_empty():
    """
    If either the train or test set is empty, a RuntimeError should be raised.
    """
    df = _make_processed_df(start_date="2020-01-01", periods=5)

    # train_end before all data -> empty train set
    with pytest.raises(RuntimeError):
        data_loader.train_test_split_by_date(
            df, train_end="2019-12-31", test_start="2020-01-01"
        )

    # test_start after all data -> empty test set
    with pytest.raises(RuntimeError):
        data_loader.train_test_split_by_date(
            df, train_end="2020-01-05", test_start="2020-01-06"
        )


# --------------------------------------------------------------------------------------
# scale_features
# --------------------------------------------------------------------------------------


def test_scale_features_uses_train_minmax_only():
    """
    scale_features should fit MinMaxScaler on train features only,
    then apply to test data.
    """
    # Simple 1D example with two features to verify scaling logic
    df_train = pd.DataFrame(
        {
            "f1": [0.0, 10.0],
            "f2": [5.0, 15.0],
        }
    )
    df_test = pd.DataFrame(
        {
            "f1": [20.0],
            "f2": [25.0],
        }
    )

    feature_cols = ["f1", "f2"]

    X_train_scaled, X_test_scaled, scaler = data_loader.scale_features(
        df_train, df_test, feature_cols
    )

    # Train scaling: f1: [0, 10] -> [0, 1], f2: [5, 15] -> [0, 1]
    np.testing.assert_allclose(
        X_train_scaled,
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        rtol=1e-6,
        atol=1e-6,
    )

    # Test scaling uses same train min/max, then clips to [0, 1]:
    #   f1 = 20 -> (20 - 0) / (10 - 0) = 2 -> clipped to 1
    #   f2 = 25 -> (25 - 5) / (15 - 5) = 2 -> clipped to 1
    np.testing.assert_allclose(
        X_test_scaled,
        np.array([[1.0, 1.0]]),
        rtol=1e-6,
        atol=1e-6,
    )

    # Scaler was fitted on train only
    assert np.allclose(scaler.data_min_, np.array([0.0, 5.0]))
    assert np.allclose(scaler.data_max_, np.array([10.0, 15.0]))


# --------------------------------------------------------------------------------------
# create_sequences
# --------------------------------------------------------------------------------------


def test_create_sequences_sliding_window():
    """
    create_sequences should produce a sliding window of length `lookback`
    and align targets with the last element of each window.
    """
    # X has 6 timesteps, y has one value per timestep
    X = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ]
    )
    y = np.array([10, 11, 12, 13, 14, 15])

    lookback = 3
    X_seq, y_seq = data_loader.create_sequences(X, y, lookback=lookback)

    # number of sequences: len(X) - lookback + 1 = 4
    assert X_seq.shape == (4, lookback, 1)
    assert y_seq.shape == (4,)

    # Windows:
    #  k=0: X[0:3] -> y[2]
    #  k=1: X[1:4] -> y[3]
    #  k=2: X[2:5] -> y[4]
    #  k=3: X[3:6] -> y[5]
    np.testing.assert_array_equal(X_seq[0].flatten(), np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(X_seq[-1].flatten(), np.array([3.0, 4.0, 5.0]))
    np.testing.assert_array_equal(y_seq, np.array([12, 13, 14, 15]))


# --------------------------------------------------------------------------------------
# prepare_lstm_data (end-to-end)
# --------------------------------------------------------------------------------------


def test_prepare_lstm_data_creates_arrays_and_scaler(
    monkeypatch, tmp_path: pathlib.Path
):
    """
    prepare_lstm_data should:
      - load processed CSV (patched to a temp file)
      - split by date (we patch the splitter itself)
      - scale features
      - create sequences with given LOOKBACK
      - save all numpy arrays and the scaler under LSTM_DIR.
    """
    # Create a synthetic processed dataset and save it
    df = _make_processed_df(start_date="2020-01-01", periods=10)
    processed_csv = tmp_path / "processed.csv"
    df.to_csv(processed_csv)

    # Patch get_default_processed_csv_path to point to our temp file
    monkeypatch.setattr(
        data_loader,
        "get_default_processed_csv_path",
        lambda: processed_csv,
    )

    # Patch LSTM_DIR to a temp "lstm" directory
    lstm_dir = tmp_path / "lstm"
    lstm_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(data_loader, "LSTM_DIR", lstm_dir)

    # Use a small lookback for test so shapes are easy to reason about
    monkeypatch.setattr(data_loader, "LOOKBACK", 3)

    # Patch the splitter itself so we fully control train/test sizes.
    # Data has 10 rows -> we make:
    #   train: first 6 rows
    #   test:  last  4 rows
    def fake_split(df_in, train_end=None, test_start=None):
        df_train = df_in.iloc[:6].copy()
        df_test = df_in.iloc[6:].copy()
        return df_train, df_test

    monkeypatch.setattr(data_loader, "train_test_split_by_date", fake_split)

    # Run preprocessing
    data_loader.prepare_lstm_data()

    # Check that all expected files exist
    expected_files = [
        "X_train_seq.npy",
        "y_train_reg_seq.npy",  # scaled targets
        "y_train_reg_raw_seq.npy",  # raw targets for evaluation
        "y_train_cls_seq.npy",
        "X_test_seq.npy",
        "y_test_reg_seq.npy",  # scaled targets
        "y_test_reg_raw_seq.npy",  # raw targets for evaluation
        "y_test_cls_seq.npy",
        "feature_scaler.joblib",
        "target_scaler.joblib",  # StandardScaler for regression targets
    ]
    for name in expected_files:
        assert (lstm_dir / name).exists(), f"Missing expected file: {name}"

    # Load arrays and check shapes.
    # With 6 train rows and lookback=3 -> 6-3+1 = 4 train sequences
    # With 4 test rows and lookback=3  -> 4-3+1 = 2 test sequences
    X_train_seq = np.load(lstm_dir / "X_train_seq.npy")
    y_train_reg_seq = np.load(lstm_dir / "y_train_reg_seq.npy")
    y_train_cls_seq = np.load(lstm_dir / "y_train_cls_seq.npy")

    X_test_seq = np.load(lstm_dir / "X_test_seq.npy")
    y_test_reg_seq = np.load(lstm_dir / "y_test_reg_seq.npy")
    y_test_cls_seq = np.load(lstm_dir / "y_test_cls_seq.npy")

    # 12 feature columns (as in _make_processed_df) and lookback=3
    assert X_train_seq.shape == (4, 3, 12)
    assert X_test_seq.shape == (2, 3, 12)

    assert y_train_reg_seq.shape == (4,)
    assert y_train_cls_seq.shape == (4,)
    assert y_test_reg_seq.shape == (2,)
    assert y_test_cls_seq.shape == (2,)

    # Basic sanity: classification targets are still 0/1
    assert set(np.unique(y_train_cls_seq)).issubset({0, 1})
    assert set(np.unique(y_test_cls_seq)).issubset({0, 1})
