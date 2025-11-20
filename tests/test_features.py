import pathlib

import numpy as np
import pandas as pd
import pytest

from finance_lstm import config
from finance_lstm import features


# --------------------------------------------------------------------------------------
# get_default_processed_csv_path
# --------------------------------------------------------------------------------------


def test_get_default_processed_csv_path_uses_years_from_config():
    """
    The processed CSV path should contain START_DATE and END_DATE years
    in the filename, e.g. sp500_2018_2024_features_targets.csv.
    """
    path = features.get_default_processed_csv_path()

    start_year = config.START_DATE[:4]
    end_year = config.END_DATE[:4]
    expected_name = f"sp500_{start_year}_{end_year}_features_targets.csv"

    assert path.name == expected_name
    # Should live under data/processed/
    assert "data" in str(path.parent)
    assert "processed" in str(path.parent)


# --------------------------------------------------------------------------------------
# load_raw_data
# --------------------------------------------------------------------------------------


def test_load_raw_data_ok(tmp_path: pathlib.Path):
    """
    load_raw_data should:
      - read a CSV with OHLCV columns
      - lowercase the column names
      - keep a DatetimeIndex
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    df_raw = pd.DataFrame(
        {
            "Open": np.linspace(100, 104, 5),
            "High": np.linspace(101, 105, 5),
            "Low": np.linspace(99, 103, 5),
            "Close": np.linspace(100.5, 104.5, 5),
            "Volume": [1000, 1200, 1100, 1300, 1250],
        },
        index=dates,
    )

    csv_path = tmp_path / "raw.csv"
    df_raw.to_csv(csv_path, index=True)

    df_loaded = features.load_raw_data(csv_path)

    # Index should be datetime and match
    assert isinstance(df_loaded.index, pd.DatetimeIndex)
    assert list(df_loaded.index) == list(dates)

    # Columns should be lowercase and contain required OHLCV
    expected_cols = {"open", "high", "low", "close", "volume"}
    assert expected_cols.issubset(set(df_loaded.columns))


def test_load_raw_data_missing_columns_raises(tmp_path: pathlib.Path):
    """
    If any of the required OHLCV columns are missing, load_raw_data must raise.
    """
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    # Intentionally missing 'High'
    df_raw = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.1, 2.1, 3.1],
            "Volume": [100, 200, 300],
        },
        index=dates,
    )

    csv_path = tmp_path / "raw_missing.csv"
    df_raw.to_csv(csv_path, index=True)

    with pytest.raises(RuntimeError, match="Missing required columns"):
        features.load_raw_data(csv_path)


# --------------------------------------------------------------------------------------
# add_targets
# --------------------------------------------------------------------------------------


def test_add_targets_computes_returns_and_directions():
    """
    add_targets should compute:
      next_day_return_t = ((C_{t+1} - C_t) / C_t) * 100
      next_day_direction = 1 if next_day_return > threshold else 0
    """
    # Simple close series: up, down, up
    closes = [100.0, 105.0, 100.0, 110.0]
    dates = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    df = pd.DataFrame({"close": closes}, index=dates)

    df_with_targets = features.add_targets(df.copy(), threshold=0.0)

    # Expected next-day returns for first 3 rows; last is NaN
    expected_returns = []
    for i in range(len(closes) - 1):
        r = (closes[i + 1] - closes[i]) / closes[i] * 100.0
        expected_returns.append(r)
    expected_returns.append(np.nan)

    # Check numeric values (up to small tolerance)
    np.testing.assert_allclose(
        df_with_targets["next_day_return"].iloc[:-1].values,
        np.array(expected_returns[:-1]),
        rtol=1e-7,
        atol=1e-7,
    )
    assert np.isnan(df_with_targets["next_day_return"].iloc[-1])

    # Directions: > 0 -> 1, else 0
    expected_dirs = [1 if r > 0.0 else 0 for r in expected_returns[:-1]] + [0]
    np.testing.assert_array_equal(
        df_with_targets["next_day_direction"].values,
        np.array(expected_dirs),
    )


# --------------------------------------------------------------------------------------
# add_technical_indicators
# --------------------------------------------------------------------------------------


def _make_simple_ohlcv(n: int = 80) -> pd.DataFrame:
    """
    Helper: create a simple OHLCV DataFrame long enough to produce
    non-NaN values for 14/20/50-day indicators.
    """
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = np.linspace(100, 120, n)
    df = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000, 2_000, n),
        },
        index=dates,
    )
    return df


def test_add_technical_indicators_adds_all_expected_columns():
    """
    add_technical_indicators should add all 12 feature columns,
    with at least some non-NaN values.
    """
    df = _make_simple_ohlcv(n=80)

    df_ind = features.add_technical_indicators(df.copy())

    expected_cols = [
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

    for col in expected_cols:
        assert col in df_ind.columns, f"Missing indicator column: {col}"
        # At least one non-null value (warm-up rows can be NaN)
        assert df_ind[col].notna().any(), f"Column {col} is entirely NaN"


# --------------------------------------------------------------------------------------
# build_feature_target_dataset & build_and_save_feature_target_dataset
# --------------------------------------------------------------------------------------


def test_build_feature_target_dataset_uses_raw_csv(monkeypatch, tmp_path):
    """
    build_feature_target_dataset should:
      - load the RAW_CSV path
      - add all indicators + targets
      - drop NaNs, leaving a non-empty dataset
      - return exactly 12 feature columns + 2 target columns.
    """
    # Prepare a fake raw CSV
    df_raw = _make_simple_ohlcv(n=90)
    raw_path = tmp_path / "sp500_fake_raw.csv"
    df_raw.to_csv(raw_path, index=True)

    # Make the module use our temp raw path
    monkeypatch.setattr(features, "RAW_CSV", raw_path)

    df_final = features.build_feature_target_dataset()

    # Should not be empty after dropping NaNs
    assert len(df_final) > 0

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
    target_cols = ["next_day_return", "next_day_direction"]

    assert list(df_final.columns) == feature_cols + target_cols
    # No NaNs remaining
    assert not df_final.isna().any().any()
    # Index should be sorted and datetime
    assert isinstance(df_final.index, pd.DatetimeIndex)
    assert list(df_final.index) == sorted(df_final.index)


def test_build_and_save_feature_target_dataset_writes_to_processed_csv(
    monkeypatch,
    tmp_path: pathlib.Path,
):
    """
    build_and_save_feature_target_dataset should:
      - build the dataset
      - save it to PROCESSED_CSV
      - return the processed path
    All without touching the real data/ directory (we patch paths).
    """
    # 1) Prepare a fake RAW_CSV
    df_raw = _make_simple_ohlcv(n=90)
    raw_path = tmp_path / "sp500_fake_raw.csv"
    df_raw.to_csv(raw_path, index=True)

    monkeypatch.setattr(features, "RAW_CSV", raw_path)

    # 2) Point PROCESSED_CSV to a temp file
    processed_path = tmp_path / "sp500_fake_features_targets.csv"
    monkeypatch.setattr(features, "PROCESSED_CSV", processed_path)

    # 3) Run
    out_path = features.build_and_save_feature_target_dataset()

    assert out_path == processed_path
    assert processed_path.exists()

    # 4) Reload and sanity-check the saved dataset
    df_loaded = pd.read_csv(processed_path, index_col=0, parse_dates=True)

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
    target_cols = ["next_day_return", "next_day_direction"]

    assert list(df_loaded.columns) == feature_cols + target_cols
    assert len(df_loaded) > 0
    assert not df_loaded.isna().any().any()
