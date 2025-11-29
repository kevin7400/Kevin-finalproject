import numpy as np
import pandas as pd
import pytest

import src.data_loader as data_loader


# --------------------------------------------------------------------------------------
# get_default_raw_csv_path
# --------------------------------------------------------------------------------------


def test_get_default_raw_csv_path_uses_config_years():
    """
    The default raw CSV path should incorporate the START_DATE and END_DATE years
    from data_loader.py in the filename.
    """
    path = data_loader.get_default_raw_csv_path()

    start_year = data_loader.START_DATE[:4]
    end_year = data_loader.END_DATE[:4]
    expected_name = f"sp500_{start_year}_{end_year}.csv"

    assert path.name == expected_name
    # Should live under the project's data/raw directory
    assert "data" in str(path.parent)
    assert "raw" in str(path.parent)


# --------------------------------------------------------------------------------------
# download_sp500_data
# --------------------------------------------------------------------------------------


def test_download_sp500_data_success(monkeypatch):
    """
    When yfinance.download returns a normal OHLCV DataFrame, download_sp500_data
    should:
      - not raise
      - sort by index
      - keep OHLCV columns (and Adj Close if present)
    """

    # Build a small fake OHLCV DataFrame
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    df_fake = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "High": [1.5, 2.5, 3.5],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.1, 2.1, 3.1],
            "Adj Close": [1.0, 2.0, 3.0],
            "Volume": [100, 200, 300],
        },
        index=dates[::-1],  # reverse order to check sorting
    )

    def fake_yf_download(*args, **kwargs):
        return df_fake

    monkeypatch.setattr(data_loader.yf, "download", fake_yf_download)

    out = data_loader.download_sp500_data(
        ticker="FAKE",
        start="2020-01-01",
        end="2020-01-03",
    )

    # Should be sorted ascending by date
    assert list(out.index) == sorted(dates)

    # Should contain at least OHLCV, and keep Adj Close if present
    expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert list(out.columns) == expected_cols

    # Values should match the original (after sorting)
    df_expected = df_fake.sort_index()[expected_cols]
    pd.testing.assert_frame_equal(out, df_expected)


def test_download_sp500_data_empty_raises(monkeypatch):
    """
    If yfinance.download returns an empty DataFrame, download_sp500_data
    should raise a RuntimeError.
    """

    def fake_yf_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(data_loader.yf, "download", fake_yf_download)

    with pytest.raises(RuntimeError, match="Downloaded DataFrame is empty"):
        data_loader.download_sp500_data(
            ticker="FAKE",
            start="2020-01-01",
            end="2020-01-03",
        )


def test_download_sp500_data_missing_ohlcv_raises(monkeypatch):
    """
    If the DataFrame returned by yfinance is missing any of the required
    OHLCV columns, download_sp500_data should raise a RuntimeError.
    """

    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    # Missing 'High' and 'Low' on purpose
    df_fake = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "Close": [1.1, 2.1, 3.1],
            "Volume": [100, 200, 300],
        },
        index=dates,
    )

    def fake_yf_download(*args, **kwargs):
        return df_fake

    monkeypatch.setattr(data_loader.yf, "download", fake_yf_download)

    with pytest.raises(RuntimeError, match="Missing required OHLCV columns"):
        data_loader.download_sp500_data(
            ticker="FAKE",
            start="2020-01-01",
            end="2020-01-03",
        )


# --------------------------------------------------------------------------------------
# download_and_save_raw_data
# --------------------------------------------------------------------------------------


def test_download_and_save_raw_data_reuses_existing(tmp_path, monkeypatch):
    """
    If the CSV already exists and force=False, download_and_save_raw_data
    should NOT call download_sp500_data and should return the existing path.
    """

    # We force the function to use a temp raw path
    fake_csv_path = tmp_path / "sp500_fake.csv"

    # Create an existing CSV
    df_existing = pd.DataFrame({"x": [1, 2, 3]})
    df_existing.to_csv(fake_csv_path, index=False)

    # Make get_default_raw_csv_path point to our temporary file
    monkeypatch.setattr(
        data_loader, "get_default_raw_csv_path", lambda: fake_csv_path
    )

    # If this is called, we want to fail the test
    def fake_download_sp500_data(*args, **kwargs):
        raise AssertionError("download_sp500_data should not be called")

    monkeypatch.setattr(data_loader, "download_sp500_data", fake_download_sp500_data)

    path = data_loader.download_and_save_raw_data(force=False)

    assert path == fake_csv_path
    assert path.exists()


def test_download_and_save_raw_data_force_download(tmp_path, monkeypatch):
    """
    If force=True or the file does not exist, download_and_save_raw_data
    should call download_sp500_data and write the returned DataFrame to CSV.
    """

    fake_csv_path = tmp_path / "sp500_force.csv"

    # Make get_default_raw_csv_path point to our temporary file
    monkeypatch.setattr(
        data_loader, "get_default_raw_csv_path", lambda: fake_csv_path
    )

    # Small fake OHLCV DataFrame
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    df_fake = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.5, 2.5],
            "Low": [0.5, 1.5],
            "Close": [1.1, 2.1],
            "Volume": [100, 200],
        },
        index=dates,
    )

    def fake_download_sp500_data(ticker, start, end, interval="1d"):
        # We don't care about the arguments too much here, just return the DF
        return df_fake

    monkeypatch.setattr(data_loader, "download_sp500_data", fake_download_sp500_data)

    path = data_loader.download_and_save_raw_data(force=True)

    assert path == fake_csv_path
    assert path.exists()

    # Reload and check that contents match (up to column order & index)
    df_loaded = pd.read_csv(path, index_col=0, parse_dates=True)
    pd.testing.assert_index_equal(df_loaded.index, df_fake.index)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert col in df_loaded.columns
        np.testing.assert_allclose(df_loaded[col].values, df_fake[col].values)
