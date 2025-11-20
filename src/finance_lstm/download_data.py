"""
Raw data download utilities using yfinance.

This module:
- Defines a canonical location for the raw S&P 500 CSV under `data/raw/`.
- Downloads OHLCV data using configuration from `config.py`.
- Exposes `download_and_save_raw_data()` used by the pipeline.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import yfinance as yf

from . import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# src/finance_lstm/download_data.py -> src -> project root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def get_default_raw_csv_path() -> pathlib.Path:
    """
    Default path for the raw S&P 500 CSV, based on config dates.

    Example: data/raw/sp500_2018_2024.csv
    """
    start_year = config.START_DATE[:4]
    end_year = config.END_DATE[:4]
    return RAW_DIR / f"sp500_{start_year}_{end_year}.csv"


# ---------------------------------------------------------------------------
# Core download logic
# ---------------------------------------------------------------------------


def download_sp500_data(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download historical OHLCV data via yfinance.

    Returns:
        DataFrame with DatetimeIndex and at least:
            ['Open', 'High', 'Low', 'Close', 'Volume']
    """
    print(f"Calling yfinance.download(ticker={ticker}, start={start}, end={end})...")
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=True,
    )

    if df.empty:
        raise RuntimeError(
            "Downloaded DataFrame is empty. " "Check ticker or date range in config.py."
        )

    # Handle MultiIndex columns like ('Close', '^GSPC')
    if isinstance(df.columns, pd.MultiIndex):
        if "Price" in df.columns.names:
            df.columns = df.columns.get_level_values("Price")
        else:
            df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing required OHLCV columns: {missing}. "
            f"Got columns: {list(df.columns)}"
        )

    cols_to_keep = [
        c
        for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        if c in df.columns
    ]
    df = df[cols_to_keep]

    print(f"Downloaded {len(df)} rows with columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Public helper for pipeline
# ---------------------------------------------------------------------------


def download_and_save_raw_data(force: bool = False) -> pathlib.Path:
    """
    Download raw data (if needed) and return the CSV path.

    Args:
        force:
            If True, always re-download and overwrite the CSV.
            If False, reuse existing CSV when available.

    Returns:
        Path to the raw CSV file.
    """
    csv_path = get_default_raw_csv_path()

    if csv_path.exists() and not force:
        print(f"Raw CSV already exists at {csv_path}, reusing it.")
        return csv_path

    print(
        f"Downloading raw data for {config.TICKER} "
        f"from {config.START_DATE} to {config.END_DATE}..."
    )
    df = download_sp500_data(
        ticker=config.TICKER,
        start=config.START_DATE,
        end=config.END_DATE,
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=True)
    print(f"Saved raw data to: {csv_path.resolve()}")

    return csv_path


def main() -> None:
    """
    Manual entry point:

        python -m finance_lstm.download_data

    This will force a fresh download/overwrite of the raw CSV.
    """
    download_and_save_raw_data(force=True)


if __name__ == "__main__":
    main()
