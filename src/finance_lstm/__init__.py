"""
Top-level package for the S&P 500 LSTM forecasting project.

This package provides:

- config: Central configuration (dates, ticker, LSTM hyperparameters, etc.).
- data_download: Utilities to download raw OHLCV data via yfinance.
- features: Feature engineering for the 12 technical indicators + 2 targets.
- preprocessing: Preprocessing routines to build LSTM-ready 3D sequences.
- models: LSTM and baseline regression models.
- evaluation: Model comparison utilities (LSTM vs. baselines).
- pipeline: End-to-end orchestration of the full workflow.

Typical entry points:

    from finance_lstm import config
    from finance_lstm.pipeline import run_pipeline

"""

from __future__ import annotations

from . import config

__all__ = ["config"]
