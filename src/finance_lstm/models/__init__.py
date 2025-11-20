"""
Models subpackage for the S&P 500 LSTM forecasting project.

This subpackage contains:

- lstm:   Stacked LSTM model used to predict next-day percentage returns.
- baselines: Simple regression baselines (Linear, Random Forest, XGBoost).

Typical usage:

    from finance_lstm.models import (
        build_lstm_model,
        train_lstm_model,
        train_and_evaluate_lstm,
        train_baseline_models,
    )

For most users, you don't need to import this directly: the main pipeline
already calls these pieces in the right order.
"""

from __future__ import annotations

from .lstm import build_lstm_model, train_lstm_model, train_and_evaluate_lstm
from .baselines import train_baseline_models

__all__ = [
    "build_lstm_model",
    "train_lstm_model",
    "train_and_evaluate_lstm",
    "train_baseline_models",
]
