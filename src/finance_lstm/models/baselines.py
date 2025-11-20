"""
Baseline models for comparison with LSTM.

This module does NOT handle any file I/O or date splits.
It only contains generic utilities to:

    - train baseline regressors on (X_train, y_train_reg)
    - evaluate regression + direction metrics from predictions
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from xgboost import XGBRegressor


def train_baseline_models(X_train: np.ndarray, y_train_reg: np.ndarray):
    """
    Train 3 regression baselines:
      - LinearRegression
      - RandomForestRegressor
      - XGBRegressor

    All are trained to predict next_day_return (regression).
    Direction metrics will be based on the sign of the predicted return.
    """
    models = {}

    # Linear baseline
    lin = LinearRegression()
    lin.fit(X_train, y_train_reg)
    models["LinearRegression"] = lin

    # Random Forest baseline
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train_reg)
    models["RandomForest"] = rf

    # XGBoost baseline
    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train_reg)
    models["XGBoost"] = xgb

    return models


def evaluate_regression_and_direction(
    y_true_reg: np.ndarray,
    y_true_cls: np.ndarray,
    y_pred_reg: np.ndarray,
):
    """
    Compute RMSE, MAE for regression & Accuracy, F1 for direction, using:
      - y_pred_reg (continuous %) for magnitude
      - sign(y_pred_reg) as classification (Up if > 0, else Down)
    """
    rmse = float(np.sqrt(mean_squared_error(y_true_reg, y_pred_reg)))
    mae = float(mean_absolute_error(y_true_reg, y_pred_reg))

    y_pred_dir = (y_pred_reg > 0.0).astype(int)

    acc = float(accuracy_score(y_true_cls, y_pred_dir))
    f1 = float(f1_score(y_true_cls, y_pred_dir))

    return rmse, mae, acc, f1
