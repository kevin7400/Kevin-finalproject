#!/usr/bin/env python

"""
Main entry point for the S&P 500 LSTM stock prediction project.

This script orchestrates the full pipeline:
    1. Download raw OHLCV data via yfinance
    2. Build 12 technical indicators + 2 targets
    3. Preprocess data into LSTM-ready sequences
    4. Train and evaluate LSTM model
    5. Train baseline models and compare results

Usage (from project root):

    python main.py

Or with conda:

    conda activate kevin-lstm
    python main.py
"""

from __future__ import annotations

import os
import pathlib

import pandas as pd

from src.data_loader import (
    download_and_save_raw_data,
    build_and_save_feature_target_dataset,
    get_default_processed_csv_path,
    prepare_lstm_data,
    START_DATE,
    END_DATE,
    TRAIN_END,
    TEST_START,
)
from src.models import train_and_evaluate_lstm
from src.evaluation import evaluate_all_models

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LSTM_DIR = DATA_DIR / "lstm"
RESULTS_DIR = PROJECT_ROOT / "results"  # Changed from data/results/


# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------


def ensure_venv() -> None:
    """Warn if the user is not inside a virtual environment."""
    if "VIRTUAL_ENV" not in os.environ and "CONDA_PREFIX" not in os.environ:
        print("---- ENVIRONMENT WARNING ----")
        print("You are not running inside a virtual environment.")
        print("Recommended steps (from project root):")
        print("\nUsing pip:")
        print("  python3 -m venv .venv")
        print("  source .venv/bin/activate")
        print("  pip install -r requirements.txt")
        print("\nOr using conda:")
        print("  conda env create -f environment.yml")
        print("  conda activate kevin-lstm")
        print()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    force_download: bool = False,
    rebuild_features: bool = True,
) -> pd.DataFrame:
    """Run the full end-to-end pipeline.

    Args:
        force_download:
            If True, re-download raw data even if the CSV already exists.
            If False, reuse the existing raw CSV when available.
        rebuild_features:
            If True, recompute indicators + targets and overwrite the processed CSV.
            If False, reuse the existing processed CSV (if present).

    Returns:
        A pandas DataFrame with one row per model (LSTM + baselines)
        and the metrics columns: rmse, mae, accuracy, f1.
    """
    print("\n=== STEP 1: Download raw data ===")
    raw_path = download_and_save_raw_data(force=force_download)

    print("\n=== STEP 2: Build features & targets ===")
    if rebuild_features:
        processed_path = build_and_save_feature_target_dataset()
    else:
        processed_path = get_default_processed_csv_path()
        print(f"Reusing existing processed CSV at {processed_path}")

    print("\n=== STEP 3: Preprocess data for LSTM ===")
    prepare_lstm_data()

    print("\n=== STEP 4: Train and evaluate LSTM ===")
    test_mse, test_mae, history = train_and_evaluate_lstm()
    print(f"\nLSTM test metrics: MSE={test_mse:.4f}, MAE={test_mae:.4f}")

    print("\n=== STEP 5: Evaluate baselines + LSTM ===")
    results_df = evaluate_all_models()

    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print(f"  Raw CSV:       {raw_path}")
    print(f"  Processed CSV: {processed_path}")
    print(f"  LSTM dir:      {LSTM_DIR}")
    print(f"  Results dir:   {RESULTS_DIR}")
    print(f"  Visualizations:")
    print(f"    - Learning curves:     {RESULTS_DIR / 'learning_curves.png'}")
    print(f"    - Confusion matrices:  {RESULTS_DIR / 'confusion_matrix_*.png'}")
    print(
        f"  Train period:  {START_DATE} -> {TRAIN_END} "
        f"| Test period: {TEST_START} -> {END_DATE}"
    )

    return results_df


def main() -> None:
    """Main entry point."""
    ensure_venv()
    run_pipeline()


if __name__ == "__main__":
    main()
