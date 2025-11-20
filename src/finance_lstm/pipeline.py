"""
End-to-end pipeline orchestration for the S&P 500 LSTM project.

This module coordinates the main steps:

    1. Download raw OHLCV data via yfinance.
    2. Build the 12 indicators + 2 targets and save a processed CSV.
    3. Preprocess the data into LSTM-ready 3D sequences.
    4. Train and evaluate the LSTM model.
    5. Train baseline models and compare all models (LSTM + baselines).

Typical usage (from the project root):

    from finance_lstm.pipeline import run_pipeline
    run_pipeline()

The root-level run_pipeline.py script will just import and call this function.
"""

from __future__ import annotations

import pathlib

import pandas as pd

from . import config
from .download_data import download_and_save_raw_data
from .features import (
    build_and_save_feature_target_dataset,
    get_default_processed_csv_path,
)
from .preprocessing import prepare_lstm_data
from .models.lstm import train_and_evaluate_lstm
from .evaluation import evaluate_all_models

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# src/finance_lstm/pipeline.py -> src -> project root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LSTM_DIR = DATA_DIR / "lstm"
RESULTS_DIR = DATA_DIR / "results"


# ---------------------------------------------------------------------------
# Orchestrator
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
    test_mse, test_mae = train_and_evaluate_lstm()
    print(f"\nLSTM test metrics: MSE={test_mse:.4f}, MAE={test_mae:.4f}")

    print("\n=== STEP 5: Evaluate baselines + LSTM ===")
    results_df = evaluate_all_models()

    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print(f"  Raw CSV:       {raw_path}")
    print(f"  Processed CSV: {processed_path}")
    print(f"  LSTM dir:      {LSTM_DIR}")
    print(f"  Results dir:   {RESULTS_DIR}")
    print(
        f"  Train period:  {config.START_DATE} -> {config.TRAIN_END} "
        f"| Test period: {config.TEST_START} -> {config.END_DATE}"
    )

    return results_df


def main() -> None:
    """Allow running this module directly:

    python -m finance_lstm.pipeline
    """
    run_pipeline()


if __name__ == "__main__":
    main()
