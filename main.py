#!/usr/bin/env python

"""
Main entry point for the S&P 500 LSTM stock prediction project.

This script orchestrates the full pipeline:
    1. Download raw OHLCV data via yfinance
    2. Build 12 technical indicators + 2 targets
    3. Preprocess data into LSTM-ready sequences
    3.5. (Optional) Hyperparameter tuning
    4. Train and evaluate LSTM model
    5. Train baseline models and compare results

Usage (from project root):

    python main.py                    # Standard run with LSTM regressor
    python main.py --tune             # Run hyperparameter tuning (~35 min)
    python main.py --no-tuned         # Run with default parameters instead of tuned
    python main.py --classifier       # Use LSTM classifier instead of regressor

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
from src.models import train_and_evaluate_lstm, train_and_evaluate_lstm_classifier
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
    run_tuning: bool = False,
    use_tuned_params: bool = False,
    use_classifier: bool = False,
) -> pd.DataFrame:
    """Run the full end-to-end pipeline.

    Args:
        force_download:
            If True, re-download raw data even if the CSV already exists.
            If False, reuse the existing raw CSV when available.
        rebuild_features:
            If True, recompute indicators + targets and overwrite the processed CSV.
            If False, reuse the existing processed CSV (if present).
        run_tuning:
            If True, run hyperparameter tuning before final training.
            This takes ~35 minutes but finds optimal hyperparameters.
        use_tuned_params:
            If True, load previously saved tuned params from data/tuning/best_params.json.
            Ignored if run_tuning is True.
        use_classifier:
            If True, train LSTM as a classifier (binary_crossentropy loss,
            sigmoid output) instead of regressor. This directly predicts
            direction rather than returns.

    Returns:
        A pandas DataFrame with one row per model (LSTM + baselines)
        and the metrics columns: rmse, mae, accuracy, f1, precision, recall.
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

    # Hyperparameter tuning (optional)
    tuned_params = None
    lstm_config = None
    lstm_key = 'LSTMClassifier' if use_classifier else 'LSTM'

    if run_tuning:
        print("\n=== STEP 3.5: Hyperparameter Tuning ===")
        from src.hyperparameter_tuning import tune_all_models
        tuned_params = tune_all_models(use_classifier=use_classifier)
        if lstm_key in tuned_params and tuned_params[lstm_key].get('best_params'):
            lstm_config = tuned_params[lstm_key]['best_params']
    elif use_tuned_params:
        print("\n=== STEP 3.5: Loading Tuned Parameters ===")
        from src.hyperparameter_tuning import load_best_params
        try:
            tuned_params = load_best_params()
            print(f"Loaded tuned params: {list(tuned_params.keys())}")
            if lstm_key in tuned_params and tuned_params[lstm_key].get('best_params'):
                lstm_config = tuned_params[lstm_key]['best_params']
            elif 'LSTM' in tuned_params and tuned_params['LSTM'].get('best_params'):
                # Fallback to LSTM if LSTMClassifier not found
                lstm_config = tuned_params['LSTM']['best_params']
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Running with default parameters.")

    print("\n=== STEP 4: Train and evaluate LSTM ===")
    if use_classifier:
        print("Using LSTM CLASSIFIER mode (binary_crossentropy)")
        metrics, history = train_and_evaluate_lstm_classifier(config=lstm_config)
        print(f"\nLSTM Classifier metrics: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    else:
        print("Using LSTM REGRESSOR mode (MSE/MAE)")
        test_mse, test_mae, history = train_and_evaluate_lstm(config=lstm_config)
        print(f"\nLSTM Regressor metrics: MSE={test_mse:.4f}, MAE={test_mae:.4f}")

    print("\n=== STEP 5: Evaluate baselines + LSTM ===")
    results_df = evaluate_all_models(tuned_params=tuned_params)

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
    if tuned_params:
        print(f"  Tuned params:  data/tuning/best_params.json")

    return results_df


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="S&P 500 LSTM Stock Prediction Pipeline"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning (~35 min)",
    )
    parser.add_argument(
        "--use-tuned",
        action="store_true",
        default=True,
        help="Use previously saved tuned parameters (default: True)",
    )
    parser.add_argument(
        "--no-tuned",
        action="store_true",
        help="Disable using tuned parameters (use defaults instead)",
    )
    parser.add_argument(
        "--classifier",
        action="store_true",
        help="Use LSTM classifier (binary_crossentropy) instead of regressor",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of raw data",
    )
    args = parser.parse_args()

    ensure_venv()
    # --no-tuned overrides --use-tuned
    use_tuned = args.use_tuned and not args.no_tuned
    run_pipeline(
        force_download=args.force_download,
        run_tuning=args.tune,
        use_tuned_params=use_tuned,
        use_classifier=args.classifier,
    )


if __name__ == "__main__":
    main()
