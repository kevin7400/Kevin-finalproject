# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LSTM-based S&P 500 forecasting pipeline that predicts next-day returns and direction using 12 technical indicators. Compares a stacked LSTM against baseline models (Linear Regression, Random Forest, XGBoost).

## Common Commands

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run full pipeline (requires tuning first)
python main.py --tune           # Run tuning (~35 min, required once)
python main.py                  # Run evaluation with tuned params

# Tests
pytest tests/                              # Run all tests
pytest tests/test_features.py -v           # Run single test file
pytest --cov=src --cov-report=term-missing tests/  # With coverage

# Linting
black src tests main.py          # Format code
flake8 src tests main.py         # Check style
```

## Architecture

**4-file structure under `src/`:**

| File | Responsibility |
|------|----------------|
| `data_loader.py` | Configuration constants, data download (yfinance), feature engineering (12 technical indicators), preprocessing, LSTM sequence generation |
| `models.py` | LSTM model architecture and training, model evaluation utilities |
| `evaluation.py` | Baseline model training, metrics computation (RMSE, MAE, Accuracy, F1), visualizations, model comparison |
| `hyperparameter_tuning.py` | Grid/random search for all models, threshold tuning, class imbalance handling |

**Data flow:** Raw data (`data/raw/`) → Features + targets (`data/processed/`) → LSTM sequences (`data/lstm/`) → Results (`results/`)

**Configuration:** All constants (ticker, dates, LSTM hyperparameters) are centralized at the top of `src/data_loader.py`. Other modules import from there.

## Key Design Decisions

- **Time-based splits only** - Train period strictly before test period (no random shuffling)
- **Regression-first approach** - Models predict returns; direction derived from sign of prediction
- **Balanced sample weights** - RF and XGBoost use balanced weights during training to handle class imbalance
- **Fair LSTM comparison** - Baseline predictions truncated to match LSTM's effective test window (accounts for LOOKBACK)
- **Threshold tuning** - All models tune optimal classification threshold using quantile-based search
