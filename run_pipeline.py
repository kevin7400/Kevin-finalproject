#!/usr/bin/env python

"""
Command-line entry point for the S&P 500 LSTM pipeline.

Usage (from project root, inside .venv):

    python run_pipeline.py
"""

import os

from finance_lstm.pipeline import run_pipeline


def ensure_venv() -> None:
    """Warn if the user is not inside a virtual environment."""
    if "VIRTUAL_ENV" not in os.environ:
        print("---- ENVIRONMENT WARNING ----")
        print("You are not running inside a virtual environment.")
        print("Recommended steps (from project root):")
        print("  python3 -m venv .venv")
        print("  source .venv/bin/activate")
        print("  pip install -r requirements.txt")
        print()
        # If you want to enforce venv strictly, you could:
        # import sys
        # sys.exit(1)


def main() -> None:
    ensure_venv()
    run_pipeline()


if __name__ == "__main__":
    main()
