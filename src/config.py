from pathlib import Path

"""
Project-wide paths.

Why:
- Avoid hardcoding file paths in multiple scripts.
- Keep the project structure consistent and easy to maintain.
"""

# Project root = parent directory of /src
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_TELCO_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RAW_TELCO_PATH = RAW_DIR / RAW_TELCO_FILENAME

PROCESSED_TELCO_FILENAME = "telco_clean.csv"
PROCESSED_TELCO_PATH = PROCESSED_DIR / PROCESSED_TELCO_FILENAME
