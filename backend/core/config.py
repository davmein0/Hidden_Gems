import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "xgboost_model.pkl"
MODEL_CONFIG_PATH = BASE_DIR / "model_config.json"
MIDCAP_DIR = BASE_DIR / "midcap_financials"
MIDCAP_FILE = BASE_DIR / "midcaps.csv"