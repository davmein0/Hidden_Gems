import os
import pandas as pd
from backend.core.config import MIDCAP_DIR
from backend.services.model_loader import FEATURE_COLUMNS


def safe_to_float(x):
    """
    Convert CSV values into float, preserving missing values as NaN.
    """
    if pd.isna(x):
        return float('nan')

    # Already numeric
    if isinstance(x, (int, float)):
        return float(x)

    try:
        s = str(x).strip().replace(",", "").replace("%", "")
        if s in ["", "-", "--", "—", "None", "nan", "NaN", "Upgrade"]:
            return float('nan')
        return float(s)
    except:
        return float('nan')


def find_latest_dec_column(columns):
    """
    Select the most recent December column.
    """
    dec_cols = [c for c in columns if "Dec" in c or "DEC" in c]
    if not dec_cols:
        return None
    return sorted(dec_cols)[-1]  # Most recent


def load_features_for_ticker(ticker: str):
    """
    Loads a ticker → extracts only relevant rows → keeps NaN (not 0).
    """
    ticker = ticker.upper()
    folder = MIDCAP_DIR / ticker

    ratio_files = [f for f in os.listdir(folder) if "ratio" in f.lower()]
    if not ratio_files:
        raise FileNotFoundError(f"No ratio file for {ticker}")

    df = pd.read_csv(folder / ratio_files[0])
    df.columns = df.columns.str.strip()

    # First column becomes row index
    df = df.set_index(df.columns[0])

    # Convert everything using safe_to_float()
    df = df.applymap(safe_to_float)

    # Find most recent December column
    dec_col = find_latest_dec_column(df.columns)
    if dec_col is None:
        raise ValueError(f"No December column found in ratios for {ticker}")

    # Mapping from your model feature names → CSV row names
    row_map = {
        "MarketCap": "Market Capitalization",
        "PE_Ratio": "PE Ratio",
        "PB_Ratio": "PB Ratio",
        "PS_Ratio": "PS Ratio",
        "EV_EBITDA": "EV/EBITDA Ratio",
        "ROE": "Return on Equity (ROE)",
        "FCF_Yield": "FCF Yield",
        "Quick_Ratio": "Quick Ratio",
    }

    features = {}

    for key in FEATURE_COLUMNS:
        row_label = row_map[key]

        # Find matching row in CSV
        matches = [idx for idx in df.index if row_label.lower() in str(idx).lower()]

        if not matches:
            features[key] = float('nan')   # Missing row → NaN
            continue

        val = df.loc[matches[0], dec_col]

        # Keep NaN as NaN — DO NOT REPLACE WITH ZERO
        features[key] = val

    return features

def has_required_display_fields(features: dict) -> bool:
    """
    Only block stocks missing metrics shown in AnalysisPanel.
    """
    required = ["MarketCap", "EV_EBITDA", "FCF_Yield", "PB_Ratio"]

    for key in required:
        val = features.get(key)
        if val is None or pd.isna(val):
            return False
    return True
