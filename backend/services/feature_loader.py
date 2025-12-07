import os
import pandas as pd
from backend.core.config import MIDCAP_DIR
from backend.services.model_loader import FEATURE_COLUMNS

def safe_to_float(x):
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return x
    try:
        x = str(x).strip().replace(",", "").replace("%", "")
        if x in ["-", "", "Upgrade"]:
            return None
        return float(x)
    except:
        return None
    
def find_latest_december_columns(columns):
    candidates = [c for c in columns if "Dec" in c or "DEC" in c]
    if not candidates:
        return None
    
    return sorted(candidates)[-1]

def load_features_for_ticker(ticker: str):
    ticker = ticker.upper()
    folder = MIDCAP_DIR / ticker

    ratio_files = [f for f in os.listdir(folder) if "ratio" in f.lower()]
    if not ratio_files:
        raise FileNotFoundError(f"No ratios CSV found for {ticker}")
    
    path = folder / ratio_files[0]
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()
    df = df.set_index(df.columns[0])
    df = df.map(safe_to_float)

    latest_col = find_latest_december_columns(df.columns)
    if latest_col is None:
        raise ValueError(f"No December column found in {ticker} ratios")
    
    row_map = {
        "MarketCap": "Market Capitalization",
        "PE_Ratio": "PE Ratio",
        "PB_Ratio": "PB Ratio",
        "PS_Ratio": "PS Ratio",
        "EV_EBITDA": "EV/EBITDA Ratio",
        "ROE": "Return on Equity (ROE)",
        "FCF_Yield": "FCF Yield",
        "Quick_Ratio": "Quick Ratio"
    }

    features = {}
    for key in FEATURE_COLUMNS:
        row_key = row_map[key]
        matching_rows = [r for r in df.index if row_key.lower() in str(r).lower()]
        if not matching_rows:
            raise ValueError(f"Missing row '{row_key} in {ticker} ratios")
        features[key] = float(df.loc[matching_rows[0], latest_col])

    return features