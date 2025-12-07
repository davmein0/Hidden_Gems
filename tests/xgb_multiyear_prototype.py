# =====================================
# Train XGBoost Model for Undervalued Stocks (Multi-Year)
# =====================================

import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt

# --- 1. Config ---
DATA_DIR = "./midcap_financials"

# Define year pairs: (start_year, end_year, nasdaq_start, nasdaq_end)
YEAR_CONFIGS = [
    {
        'start_year': '20',
        'end_year': '21',
        'nasdaq_start': 8952.88,  # Dec 31, 2020
        'nasdaq_end': 15644.97    # Dec 31, 2021
    },
    {
        'start_year': '21',
        'end_year': '22',
        'nasdaq_start': 15644.97,  # Dec 31, 2021
        'nasdaq_end': 10466.48     # Dec 31, 2022
    },
    {
        'start_year': '22',
        'end_year': '23',
        'nasdaq_start': 10466.48,  # Dec 31, 2022
        'nasdaq_end': 15011.35     # Dec 31, 2023
    },
    {
        'start_year': '23',
        'end_year': '24',
        'nasdaq_start': 15011.35,  # Dec 31, 2023
        'nasdaq_end': 19310.79     # Dec 31, 2024
    }
]

OUTPERFORMANCE_THRESHOLD = 0.10  # 10% outperformance required


# --- 2. Helpers ---

def safe_to_float(x):
    """Convert messy strings like '4,532', '-14.56%', or '-' to float or NaN."""
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return x
    try:
        x = str(x).strip().replace(",", "").replace("%", "")
        if x in ["-", "", "Upgrade"]:
            return None
        return float(x)
    except ValueError:
        return None


def find_col(cols, year):
    """Find column containing December data for a given 2-digit year (e.g. '23' -> Dec 2023)."""
    year = year.strip().replace("'", "")
    for c in cols:
        clean = c.lower().replace(" ", "").replace("'", "")
        if f"dec{year}" in clean or f"dec31,20{year}" in clean:
            return c
    return None


def find_row(df, key):
    """Find row index that contains the key string."""
    for idx in df.index:
        if key.lower() in str(idx).lower().replace('"', '').strip():
            return idx
    return None


def load_year_data(ticker, folder_path, start_year, end_year):
    """
    Load data for a specific ticker and year pair.
    Returns dict with features and label, or None if data unavailable.
    """
    ratio_files = [f for f in os.listdir(folder_path) if "ratio" in f.lower()]
    if not ratio_files:
        return None

    ratio_path = os.path.join(folder_path, ratio_files[0])
    df = pd.read_csv(ratio_path)

    # Clean the dataframe
    df.columns = df.columns.str.strip()
    df = df.set_index(df.columns[0])
    df = df.map(safe_to_float, na_action='ignore') if hasattr(df, 'map') else df.applymap(safe_to_float)

    # Find columns for start and end year
    cols = df.columns
    start_col = find_col(cols, start_year)
    end_col = find_col(cols, end_year)

    if not start_col or not end_col:
        return None

    # Locate rows
    price_row = find_row(df, "Last Close Price")
    marketcap_row = find_row(df, "Market Capitalization")
    pe_row = find_row(df, "PE Ratio")
    pb_row = find_row(df, "PB Ratio")
    de_row = find_row(df, "Debt / Equity Ratio")
    fcf_yield_row = find_row(df, "FCF Yield")

    if not price_row or not marketcap_row:
        return None

    try:
        price_start = df.loc[price_row, start_col]
        price_end = df.loc[price_row, end_col]
        marketcap_start = df.loc[marketcap_row, start_col]
        pe_start = df.loc[pe_row, start_col] if pe_row else None
        pb_start = df.loc[pb_row, start_col] if pb_row else None
        de_start = df.loc[de_row, start_col] if de_row else None
        fcf_yield_start = df.loc[fcf_yield_row, start_col] if fcf_yield_row else None
    except KeyError:
        return None

    # Check if we have price data
    if not price_start or not price_end:
        return None

    return {
        'price_start': price_start,
        'price_end': price_end,
        'marketcap_start': marketcap_start,
        'pe': pe_start,
        'pb': pb_start,
        'de': de_start,
        'fcf_yield': fcf_yield_start
    }


# --- 3. Load all tickers for all years ---
rows = []

for ticker in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, ticker)
    if not os.path.isdir(folder_path):
        continue

    # Try to load data for each year configuration
    for config in YEAR_CONFIGS:
        data = load_year_data(
            ticker, 
            folder_path, 
            config['start_year'], 
            config['end_year']
        )
        
        if data is None:
            continue

        # Calculate returns
        nasdaq_return = (config['nasdaq_end'] - config['nasdaq_start']) / config['nasdaq_start']
        stock_return = (data['price_end'] - data['price_start']) / data['price_start']
        
        # Create label
        label = 1 if (stock_return - nasdaq_return) > OUTPERFORMANCE_THRESHOLD else 0

        rows.append({
            "Ticker": ticker,
            "Year": f"20{config['start_year']}-20{config['end_year']}",
            "Price_Start": data['price_start'],
            "Price_End": data['price_end'],
            "Stock_Return": stock_return,
            "NASDAQ_Return": nasdaq_return,
            "MarketCap": data['marketcap_start'],
            "PE_Ratio": data['pe'],
            "PB_Ratio": data['pb'],
            "DE_Ratio": data['de'],
            "FCF_Yield": data['fcf_yield'],
            "Label": label
        })

# Combine all data
df_all = pd.DataFrame(rows).dropna()
print(f"✅ Loaded {len(df_all)} ticker-year combinations successfully.")
print(f"   Unique tickers: {df_all['Ticker'].nunique()}")
print(f"\n--- Class Distribution ---")
print(df_all['Label'].value_counts())
print(f"Positive class: {(df_all['Label'].sum() / len(df_all) * 100):.1f}%")
print("\n--- Year Distribution ---")
print(df_all['Year'].value_counts().sort_index())
print("\nSample data:")
print(df_all.head(10))


# --- 4. Define Features & Target ---
X = df_all[["MarketCap", "PE_Ratio", "PB_Ratio", "DE_Ratio", "FCF_Yield"]]
y = df_all["Label"]


# --- 5. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n--- Train/Test Split ---")
print(f"Training samples: {len(X_train)} (Positive: {y_train.sum()})")
print(f"Test samples: {len(X_test)} (Positive: {y_test.sum()})")


# --- 6. Train Model ---
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=len(y[y==0]) / len(y[y==1]),
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train, y_train)
print("\n✅ Model training complete!")


# --- 7. Evaluate ---
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Model Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# --- 8. Feature Importance ---
plt.figure(figsize=(8, 5))
plot_importance(model, max_num_features=5)
plt.title("Top 5 Most Important Features")
plt.tight_layout()
plt.show()