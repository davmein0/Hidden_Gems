# =====================================
# Train XGBoost Model for Undervalued Stocks
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

############# CHANGE THESE BENCHMARKS DEPENDING ON TRAINING YEAR #################
# Dec 31, 2023 - Dec 31, 2024 closing prices
NASDAQ_START = 15011.35
NASDAQ_END = 19310.79

# Define how much the stock must overperform the NASDAQ to be considered undervalued
OUTPERFORMANCE = 0.10  # > 10%
###################################################################################

NASDAQ_RETURN = (NASDAQ_END - NASDAQ_START) / NASDAQ_START



# --- 2. Helpers ---

# Parse numeric strings safely
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
    for idx in df.index:
        if key.lower() in str(idx).lower().replace('"', '').strip():
            return idx
    return None



# --- 3. Load all tickers ---

rows = []
for ticker in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, ticker)
    if not os.path.isdir(folder_path):
        continue

    ratio_files = [f for f in os.listdir(folder_path) if "ratio" in f.lower()]
    if not ratio_files:
        continue

    ratio_path = os.path.join(folder_path, ratio_files[0])
    df = pd.read_csv(ratio_path)

    # Clean the dataframe: set first column as index (metric names)
    df.columns = df.columns.str.strip()
    df = df.set_index(df.columns[0])
    print(f"\n{ticker} index sample:", list(df.index[:10])) #debugging
    df = df.map(safe_to_float, na_action='ignore') if hasattr(df, 'map') else df.applymap(safe_to_float)

    '''# --- DEBUG ONE SAMPLE ---
    if ticker in ["AAL", "NVDA", "AAPL"]:  # pick one or two tickers that exist
        print(f"\n\n========== {ticker} SAMPLE ==========")
        print("Columns:", list(df.columns))
        print(df.head(5))
        print("====================================\n\n")
        break  # stop after printing one ticker so the output is readable
    '''

    # Use Dec 2023 and Dec 2024 columns
    cols = df.columns

    dec2024_col = find_col(cols, "24")
    dec2023_col = find_col(cols, "23")

    if not dec2024_col or not dec2023_col:
        print(f"[!] Skipping {ticker} — couldn't find Dec 2023/2024 columns.")
        continue

    # Locate rows
    marketcap_row = find_row(df, "Market Capitalization")
    pe_row        = find_row(df, "PE Ratio")
    pb_row        = find_row(df, "PB Ratio")
    de_row        = find_row(df, "Debt / Equity Ratio")
    fcf_yield_row = find_row(df, "FCF Yield")

    if not marketcap_row:
        print(f"[!] {ticker} missing Market Cap row.")
        continue


    try:
        marketcap_2023 = df.loc[marketcap_row, dec2023_col]
        marketcap_2024 = df.loc[marketcap_row, dec2024_col]
        pe_2023 = df.loc[pe_row, dec2023_col] if pe_row else None
        pb_2023 = df.loc[pb_row, dec2023_col] if pb_row else None
        de_2023 = df.loc[de_row, dec2023_col] if de_row else None
        fcf_yield_2023 = df.loc[fcf_yield_row, dec2023_col] if fcf_yield_row else None
    except KeyError:
        print(f"[!] Skipping {ticker} — missing Market Cap row.")
        continue


   # Compute stock return & label
    if marketcap_2023 and marketcap_2024:
        stock_return = (marketcap_2024 - marketcap_2023) / marketcap_2023
        label = 1 if (stock_return - NASDAQ_RETURN) > OUTPERFORMANCE else 0
    else:
        continue

    rows.append({
        "Ticker": ticker,
        "MarketCap": marketcap_2023,
        "PE_Ratio": pe_2023,
        "PB_Ratio": pb_2023,
        "DE_Ratio": de_2023,
        "FreeCashFlow": fcf_yield_2023,
        "Label": label
    })

# Combine all tickers
data = pd.DataFrame(rows).dropna()
print(f"✅ Loaded {len(data)} tickers successfully.")
print(data.head())


######## CHECKING DATA ########
print("\n--- Class Distribution ---")
print(data['Label'].value_counts())
print(f"Positive class: {(data['Label'].sum() / len(data) * 100):.1f}%")
###############################


# --- 4. Define Features & Target ---

X = data[["MarketCap", "PE_Ratio", "PB_Ratio", "DE_Ratio", "FreeCashFlow"]]
y = data["Label"]



# --- 5. Train/Test Split ---

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



# --- 6. Train Model ---

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=len(y[y==0]) / len(y[y==1]),  # penalize missing undervalued stocks
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train, y_train)
print("✅ Model training complete!")



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
plt.show()
