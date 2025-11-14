# =====================================
# Part 1: Train XGBoost Classification Model with Probability Scores
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
    #TODO: add additional years when Lucky finishes scraping more data.
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

    # Locate rows - Optimized feature set
    price_row = find_row(df, "Last Close Price")
    marketcap_row = find_row(df, "Market Capitalization")
    pe_row = find_row(df, "PE Ratio")
    pb_row = find_row(df, "PB Ratio")
    ps_row = find_row(df, "PS Ratio")
    ev_ebitda_row = find_row(df, "EV/EBITDA Ratio")
    roe_row = find_row(df, "Return on Equity (ROE)")
    fcf_yield_row = find_row(df, "FCF Yield")
    quick_ratio_row = find_row(df, "Quick Ratio")

    if not price_row or not marketcap_row:
        return None

    try:
        price_start = df.loc[price_row, start_col]
        price_end = df.loc[price_row, end_col]
        marketcap = df.loc[marketcap_row, start_col]
        pe = df.loc[pe_row, start_col] if pe_row else None
        pb = df.loc[pb_row, start_col] if pb_row else None
        ps = df.loc[ps_row, start_col] if ps_row else None
        ev_ebitda = df.loc[ev_ebitda_row, start_col] if ev_ebitda_row else None
        roe = df.loc[roe_row, start_col] if roe_row else None
        fcf_yield = df.loc[fcf_yield_row, start_col] if fcf_yield_row else None
        quick_ratio = df.loc[quick_ratio_row, start_col] if quick_ratio_row else None
    except KeyError:
        return None

    # Check if we have price data
    if not price_start or not price_end:
        return None

    return {
        'price_start': price_start,
        'price_end': price_end,
        'marketcap': marketcap,
        'pe': pe,
        'pb': pb,
        'ps': ps,
        'ev_ebitda': ev_ebitda,
        'roe': roe,
        'fcf_yield': fcf_yield,
        'quick_ratio': quick_ratio
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
        outperformance = stock_return - nasdaq_return
        
        # Create binary label
        label = 1 if outperformance > OUTPERFORMANCE_THRESHOLD else 0

        rows.append({
            "Ticker": ticker,
            "Year": f"20{config['start_year']}-20{config['end_year']}",
            "Price_Start": data['price_start'],
            "Price_End": data['price_end'],
            "Stock_Return": stock_return,
            "NASDAQ_Return": nasdaq_return,
            "Outperformance": outperformance,
            "MarketCap": data['marketcap'],
            "PE_Ratio": data['pe'],
            "PB_Ratio": data['pb'],
            "PS_Ratio": data['ps'],
            "EV_EBITDA": data['ev_ebitda'],
            "ROE": data['roe'],
            "FCF_Yield": data['fcf_yield'],
            "Quick_Ratio": data['quick_ratio'],
            "Label": label
        })

# Combine all data
df_all = pd.DataFrame(rows).dropna()
print(f"✅ Loaded {len(df_all)} ticker-year combinations successfully.")
print(f"   Unique tickers: {df_all['Ticker'].nunique()}")
print(f"\n--- Class Distribution ---")
print(df_all['Label'].value_counts())
print(f"Positive class (Undervalued): {(df_all['Label'].sum() / len(df_all) * 100):.1f}%")
print(f"\n--- Outperformance Stats ---")
print(f"Mean: {df_all['Outperformance'].mean():.2%}")
print(f"Median: {df_all['Outperformance'].median():.2%}")
print("\n--- Year Distribution ---")
print(df_all['Year'].value_counts().sort_index())


# --- 4. Define Features & Target ---
feature_columns = [
    "MarketCap", "PE_Ratio", "PB_Ratio", "PS_Ratio", 
    "EV_EBITDA", "ROE", "FCF_Yield", "Quick_Ratio"
]

X = df_all[feature_columns]
y = df_all["Label"]

print(f"\n--- Using {len(feature_columns)} Features ---")
for col in feature_columns:
    print(f"  • {col}")


# --- 5. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Keep track of test set indices
test_indices = X_test.index

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


# --- 7. Evaluate with Probabilities ---
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of being undervalued

print("\n--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --- 8. Probability Score Analysis ---
test_data = df_all.loc[test_indices].copy()
test_data['Undervalued_Probability'] = y_pred_proba
test_data['Predicted_Label'] = y_pred

# Categorize by confidence
def categorize_confidence(prob):
    if prob < 0.3:
        return "Likely Overvalued"
    elif prob < 0.5:
        return "Fairly Valued"
    elif prob < 0.7:
        return "Slightly Undervalued"
    else:
        return "Very Undervalued"

test_data['Confidence_Category'] = test_data['Undervalued_Probability'].apply(categorize_confidence)

print("\n--- Probability Score Distribution ---")
print(test_data['Confidence_Category'].value_counts().sort_index())

print("\n--- Top 15 Highest Probability Undervalued Stocks ---")
top_picks = test_data.nlargest(15, 'Undervalued_Probability')[
    ['Ticker', 'Year', 'Undervalued_Probability', 'Confidence_Category', 
     'Outperformance', 'Label', 'PE_Ratio', 'PB_Ratio', 'ROE']
]
print(top_picks.to_string(index=False))

print("\n--- Performance by Confidence Category ---")
for category in ["Very Undervalued", "Slightly Undervalued", "Fairly Valued", "Likely Overvalued"]:
    subset = test_data[test_data['Confidence_Category'] == category]
    if len(subset) > 0:
        actual_winners = subset['Label'].sum()
        total = len(subset)
        avg_outperformance = subset['Outperformance'].mean()
        print(f"\n{category}: {total} stocks")
        print(f"  Actual undervalued: {actual_winners}/{total} ({actual_winners/total*100:.1f}%)")
        print(f"  Avg outperformance: {avg_outperformance:.2%}")


# --- 9. Feature Importance ---
plt.figure(figsize=(10, 6))
plot_importance(model, max_num_features=8)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()


# --- 10. Probability Distribution Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Distribution for actual undervalued vs not
ax1.hist(test_data[test_data['Label']==0]['Undervalued_Probability'], 
         bins=20, alpha=0.7, label='Actually Not Undervalued', color='red')
ax1.hist(test_data[test_data['Label']==1]['Undervalued_Probability'], 
         bins=20, alpha=0.7, label='Actually Undervalued', color='green')
ax1.set_xlabel('Predicted Probability of Being Undervalued')
ax1.set_ylabel('Count')
ax1.set_title('Probability Distribution by Actual Label')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Calibration: Does 70% probability mean 70% chance?
bins = [0, 0.3, 0.5, 0.7, 1.0]
labels = ['<0.3', '0.3-0.5', '0.5-0.7', '0.7+']
test_data['prob_bin'] = pd.cut(test_data['Undervalued_Probability'], bins=bins, labels=labels)
calibration = test_data.groupby('prob_bin')['Label'].agg(['mean', 'count'])
ax2.bar(range(len(calibration)), calibration['mean'], alpha=0.7, color='blue')
ax2.set_xticks(range(len(calibration)))
ax2.set_xticklabels(labels)
ax2.set_xlabel('Predicted Probability Range')
ax2.set_ylabel('Actual Success Rate')
ax2.set_title('Model Calibration')
ax2.grid(True, alpha=0.3, axis='y')

# Add count labels on bars
for i, (idx, row) in enumerate(calibration.iterrows()):
    ax2.text(i, row['mean'], f"n={int(row['count'])}", ha='center', va='bottom')

plt.tight_layout()
plt.show()



# =====================================
# Part 2: Saving the Trained Model
# =====================================

import pickle
import json

# Save the trained model
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save feature column names (important!)
feature_info = {
    'feature_columns': feature_columns,
    'training_date': '2025-11-13',
    'model_version': '1.0'
}

with open('model_config.json', 'w') as f:
    json.dump(feature_info, f)

print("✅ Model saved to xgboost_model.pkl")
print("✅ Config saved to model_config.json")



# =====================================
# Part 3: Test the Saved Model
# =====================================

print("\n--- Testing Saved Model ---")

# Load the model back
with open('xgboost_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('model_config.json', 'r') as f:
    loaded_config = json.load(f)

# Test prediction on a sample
test_sample = pd.DataFrame({
    'MarketCap': [5000],
    'PE_Ratio': [12.5],
    'PB_Ratio': [2.1],
    'PS_Ratio': [1.8],
    'EV_EBITDA': [8.5],
    'ROE': [18.5],
    'FCF_Yield': [6.2],
    'Quick_Ratio': [1.5]
})

probability = loaded_model.predict_proba(test_sample)[0][1]
print(f"✅ Model loaded and tested successfully!")
print(f"Test prediction: {probability:.2%} probability of being undervalued")
print(f"Model version: {loaded_config['model_version']}")


'''

Then For the API (GARRETT please look at this part):

Just copy `xgboost_model.pkl` and `model_config.json` to wherever the 
Flask/FastAPI backend runs, and use the following API code. The API will load these files when it starts up.

# =====================================
# API Endpoint (Flask Example) - adapt it to Fast API if you'd like instead of Flask
# =====================================
# This would be in the backend API server

from flask import Flask, request, jsonify
import pickle
import pandas as pd
import json

app = Flask(__name__)

# Load model once when server starts (not on every request!)
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_config.json', 'r') as f:
    model_config = json.load(f)

feature_columns = model_config['feature_columns']


@app.route('/api/analyze/xgboost', methods=['POST'])
def analyze_stock():
    """
    API endpoint to get XGBoost fundamental analysis for a stock
    
    Request body:
    {
        "ticker": "AAPL",
        "MarketCap": 5000,
        "PE_Ratio": 12.5,
        "PB_Ratio": 2.1,
        "PS_Ratio": 1.8,
        "EV_EBITDA": 8.5,
        "ROE": 18.5,
        "FCF_Yield": 6.2,
        "Quick_Ratio": 1.5
    }
    """
    try:
        data = request.json
        ticker = data.get('ticker')
        
        # Extract features in correct order
        features = pd.DataFrame({
            col: [data.get(col)] for col in feature_columns
        })
        
        # Check for missing features
        if features.isnull().any().any():
            missing = features.columns[features.isnull().any()].tolist()
            return jsonify({
                'error': f'Missing features: {missing}'
            }), 400
        
        # Get predictions
        probability = model.predict_proba(features)[0][1]
        prediction = model.predict(features)[0]
        
        # Categorize confidence
        if probability < 0.3:
            category = "Likely Overvalued"
            recommendation = "AVOID"
        elif probability < 0.5:
            category = "Fairly Valued"
            recommendation = "HOLD"
        elif probability < 0.7:
            category = "Slightly Undervalued"
            recommendation = "CONSIDER"
        else:
            category = "Very Undervalued"
            recommendation = "STRONG BUY"
        
        # Return analysis
        return jsonify({
            'ticker': ticker,
            'xgboost_analysis': {
                'undervalued_probability': round(probability, 4),
                'confidence_category': category,
                'recommendation': recommendation,
                'predicted_label': int(prediction),
                'model_version': model_config['model_version']
            },
            'features_analyzed': {
                col: data.get(col) for col in feature_columns
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/combined', methods=['POST'])
def analyze_combined():
    """
    Combined XGBoost + FinBERT analysis (placeholder for future)
    
    Request body:
    {
        "ticker": "AAPL",
        "fundamentals": { ... },
        "sentiment_data": {
            "news_headlines": [...],
            "filing_text": "..."
        }
    }
    """
    try:
        data = request.json
        ticker = data.get('ticker')
        fundamentals = data.get('fundamentals')
        sentiment_data = data.get('sentiment_data')
        
        # Get XGBoost score
        features = pd.DataFrame({
            col: [fundamentals.get(col)] for col in feature_columns
        })
        fundamental_score = model.predict_proba(features)[0][1]
        
        # TODO: Get FinBERT sentiment score
        # sentiment_score = finbert_analyze(sentiment_data)
        # For now, placeholder:
        sentiment_score = 0.0  # Range: -1.0 to 1.0
        
        # Combine scores (60% fundamental, 40% sentiment)
        combined_score = 0.6 * fundamental_score + 0.4 * ((sentiment_score + 1) / 2)
        
        # Final recommendation
        if combined_score > 0.7 and fundamental_score > 0.5:
            final_recommendation = "STRONG BUY"
        elif combined_score > 0.6:
            final_recommendation = "BUY"
        elif combined_score > 0.4:
            final_recommendation = "HOLD"
        else:
            final_recommendation = "AVOID"
        
        return jsonify({
            'ticker': ticker,
            'fundamental_score': round(fundamental_score, 4),
            'sentiment_score': round(sentiment_score, 4),
            'combined_score': round(combined_score, 4),
            'recommendation': final_recommendation,
            'analysis': {
                'xgboost': {
                    'probability': round(fundamental_score, 4),
                    'features': fundamentals
                },
                'finbert': {
                    'sentiment': round(sentiment_score, 4),
                    'status': 'not_implemented'  # TODO
                }
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)


# =====================================
# Frontend Integration Example - for Lucky to reference
# =====================================

JavaScript fetch example for your frontend:

// When user clicks on a stock
async function analyzeStock(ticker, financialData) {
    const response = await fetch('http://your-api.com/api/analyze/xgboost', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            ticker: ticker,
            MarketCap: financialData.marketCap,
            PE_Ratio: financialData.peRatio,
            PB_Ratio: financialData.pbRatio,
            PS_Ratio: financialData.psRatio,
            EV_EBITDA: financialData.evEbitda,
            ROE: financialData.roe,
            FCF_Yield: financialData.fcfYield,
            Quick_Ratio: financialData.quickRatio
        })
    });
    
    const result = await response.json();
    
    // Display results
    console.log(result);
    // {
    //   ticker: "AAPL",
    //   xgboost_analysis: {
    //     undervalued_probability: 0.7234,
    //     confidence_category: "Very Undervalued",
    //     recommendation: "STRONG BUY"
    //   }
    // }
}
'''