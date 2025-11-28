# =====================================
# Interactive Stock Prediction Demo
# =====================================

import os
import pandas as pd
import pickle
import json

DATA_DIR = "./midcap_financials"

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


def find_row(df, key):
    """Find row index that contains the key string."""
    for idx in df.index:
        if key.lower() in str(idx).lower().replace('"', '').strip():
            return idx
    return None


def get_latest_stock_data(ticker):
    """
    Load the most recent financial data for a given ticker.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'AAL')
    
    Returns:
        dict with feature values, or None if ticker not found
    """
    ticker = ticker.upper().strip()
    folder_path = os.path.join(DATA_DIR, ticker)
    
    # Check if ticker folder exists
    if not os.path.isdir(folder_path):
        print(f"❌ Ticker '{ticker}' not found in database.")
        available_tickers = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        print(f"Available tickers: {', '.join(sorted(available_tickers[:10]))}... (showing first 10)")
        return None
    
    # Find ratio file
    ratio_files = [f for f in os.listdir(folder_path) if "ratio" in f.lower()]
    if not ratio_files:
        print(f"❌ No ratio data found for {ticker}")
        return None
    
    ratio_path = os.path.join(folder_path, ratio_files[0])
    df = pd.read_csv(ratio_path)
    
    # Clean the dataframe
    df.columns = df.columns.str.strip()
    df = df.set_index(df.columns[0])
    df = df.map(safe_to_float, na_action='ignore') if hasattr(df, 'map') else df.applymap(safe_to_float)
    
    # Get the most recent column (first data column after index)
    latest_col = df.columns[0]  # First column is most recent
    print(f"📊 Using data from: {latest_col}")
    
    # Locate required rows
    marketcap_row = find_row(df, "Market Capitalization")
    pe_row = find_row(df, "PE Ratio")
    pb_row = find_row(df, "PB Ratio")
    ps_row = find_row(df, "PS Ratio")
    ev_ebitda_row = find_row(df, "EV/EBITDA Ratio")
    roe_row = find_row(df, "Return on Equity (ROE)")
    fcf_yield_row = find_row(df, "FCF Yield")
    quick_ratio_row = find_row(df, "Quick Ratio")
    
    # Extract values
    try:
        features = {
            'MarketCap': df.loc[marketcap_row, latest_col] if marketcap_row else None,
            'PE_Ratio': df.loc[pe_row, latest_col] if pe_row else None,
            'PB_Ratio': df.loc[pb_row, latest_col] if pb_row else None,
            'PS_Ratio': df.loc[ps_row, latest_col] if ps_row else None,
            'EV_EBITDA': df.loc[ev_ebitda_row, latest_col] if ev_ebitda_row else None,
            'ROE': df.loc[roe_row, latest_col] if roe_row else None,
            'FCF_Yield': df.loc[fcf_yield_row, latest_col] if fcf_yield_row else None,
            'Quick_Ratio': df.loc[quick_ratio_row, latest_col] if quick_ratio_row else None
        }
        
        # Check for missing critical features
        missing = [k for k, v in features.items() if v is None]
        if missing:
            print(f"⚠️  Warning: Missing features for {ticker}: {', '.join(missing)}")
            print("Prediction may be less accurate.")
        
        return features
        
    except Exception as e:
        print(f"❌ Error extracting data for {ticker}: {e}")
        return None


def categorize_confidence(prob):
    """Convert probability to human-readable category"""
    if prob < 0.3:
        return "Likely Overvalued", "🔴 AVOID"
    elif prob < 0.5:
        return "Fairly Valued", "🟡 HOLD"
    elif prob < 0.7:
        return "Slightly Undervalued", "🟢 CONSIDER"
    else:
        return "Very Undervalued", "🟢 STRONG BUY"


def predict_stock(ticker, model, config):
    """
    Make prediction for a given ticker
    
    Args:
        ticker: Stock symbol
        model: Loaded XGBoost model
        config: Model configuration dict
    
    Returns:
        dict with prediction results
    """
    # Get latest data
    features = get_latest_stock_data(ticker)
    
    if features is None:
        return None
    
    # Prepare DataFrame in correct order
    feature_columns = config['feature_columns']
    df = pd.DataFrame({col: [features.get(col)] for col in feature_columns})
    
    # CRITICAL FIX: Ensure all columns are numeric, replace None with NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check if we have too many missing features
    missing_count = df.isnull().sum().sum()
    if missing_count > 4:  # Allow up to 4 missing features
        print(f"❌ Too many missing features ({missing_count}/8) for {ticker}")
        print("   Cannot make reliable prediction.")
        return None
    
    # Make prediction (XGBoost can handle some NaN values)
    try:
        probability = model.predict_proba(df)[0][1]
        prediction = model.predict(df)[0]
        category, recommendation = categorize_confidence(probability)
        
        return {
            'ticker': ticker,
            'probability': probability,
            'category': category,
            'recommendation': recommendation,
            'predicted_label': int(prediction),
            'features': features
        }
    except Exception as e:
        print(f"❌ Error making prediction for {ticker}: {e}")
        return None


def print_prediction_results(result):
    """Pretty print prediction results"""
    print("\n" + "="*60)
    print(f"📈 PREDICTION RESULTS FOR {result['ticker']}")
    print("="*60)
    print(f"\n🎯 Undervalued Probability: {result['probability']:.1%}")
    print(f"📊 Category: {result['category']}")
    print(f"💡 Recommendation: {result['recommendation']}")
    
    print(f"\n📋 Features Analyzed:")
    for feature, value in result['features'].items():
        if value is not None:
            print(f"   • {feature}: {value:,.2f}")
        else:
            print(f"   • {feature}: N/A")
    
    print("\n" + "="*60)


def interactive_demo():
    """
    Interactive command-line demo for stock prediction
    """
    print("\n" + "="*60)
    print("🚀 STOCK UNDERVALUATION PREDICTOR")
    print("="*60)
    print("\nThis demo uses XGBoost to analyze fundamental financial")
    print("metrics and predict if a stock is undervalued.\n")
    
    # Load model
    try:
        with open('xgboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        
        print("✅ Model loaded successfully!")
        print(f"   Model version: {config['model_version']}")
        print(f"   Training date: {config['training_date']}")
        
    except FileNotFoundError:
        print("❌ Error: Model files not found!")
        print("Please run the training script first to generate:")
        print("  - xgboost_model.pkl")
        print("  - model_config.json")
        return
    
    # Main loop
    while True:
        print("\n" + "-"*60)
        ticker = input("\n🔍 Enter stock ticker (or 'quit' to exit): ").strip().upper()
        
        if ticker.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thanks for using the Stock Predictor!")
            break
        
        if not ticker:
            print("⚠️  Please enter a valid ticker symbol.")
            continue
        
        # Make prediction
        result = predict_stock(ticker, model, config)
        
        if result:
            print_prediction_results(result)
            
            # Ask if user wants to see another
            another = input("\n❓ Analyze another stock? (y/n): ").strip().lower()
            if another not in ['y', 'yes', '']:
                print("\n👋 Thanks for using the Stock Predictor!")
                break


def batch_predict(tickers):
    """
    Predict multiple tickers at once
    
    Args:
        tickers: List of ticker symbols
    
    Returns:
        List of prediction results
    """
    # Load model
    with open('xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('model_config.json', 'r') as f:
        config = json.load(f)
    
    results = []
    for ticker in tickers:
        result = predict_stock(ticker, model, config)
        if result:
            results.append(result)
    
    return results


# =====================================
# Run the Demo
# =====================================

if __name__ == '__main__':
    # Run interactive demo
    interactive_demo()
    
    # Example: Batch prediction (uncomment to use)
    # tickers_to_analyze = ['AAPL', 'AAL', 'MSFT', 'TSLA']
    # results = batch_predict(tickers_to_analyze)
    # for result in results:
    #     print_prediction_results(result)