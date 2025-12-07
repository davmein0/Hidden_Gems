# =====================================
# Train XGBoost Model for Undervalued Stocks
# =====================================

import pandas as pd
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt

# --- 1. Load the Data ---
df = pd.read_csv("./data/example.csv")
print("Data Loaded:", df.shape)
print(df.head())

# --- 2. Define Features & Target ---
# Assuming your dataset has a 'Label' column indicating undervalued (1) vs not (0)
X = df[['MarketCap', 'PE_Ratio', 'PB_Ratio', 'DE_Ratio', 'FreeCashFlow']]
y = df['Label']

# --- 3. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Initialize XGBoost Model ---
model = XGBClassifier(
    n_estimators=10, # Small example data set
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    objective='binary:logistic',
    use_label_encoder=False,
    random_state=42
)

# --- 5. Train the Model ---
model.fit(X_train, y_train)
print("✅ Model training complete!")

# --- 6. Evaluate ---
y_pred = model.predict(X_test)
# predict_proba may not be available for some estimators; protect call
try:
    y_pred_proba = model.predict_proba(X_test)[:, 1]
except Exception:
    y_pred_proba = None

print("\n--- Model Performance ---")
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
if y_pred_proba is not None:
    try:
        roc = roc_auc_score(y_test, y_pred_proba)
    except Exception:
        roc = None
    print("ROC AUC:", roc)
else:
    print("ROC AUC: predict_proba not available")

print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# --- 7. Feature Importance ---
# --- 7. Feature Importance (safe) ---
import numpy as np
try:
    booster = model.get_booster()
    scores = booster.get_score(importance_type='weight')
except Exception:
    scores = {}

if scores:
    plt.figure(figsize=(8, 5))
    # plot_importance accepts a Booster; pass booster for clarity
    plot_importance(booster, max_num_features=3)
    plt.title("Top 3 Most Important Features")
    plt.show()
else:
    # fallback to sklearn attribute
    fi = getattr(model, "feature_importances_", None)
    if fi is not None and np.any(fi):
        feat_names = X.columns
        inds = np.argsort(fi)[::-1][:3]
        plt.figure(figsize=(8, 5))
        plt.bar([feat_names[i] for i in inds], fi[inds])
        plt.title("Top 3 Most Important Features (sklearn attribute)")
        plt.show()
    else:
        print("No feature importance available (booster.get_score() empty and feature_importances_ is empty). Skipping plot.")