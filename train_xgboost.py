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
    use_label_encoder=False,
    random_state=42
)

# --- 5. Train the Model ---
model.fit(X_train, y_train)
print("✅ Model training complete!")

# --- 6. Evaluate ---
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Model Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- 7. Feature Importance ---
plt.figure(figsize=(8, 5))
plot_importance(model, max_num_features=5)
plt.title("Top 5 Most Important Features")
plt.show()

# --- 8. Save Model (Optional) ---
# import joblib
# joblib.dump(model, "../models/xgb_undervalued.pkl")
# print("Model saved to ../models/xgb_undervalued.pkl")
