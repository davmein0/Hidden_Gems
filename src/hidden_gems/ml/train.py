# -*- coding: utf-8 -*-
"""Training entrypoint for XGBoost model.
Moved from train_xgboost.py and adapted to package layout.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance
import joblib
from hidden_gems.io import processed_path, models_dir, model_path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)


def main(dataset_path: str | None = None, model_path_arg: str | None = None):
    """Train an XGBoost classifier on provided dataset (defaults to ./data/example.csv)."""
    if dataset_path is None:
        # Prefer labeled processed file, then merged combined, then example
        default_labeled = processed_path("labeled_from_merged.csv")
        default_combined = processed_path("merged_combined.csv")
        example = Path(__file__).resolve().parents[3] / "data" / "example.csv"
        if default_labeled.exists():
            dataset_path = str(default_labeled)
        elif default_combined.exists():
            dataset_path = str(default_combined)
        else:
            dataset_path = os.environ.get("DATASET_PATH", str(example))
    # If dataset is provided as a Path-like string from env or CLI, coerce to Path
    dataset_path = str(dataset_path)

    print("Loading dataset:", dataset_path)
    df = pd.read_csv(dataset_path)
    print("Data Loaded:", df.shape)
    print(df.head())

    # If Label column is missing, create a simple one using a fallback heuristic
    if 'Label' not in df.columns:
        print("No 'Label' column found in dataset. Creating labels with a simple heuristic:")
        print("Label = 1 if PE < 15 and PB < 3 and FreeCashFlow > 0 else 0")
        def gen_label(row):
            try:
                pe = float(row.get('PE_Ratio')) if row.get('PE_Ratio') not in (None, '', 'inf') else None
            except Exception:
                pe = None
            try:
                pb = float(row.get('PB_Ratio')) if row.get('PB_Ratio') not in (None, '', 'inf') else None
            except Exception:
                pb = None
            try:
                fcf = float(row.get('FreeCashFlow')) if row.get('FreeCashFlow') not in (None, '') else None
            except Exception:
                fcf = None
            if pe is not None and pb is not None and fcf is not None:
                return int(pe < 15 and pb < 3 and fcf > 0)
            return 0
        df['Label'] = df.apply(gen_label, axis=1)

    X = df[['MarketCap', 'PE_Ratio', 'PB_Ratio', 'DE_Ratio', 'FreeCashFlow']]
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=10,  # Small example data set
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        objective='binary:logistic',
        random_state=42
    )

    # If a pretrained model path is provided, use it for predictions instead of training
    if model_path_arg:
        if Path(model_path_arg).exists():
            print(f"Loading pretrained model from {model_path_arg}")
            model = joblib.load(model_path_arg)
        else:
            print(f"Pretrained model path {model_path_arg} does not exist; training new model")
            model.fit(X_train, y_train)
            # Save model to models directory
            ensure_model_dir = models_dir()
            ensure_model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path(str(models_dir() / 'xgb_undervalued.pkl')))
    else:
        model.fit(X_train, y_train)
        print("✅ Model training complete!")
        # Save trained model
        ensure_model_dir = models_dir()
        ensure_model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path('xgb_undervalued.pkl'))

    y_pred = model.predict(X_test)
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

    # --- Feature Importance (safe) ---
    try:
        booster = model.get_booster()
        scores = booster.get_score(importance_type='weight')
    except Exception:
        scores = {}

    if scores:
        plt.figure(figsize=(8, 5))
        plot_importance(booster, max_num_features=5)
        plt.title("Top 5 Most Important Features")
        plt.show()
    else:
        fi = getattr(model, "feature_importances_", None)
        if fi is not None and np.any(fi):
            feat_names = X.columns
            inds = np.argsort(fi)[::-1][:5]
            plt.figure(figsize=(8, 5))
            plt.bar([feat_names[i] for i in inds], fi[inds])
            plt.title("Top 5 Most Important Features (sklearn attribute)")
            plt.show()
        else:
            print("No feature importance available (booster.get_score() empty and feature_importances_ is empty). Skipping plot.")


if __name__ == "__main__":
    main()
