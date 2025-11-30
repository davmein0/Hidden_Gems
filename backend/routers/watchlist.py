from fastapi import APIRouter
from backend.core.config import MIDCAP_DIR
from backend.services.feature_loader import load_features_for_ticker
from backend.services.model_loader import get_model, get_feature_columns
import pandas as pd
import os

router = APIRouter(prefix="/watchlist", tags=["watchlist"])

@router.get("/")
def get_watchlist():
    tickers = [d for d in os.listdir(MIDCAP_DIR) if os.path.isdir(MIDCAP_DIR / d)]

    model = get_model()
    feature_columns = get_feature_columns()

    results = []
    for t in tickers:
        try:
            feats = load_features_for_ticker(t)
            df = pd.DataFrame([feats], columns=feature_columns)
            prob = float(model.predict_proba(df)[0][1])

            results.append({"ticker": t, "probability": prob})
        except:
            continue

    ranked = sorted(results, key=lambda x: x["probability"], reverse=True)[:15]
    return ranked