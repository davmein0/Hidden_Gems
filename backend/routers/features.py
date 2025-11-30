from fastapi import APIRouter, HTTPException
from backend.services.feature_loader import load_features_for_ticker

router = APIRouter(prefix="/features", tags=["features"])

@router.get("/{ticker}")
def get_features(ticker: str):
    try:
        feats = load_features_for_ticker(ticker)
        return feats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
