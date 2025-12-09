from fastapi import APIRouter, HTTPException
import os
from backend.core.config import MIDCAP_DIR
from backend.services.feature_loader import load_features_for_ticker, has_required_display_fields

router = APIRouter(prefix="/midcaps", tags=["midcaps"])

@router.get("/")
def get_midcaps():
    try:
        tickers = sorted(
            d for d in os.listdir(MIDCAP_DIR)
            if os.path.isdir(MIDCAP_DIR / d)
        )

        visible = []
        for t in tickers:
            try:
                feats = load_features_for_ticker(t)
                if not has_required_display_fields(feats):
                    continue
                visible.append({"Ticker": t, "Name": t})
            except:
                continue

        return visible

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
