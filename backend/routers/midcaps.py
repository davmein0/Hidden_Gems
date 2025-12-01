from fastapi import APIRouter, HTTPException
import os
from backend.core.config import MIDCAP_DIR

router = APIRouter(prefix="/midcaps", tags=["midcaps"])

@router.get("/")
def get_midcaps():
    try:
        tickers = sorted(
            d for d in os.listdir(MIDCAP_DIR)
            if os.path.isdir(MIDCAP_DIR / d)
        )

        return [{"Ticker": t, "Name": t} for t in tickers]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
