from fastapi import APIRouter
import pandas as pd
from backend.core.config import MIDCAP_FILE

router = APIRouter(prefix="/search", tags=["search"])

@router.get("/{query}")
def search_midcaps(query: str):
    df = pd.read_csv(MIDCAP_FILE)
    q = query.lower()

    filtered = df[
        df["Ticker"].str.lower().str.contains(q) |
        df["Name"].str.lower().str.contains(q)
    ]

    return filtered.to_dict(orient="records")