from fastapi import APIRouter, HTTPException
from backend.services.sentiment_loader import compute_sentiment_for_ticker

router = APIRouter(prefix="/sentiment", tags=["sentiment"])

@router.get("/{symbol}")
def get_sentiment(symbol: str, name: str = None):
    """Fetch and return finBERT sentiment for a ticker."""
    try:
        sentiment = compute_sentiment_for_ticker(symbol, name=name)
        return {
            "ticker": symbol.upper(),
            "finbert_polarity_avg": round(sentiment.get("finbert_polarity_avg"), 2) if sentiment.get("finbert_polarity_avg") is not None else None,
            "finbert_count": sentiment.get("finbert_count"),
            "finbert_category": _categorize_sentiment(sentiment.get("finbert_polarity_avg"))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _categorize_sentiment(score):
    if score is None:
        return "Unknown"
    if score > 0.3:
        return "Very Positive"
    elif score > 0.1:
        return "Positive"
    elif score > -0.1:
        return "Neutral"
    elif score > -0.3:
        return "Negative"
    else:
        return "Very Negative"