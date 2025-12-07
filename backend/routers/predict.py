from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import math

from backend.services.model_loader import get_model, get_feature_columns, get_model_version
from backend.services.sentiment_loader import compute_sentiment_for_ticker
from backend.models.prediction import PredictionResponse

router = APIRouter(prefix="/predict", tags=["predict"])


class PredictRequest(BaseModel):
    ticker: str = Field(..., example="AAPL")
    MarketCap: float | None = None
    PE_Ratio: float | None = None
    PB_Ratio: float | None = None
    PS_Ratio: float | None = None
    EV_EBITDA: float | None = None
    ROE: float | None = None
    FCF_Yield: float | None = None
    Quick_Ratio: float | None = None


def sanitize(val):
    """
    Preserve missing values as NaN — XGBoost supports missing-handling natively.
    """
    try:
        if val is None:
            return float('nan')
        if isinstance(val, float) and math.isnan(val):
            return float('nan')
        return float(val)
    except:
        return float('nan')


@router.post("/", response_model=PredictionResponse)
def predict_stock(request: PredictRequest):
    try:
        model = get_model()
        feature_columns = get_feature_columns()

        clean = {}
        for col in feature_columns:
            clean[col] = sanitize(getattr(request, col))

        df = pd.DataFrame([clean])

        prob = float(model.predict_proba(df)[0][1])
        label = int(model.predict(df)[0])

        if prob < 0.3:
            category = "Likely Overvalued"
            rec = "AVOID"
        elif prob < 0.5:
            category = "Fairly Valued"
            rec = "HOLD"
        elif prob < 0.7:
            category = "Slightly Undervalued"
            rec = "CONSIDER"
        else:
            category = "Very Undervalued"
            rec = "STRONG BUY"

# Compute finBERT sentiment (separate, optional)
        finbert_polarity = None
        finbert_count = None
        finbert_category = None
        
        try:
            sentiment = compute_sentiment_for_ticker(request.ticker, name=request.name or None)
            finbert_polarity = sentiment.get("finbert_polarity_avg")
            finbert_count = sentiment.get("finbert_count")
            
            # Map finbert score to category
            if finbert_polarity is not None:
                if finbert_polarity > 0.3:
                    finbert_category = "Very Positive"
                elif finbert_polarity > 0.1:
                    finbert_category = "Positive"
                elif finbert_polarity > -0.1:
                    finbert_category = "Neutral"
                elif finbert_polarity > -0.3:
                    finbert_category = "Negative"
                else:
                    finbert_category = "Very Negative"
        except Exception as e:
            # Sentiment computation optional; don't fail the whole prediction
            import logging
            logging.exception("FinBERT sentiment failed: %s", e)
        
        return PredictionResponse(
            ticker=request.ticker.upper(),
            undervalued_probability=prob,
            predicted_label=label,
            confidence_category=category,
            recommendation=rec,
            model_version=get_model_version(),
            features=df.iloc[0].to_dict()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
