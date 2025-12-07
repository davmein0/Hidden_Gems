from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from backend.services.model_loader import get_model, get_feature_columns, get_model_version
from backend.models.prediction import PredictionResponse

router = APIRouter(prefix="/predict", tags=["predict"])

# ---------------------------
# Request body schema
# ---------------------------
class PredictRequest(BaseModel):
    ticker: str = Field(..., example="AAPL")
    MarketCap: float
    PE_Ratio: float
    PB_Ratio: float
    PS_Ratio: float
    EV_EBITDA: float
    ROE: float
    FCF_Yield: float
    Quick_Ratio: float

# ---------------------------
# POST /predict  (JSON input)
# ---------------------------
@router.post("/", response_model=PredictionResponse)
def predict_stock(request: PredictRequest):
    try:
        model = get_model()
        feature_columns = get_feature_columns()

        # Convert request → DataFrame in correct feature order
        df = pd.DataFrame({
            col: [getattr(request, col)] for col in feature_columns
        })

        # Prediction
        prob = float(model.predict_proba(df)[0][1])
        label = int(model.predict(df)[0])

        # Confidence categories
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
