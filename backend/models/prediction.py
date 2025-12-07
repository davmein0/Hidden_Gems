from pydantic import BaseModel
from typing import Dict, Optional

class PredictionResponse(BaseModel):
    ticker: str
    undervalued_probability: float
    predicted_label: int
    confidence_category: str
    recommendation: str
    model_version: str
    features: Dict[str, float]

    # NEW: separate finbert sentiment fields
    finbert_polarity_avg: Optional[float] = None
    finbert_count: Optional[int] = None
    finbert_category: Optional[str] = None