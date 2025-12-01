from pydantic import BaseModel
from typing import Dict

class PredictionResponse(BaseModel):
    ticker: str
    undervalued_probability: float
    predicted_label: int
    confidence_category: str
    recommendation: str
    model_version: str
    features: Dict[str, float]