import pickle
import json
import joblib
import xgboost as xgb
from backend.core.config import MODEL_PATH, MODEL_CONFIG_PATH

## Load model and save configuration: pickled XGBoost model not compatible with updated xgboost
# with open(MODEL_PATH, "rb") as f:
#     model = joblib.load(f)

# booster = model.get_booster()

# booster.save_model("xgboost_model.json")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(MODEL_CONFIG_PATH, "r") as f:
    model_config = json.load(f)

FEATURE_COLUMNS = model_config["feature_columns"]
MODEL_VERSION = model_config["model_version"]

def get_model():
    return model

def get_feature_columns():
    return FEATURE_COLUMNS

def get_model_version():
    return MODEL_VERSION