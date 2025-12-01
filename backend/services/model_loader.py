import pickle
import json
from backend.core.config import MODEL_PATH, MODEL_CONFIG_PATH

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