"""Inference helpers shared by the CLI and the API."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.features import FEATURES

MODEL_PATH = Path("models/model.joblib")


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


def predict(model, records: list) -> list:
    """records: list of dicts with all FEATURES. Returns prediction + confidence."""
    X = pd.DataFrame(records)[FEATURES].astype("float64")
    proba = model.predict_proba(X)[:, 1]
    return [{"prediction": int(p >= 0.5),
             "probability_disease": float(p),
             "confidence": float(max(p, 1 - p))} for p in proba]


if __name__ == "__main__":
    # python -m src.predict '{"age": 63, ...}'
    rec = json.loads(sys.argv[1])
    print(json.dumps(predict(load_model(), [rec])[0], indent=2))
