"""FastAPI service exposing the heart disease model.

Run: uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.predict import load_model, predict

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
META_PATH = MODEL_PATH.with_name("model_metadata.json")

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load_model(MODEL_PATH)
    state["meta"] = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    yield
    state.clear()


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


class Patient(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3,
        "slope": 3, "ca": 0, "thal": 6}})

    age: float = Field(ge=1, le=120, description="Age in years")
    sex: int = Field(ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(ge=1, le=4, description="Chest pain type")
    trestbps: float = Field(ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(ge=0, le=2, description="Resting ECG result")
    thalach: float = Field(ge=40, le=260, description="Max heart rate achieved")
    exang: int = Field(ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: int = Field(ge=0, le=4, description="Major vessels coloured by fluoroscopy")
    thal: int = Field(description="Thalassemia: 3 normal, 6 fixed defect, 7 reversible")


class Prediction(BaseModel):
    prediction: int
    label: str
    probability_disease: float
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok", "model": state["meta"].get("model_name", "unknown")}


@app.post("/predict", response_model=Prediction)
def predict_endpoint(patient: Patient):
    result = predict(state["model"], [patient.model_dump()])[0]
    result["label"] = "disease" if result["prediction"] else "no disease"
    return result
