"""FastAPI service exposing the heart disease model.

Run: uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel, ConfigDict, Field

from src.predict import load_model, predict

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
META_PATH = MODEL_PATH.with_name("model_metadata.json")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("heart_api")

REQUESTS = Counter("api_requests_total", "HTTP requests",
                   ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds",
                    ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("model_predictions_total", "Predictions by class", ["label"])
CONFIDENCE = Histogram("model_prediction_confidence", "Confidence of predictions",
                       buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0))

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load_model(MODEL_PATH)
    state["meta"] = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    yield
    state.clear()


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def monitor(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        # use the route template (not raw path) to keep label cardinality low
        route = request.scope.get("route")
        endpoint = route.path if route else "unmatched"
        if endpoint != "/metrics":
            REQUESTS.labels(request.method, endpoint, str(status)).inc()
            LATENCY.labels(endpoint).observe(elapsed)
            logger.info("method=%s path=%s status=%s duration_ms=%.1f",
                        request.method, request.url.path, status, elapsed * 1000)


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


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model": state["meta"].get("model_name", "unknown")}


@app.post("/predict", response_model=Prediction)
def predict_endpoint(patient: Patient):
    result = predict(state["model"], [patient.model_dump()])[0]
    result["label"] = "disease" if result["prediction"] else "no disease"
    PREDICTIONS.labels(result["label"]).inc()
    CONFIDENCE.observe(result["confidence"])
    logger.info("prediction=%s probability=%.3f confidence=%.3f",
                result["label"], result["probability_disease"], result["confidence"])
    return result


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
