import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.data_processing import TARGET
from src.features import FEATURES
from src.models import get_models
import joblib


@pytest.fixture
def client(tmp_path, clean_df, monkeypatch):
    pipe, _ = get_models()["logistic_regression"]
    pipe.fit(clean_df[FEATURES], clean_df[TARGET])
    path = tmp_path / "model.joblib"
    joblib.dump(pipe, path)
    monkeypatch.setattr("src.api.MODEL_PATH", path)
    monkeypatch.setattr("src.api.META_PATH", tmp_path / "missing.json")
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_prediction_and_confidence(client, sample_record):
    r = client.post("/predict", json=sample_record)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("disease", "no disease")
    assert 0.5 <= body["confidence"] <= 1


def test_predict_rejects_missing_field(client, sample_record):
    sample_record.pop("age")
    assert client.post("/predict", json=sample_record).status_code == 422


def test_predict_rejects_out_of_range_value(client, sample_record):
    sample_record["sex"] = 5
    assert client.post("/predict", json=sample_record).status_code == 422


def test_metrics_endpoint_counts_requests_and_predictions(client, sample_record):
    client.post("/predict", json=sample_record)
    client.post("/predict", json={"age": 1})  # 422
    text = client.get("/metrics").text
    assert 'api_requests_total{endpoint="/predict",method="POST",status="200"}' in text
    assert 'api_requests_total{endpoint="/predict",method="POST",status="422"}' in text
    assert "model_predictions_total" in text
    assert "api_request_latency_seconds_bucket" in text


def test_metrics_endpoint_is_not_self_counted(client):
    client.get("/metrics")
    assert 'endpoint="/metrics"' not in client.get("/metrics").text


def test_request_is_logged(client, sample_record, caplog):
    with caplog.at_level("INFO", logger="heart_api"):
        client.post("/predict", json=sample_record)
    assert any("path=/predict status=200" in r.message for r in caplog.records)
