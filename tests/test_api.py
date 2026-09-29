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
