import pytest
from sklearn.model_selection import train_test_split

from src.features import FEATURES
from src.models import get_models
from src.predict import predict
from src.data_processing import TARGET
from src.train import evaluate_test


@pytest.fixture
def fitted(clean_df):
    X, y = clean_df[FEATURES], clean_df[TARGET]
    return train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest"])
def test_models_fit_and_predict_probabilities(name, fitted):
    X_tr, X_te, y_tr, _ = fitted
    pipe, _ = get_models()[name]
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)
    assert proba.shape == (len(X_te), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_search_spaces_reference_valid_params():
    for name, (pipe, grid) in get_models().items():
        valid = pipe.get_params().keys()
        assert all(k in valid for k in grid), name


def test_evaluate_test_returns_expected_metrics(fitted):
    X_tr, X_te, y_tr, y_te = fitted
    pipe, _ = get_models()["logistic_regression"]
    pipe.fit(X_tr, y_tr)
    m = evaluate_test(pipe, X_te, y_te)
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= v <= 1 for v in m.values())


def test_predict_output_format(fitted, sample_record):
    X_tr, _, y_tr, _ = fitted
    pipe, _ = get_models()["logistic_regression"]
    pipe.fit(X_tr, y_tr)
    out = predict(pipe, [sample_record])[0]
    assert out["prediction"] in (0, 1)
    assert 0.5 <= out["confidence"] <= 1
    assert 0 <= out["probability_disease"] <= 1
