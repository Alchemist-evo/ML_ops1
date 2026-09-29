import numpy as np
import pandas as pd
import pytest

from src.data_processing import TARGET


@pytest.fixture
def raw_df():
    """Small synthetic frame shaped like the raw UCI data (with a missing value)."""
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "age": rng.integers(30, 75, n).astype(float),
        "trestbps": rng.normal(130, 15, n),
        "chol": rng.normal(240, 40, n),
        "thalach": rng.normal(150, 20, n),
        "oldpeak": rng.uniform(0, 4, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "exang": rng.integers(0, 2, n),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3, 6, 7], n).astype(float),
        TARGET: rng.integers(0, 5, n),  # 0-4 severity as in the raw data
    })
    df.loc[3, "ca"] = np.nan
    df.loc[5, "thal"] = np.nan
    return df


@pytest.fixture
def clean_df(raw_df):
    from src.data_processing import clean
    return clean(raw_df)


@pytest.fixture
def sample_record():
    return {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
            "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3,
            "slope": 3, "ca": 0, "thal": 6}
