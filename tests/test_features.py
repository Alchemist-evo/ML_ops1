import numpy as np
import pandas as pd

from src.features import FEATURES, build_preprocessor


def test_preprocessor_output_has_no_nans(clean_df):
    X = clean_df[FEATURES]
    out = build_preprocessor().fit_transform(X)
    assert not np.isnan(out).any()


def test_numeric_features_are_standardised(clean_df):
    prep = build_preprocessor().fit(clean_df[FEATURES])
    out = prep.transform(clean_df[FEATURES])
    n_num = 5  # numeric block comes first
    assert np.allclose(out[:, :n_num].mean(axis=0), 0, atol=1e-6)
    assert np.allclose(out[:, :n_num].std(axis=0), 1, atol=1e-6)


def test_unseen_category_does_not_crash(clean_df):
    prep = build_preprocessor().fit(clean_df[FEATURES])
    row = clean_df[FEATURES].head(1).copy()
    row["cp"] = 99  # never seen in training
    assert prep.transform(row).shape[0] == 1


def test_preprocessor_handles_missing_at_inference(clean_df):
    prep = build_preprocessor().fit(clean_df[FEATURES])
    row = clean_df[FEATURES].head(1).astype("float64")
    row["chol"] = np.nan
    assert not pd.isna(prep.transform(row)).any()
