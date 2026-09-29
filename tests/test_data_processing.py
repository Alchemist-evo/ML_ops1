import pandas as pd

from src.data_processing import (CATEGORICAL_FEATURES, TARGET, clean,
                                 load_raw)


def test_clean_removes_missing_values(raw_df):
    assert raw_df.isna().sum().sum() > 0
    assert clean(raw_df).isna().sum().sum() == 0


def test_clean_binarises_target(raw_df):
    out = clean(raw_df)
    assert set(out[TARGET].unique()) <= {0, 1}
    assert out[TARGET].sum() == (raw_df.drop_duplicates()[TARGET] > 0).sum()


def test_clean_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    clean(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_clean_categoricals_are_integers(raw_df):
    out = clean(raw_df)
    for col in CATEGORICAL_FEATURES:
        assert pd.api.types.is_integer_dtype(out[col])


def test_clean_drops_duplicates(raw_df):
    doubled = pd.concat([raw_df, raw_df], ignore_index=True)
    assert len(clean(doubled)) == len(clean(raw_df))


def test_load_raw_treats_question_mark_as_nan(tmp_path):
    f = tmp_path / "raw.csv"
    f.write_text("a,b\n1,?\n2,3\n")
    df = load_raw(f)
    assert df["b"].isna().sum() == 1
