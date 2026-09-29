"""Cleaning and preprocessing helpers for the heart disease dataset."""
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/heart_disease_raw.csv")
CLEAN_PATH = Path("data/processed/heart_disease_clean.csv")

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang",
                        "slope", "ca", "thal"]
TARGET = "target"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values (median) and binarise the target.

    Original target is 0 (healthy) or 1-4 (disease severity); it is
    collapsed to 0/1 (absence/presence of heart disease).
    """
    df = df.copy().drop_duplicates()
    for col in df.columns.drop(TARGET):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    df[TARGET] = (df[TARGET] > 0).astype(int)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(int)
    return df.reset_index(drop=True)


def build_clean_dataset(raw: Path = RAW_PATH, out: Path = CLEAN_PATH) -> pd.DataFrame:
    df = clean(load_raw(raw))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    d = build_clean_dataset()
    print(f"Clean dataset: {d.shape} -> {CLEAN_PATH}")
