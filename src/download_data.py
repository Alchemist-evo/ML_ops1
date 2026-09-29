"""Download the UCI Heart Disease (Cleveland) dataset.

Usage: python -m src.download_data
"""
from pathlib import Path

import requests

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.cleveland.data")
RAW_PATH = Path("data/raw/heart_disease_raw.csv")

COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]


def download(dest: Path = RAW_PATH, url: str = URL) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_text(",".join(COLUMNS) + "\n" + resp.text)
    return dest


if __name__ == "__main__":
    print(f"Saved raw data to {download()}")
