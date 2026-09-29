"""Train, tune and evaluate models.

Usage: python -m src.train
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)

from src.data_processing import CLEAN_PATH, TARGET
from src.features import FEATURES
from src.models import RANDOM_STATE, get_models
from src import tracking

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def load_data(path=CLEAN_PATH):
    df = pd.read_csv(path)
    return df[FEATURES], df[TARGET]


def split(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=RANDOM_STATE)


def evaluate_test(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def tune_and_evaluate(name, pipeline, grid, X_train, X_test, y_train, y_test):
    """Grid-search (CV on train only), then cross-validate and test the best."""
    search = GridSearchCV(pipeline, grid, cv=CV, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_

    cv = cross_validate(best, X_train, y_train, cv=CV, scoring=SCORING)
    cv_metrics = {m: cv[f"test_{m}"].mean() for m in SCORING}
    cv_std = {m: cv[f"test_{m}"].std() for m in SCORING}
    return {
        "name": name,
        "best_params": search.best_params_,
        "best_cv_auc_search": search.best_score_,
        "cv_metrics": cv_metrics,
        "cv_std": cv_std,
        "test_metrics": evaluate_test(best, X_test, y_test),
        "model": best,
        "search": search,
    }


MODEL_DIR = Path("models")


def save_best(results):
    """Persist the model with the highest CV ROC-AUC + a metadata file."""
    best = max(results, key=lambda r: r["cv_metrics"]["roc_auc"])
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best["model"], MODEL_DIR / "model.joblib")
    meta = {
        "model_name": best["name"],
        "best_params": {k: str(v) for k, v in best["best_params"].items()},
        "features": FEATURES,
        "cv_metrics": best["cv_metrics"],
        "test_metrics": best["test_metrics"],
        "mlflow_run_id": best["run_id"],
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved best model '{best['name']}' to {MODEL_DIR / 'model.joblib'}")
    return best


def main():
    tracking.setup()
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)
    results = []
    for name, (pipe, grid) in get_models().items():
        r = tune_and_evaluate(name, pipe, grid, X_train, X_test, y_train, y_test)
        r["run_id"] = tracking.log_model_run(r, r["search"], X_train, X_test,
                                             y_test, str(CLEAN_PATH))
        results.append(r)
        print(f"\n=== {name} ===")
        print("best params:", r["best_params"])
        print("CV (5-fold, train):", {k: f"{v:.3f}±{r['cv_std'][k]:.3f}"
                                      for k, v in r["cv_metrics"].items()})
        print("Hold-out test    :", {k: round(v, 3) for k, v in r["test_metrics"].items()})
        print("MLflow run id    :", r["run_id"])
    save_best(results)
    return results


if __name__ == "__main__":
    main()
