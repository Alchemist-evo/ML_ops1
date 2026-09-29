"""MLflow helpers: plots and run logging."""
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay  # noqa: E402

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "heart-disease-classification"


def setup():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)


def _save_fig(fig, directory: Path, name: str) -> Path:
    path = directory / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_plots(model, X_test, y_test, directory: Path) -> list:
    paths = []
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC curve (hold-out test)")
    paths.append(_save_fig(fig, directory, "roc_curve.png"))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax,
                                          display_labels=["No disease", "Disease"])
    ax.set_title("Confusion matrix (hold-out test)")
    paths.append(_save_fig(fig, directory, "confusion_matrix.png"))

    clf = model.named_steps["clf"]
    names = model.named_steps["prep"].get_feature_names_out()
    if hasattr(clf, "feature_importances_"):
        values, title = clf.feature_importances_, "Feature importance"
    else:
        values, title = clf.coef_[0], "Logistic regression coefficients"
    order = np.argsort(np.abs(values))[-15:]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(names)[order], values[order])
    ax.set_title(title)
    paths.append(_save_fig(fig, directory, "feature_importance.png"))
    return paths


def log_search_children(search_results: dict) -> None:
    """One nested run per grid-search candidate."""
    n = len(search_results["params"])
    for i in range(n):
        with mlflow.start_run(run_name=f"candidate_{i}", nested=True):
            mlflow.log_params({k: str(v) for k, v in search_results["params"][i].items()})
            mlflow.log_metric("cv_roc_auc_mean", search_results["mean_test_score"][i])
            mlflow.log_metric("cv_roc_auc_std", search_results["std_test_score"][i])


def log_model_run(result: dict, search, X_train, X_test, y_test, data_path: str) -> str:
    """Log one tuned model (params, metrics, artifacts, plots, model)."""
    model = result["model"]
    with mlflow.start_run(run_name=result["name"]) as run:
        mlflow.set_tag("model_type", result["name"])
        mlflow.log_params({k.replace("clf__", ""): str(v)
                           for k, v in result["best_params"].items()})
        mlflow.log_params({"cv_folds": 5, "test_size": 0.2,
                           "n_train": len(X_train), "n_test": len(X_test),
                           "features": ",".join(X_train.columns)})
        for k, v in result["cv_metrics"].items():
            mlflow.log_metric(f"cv_{k}_mean", v)
            mlflow.log_metric(f"cv_{k}_std", result["cv_std"][k])
        for k, v in result["test_metrics"].items():
            mlflow.log_metric(f"test_{k}", v)

        with tempfile.TemporaryDirectory() as tmp:
            for p in make_plots(model, X_test, y_test, Path(tmp)):
                mlflow.log_artifact(str(p), artifact_path="plots")
        mlflow.log_artifact(data_path, artifact_path="data")
        mlflow.log_artifact("requirements.txt")
        mlflow.sklearn.log_model(model, name="model",
                                 serialization_format="cloudpickle",
                                 input_example=X_test.head(3).astype("float64"))
        log_search_children(search.cv_results_)
        return run.info.run_id
