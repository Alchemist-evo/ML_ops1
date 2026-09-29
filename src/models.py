"""Model definitions and hyper-parameter search spaces."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor

RANDOM_STATE = 42


def get_models() -> dict:
    """name -> (pipeline, param_grid). Grid keys use the `clf__` prefix."""
    def make(clf):
        return Pipeline([("prep", build_preprocessor()), ("clf", clf)])

    return {
        "logistic_regression": (
            make(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            {"clf__C": [0.01, 0.1, 1, 10],
             "clf__class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            make(RandomForestClassifier(random_state=RANDOM_STATE)),
            {"clf__n_estimators": [100, 300],
             "clf__max_depth": [3, 5, None],
             "clf__min_samples_leaf": [1, 3, 5]},
        ),
    }
