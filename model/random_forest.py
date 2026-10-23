
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}


def create_model(**params):
    """Tao Random Forest. Tham so truyen vao se ghi de cau hinh mac dinh."""
    config = {**DEFAULT_PARAMS, **params}
    return RandomForestClassifier(**config)