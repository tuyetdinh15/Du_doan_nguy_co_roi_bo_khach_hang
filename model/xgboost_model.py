from xgboost import XGBClassifier

RANDOM_STATE = 42

DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "scale_pos_weight": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}


def compute_scale_pos_weight(y):
    """Trong so lop duong = so mau lop 0 / so mau lop 1 (tinh tu nhan train)."""
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0:
        raise ValueError("Nhan train khong co mau Churn=1.")
    return negatives / positives


def create_model(**params):
    """Tao XGBoost. Tham so truyen vao se ghi de cau hinh mac dinh."""
    config = {**DEFAULT_PARAMS, **params}
    return XGBClassifier(**config)