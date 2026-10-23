import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = "f1"
DEFAULT_N_ITER = 20

RANDOM_FOREST_SEARCH_SPACE = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, 10, 12],
    "min_samples_split": [10, 20, 50],
    "min_samples_leaf": [5, 10, 20],
    "max_features": ["sqrt", 0.5],
}

XGBOOST_SEARCH_SPACE = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}


def to_jsonable(obj):
    """Doi kieu numpy sang kieu Python de ghi JSON."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def run_search(
    estimator,
    search_space,
    X,
    y,
    search_type="random",
    n_iter=DEFAULT_N_ITER,
    cv_folds=CV_FOLDS,
    random_state=RANDOM_STATE,
    scoring=SCORING,
    n_jobs=-1,
    verbose=1,
):
    """Chay GridSearchCV hoac RandomizedSearchCV tren train.

    Tra ve (best_estimator da refit tren toan bo train, best_params, thong tin tim kiem).
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    if search_type == "grid":
        search = GridSearchCV(
            estimator,
            param_grid=search_space,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            verbose=verbose,
        )
    elif search_type == "random":
        search = RandomizedSearchCV(
            estimator,
            param_distributions=search_space,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            random_state=random_state,
            verbose=verbose,
        )
    else:
        raise ValueError("search_type phai la 'grid' hoac 'random'.")

    start = time.time()
    search.fit(X, y)
    elapsed = time.time() - start

    best_index = search.best_index_
    info = {
        "search_method": "GridSearchCV" if search_type == "grid" else "RandomizedSearchCV",
        "scoring": f"{scoring} (lop Churn=1)",
        "cv": f"StratifiedKFold(n_splits={cv_folds}, shuffle=True)",
        "cv_folds": cv_folds,
        "random_state": random_state,
        "n_iter": n_iter if search_type == "random" else None,
        "n_candidates_evaluated": int(len(search.cv_results_["params"])),
        "search_space": search_space,
        "best_cv_f1": float(search.best_score_),
        "best_cv_f1_std": float(search.cv_results_["std_test_score"][best_index]),
        "search_seconds": round(elapsed, 1),
    }
    return search.best_estimator_, search.best_params_, to_jsonable(info)


def tune_random_forest(X, y, search_type="random", n_iter=DEFAULT_N_ITER, cv_folds=CV_FOLDS):
    """Toi uu Random Forest. Trong so lop 'balanced' nam trong cau hinh mac dinh."""
    from random_forest import create_model

    estimator = create_model()
    best_model, best_params, info = run_search(
        estimator,
        RANDOM_FOREST_SEARCH_SPACE,
        X,
        y,
        search_type=search_type,
        n_iter=n_iter,
        cv_folds=cv_folds,
    )
    info["fixed_parameters"] = {"class_weight": "balanced", "random_state": RANDOM_STATE}
    return best_model, best_params, info


def tune_xgboost(X, y, search_type="random", n_iter=DEFAULT_N_ITER, cv_folds=CV_FOLDS):
    """Toi uu XGBoost. scale_pos_weight tinh tu nhan train."""
    from xgboost_model import compute_scale_pos_weight, create_model

    scale_pos_weight = compute_scale_pos_weight(y)
    estimator = create_model(scale_pos_weight=scale_pos_weight)
    best_model, best_params, info = run_search(
        estimator,
        XGBOOST_SEARCH_SPACE,
        X,
        y,
        search_type=search_type,
        n_iter=n_iter,
        cv_folds=cv_folds,
    )
    info["fixed_parameters"] = {
        "scale_pos_weight": float(scale_pos_weight),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
    }
    return best_model, best_params, info