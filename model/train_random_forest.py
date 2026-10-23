import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Xac dinh duong dan du an
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DATA_DIR = PROJECT_ROOT / "dataset" / "processed"
MODEL_DIR = PROJECT_ROOT / "saved models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CURRENT_DIR))
from tune_hyperparameters import DEFAULT_N_ITER, CV_FOLDS, to_jsonable, tune_random_forest

THRESHOLD = 0.5
MODEL_PATH = MODEL_DIR / "random_forest.joblib"
METADATA_PATH = MODEL_DIR / "random_forest.json"


def load_split(csv_path, schema):
    """Doc mot tap, kiem tra ten/thu tu dac trung va tra ve (X, y)."""
    target_column = schema["target_column"]
    feature_names = schema["processed_feature_names"]

    if schema.get("processed_feature_order") != feature_names:
        raise ValueError("feature_schema.json: processed_feature_names va processed_feature_order khong khop.")

    df = pd.read_csv(csv_path)

    missing = [c for c in feature_names + [target_column] if c not in df.columns]
    extra = [c for c in df.columns if c not in feature_names and c != target_column]
    if missing or extra:
        raise ValueError(f"{csv_path.name}: thieu cot {missing}, thua cot {extra}.")

    file_order = [c for c in df.columns if c != target_column]
    if file_order != feature_names:
        print(f"Canh bao: thu tu cot trong {csv_path.name} khac schema, se sap xep lai theo schema.")

    X = df[feature_names]
    if X.isna().any().any():
        raise ValueError(f"{csv_path.name}: dac trung con gia tri thieu.")
    if not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
        raise ValueError(f"{csv_path.name}: co dac trung khong phai kieu so.")

    y = df[target_column]
    if not pd.api.types.is_numeric_dtype(y):
        y = y.map(schema["target_mapping"])
    if y.isna().any() or not set(y.unique()) <= {0, 1}:
        raise ValueError(f"{csv_path.name}: nhan {target_column} khong hop le.")

    return X, y.astype(int)


def compute_metrics(y_true, y_prob, threshold=THRESHOLD):
    """Cac chi so tren tap valid; Precision/Recall/F1 tinh cho lop Churn=1."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def main():
    parser = argparse.ArgumentParser(description="Huan luyen Random Forest")
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    args = parser.parse_args()

    with open(DATA_DIR / "feature_schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)

    target_column = schema["target_column"]
    feature_names = schema["processed_feature_names"]

    X_train, y_train = load_split(DATA_DIR / "train.csv", schema)
    X_valid, y_valid = load_split(DATA_DIR / "valid.csv", schema)

    print(f"Train: {X_train.shape}, Valid: {X_valid.shape}")
    print(f"Ty le Churn=1 trong train: {y_train.mean():.4f}")

    # Toi uu sieu tham so bang kiem dinh cheo tren train
    print("Bat dau toi uu sieu tham so Random Forest...")
    model, best_params, search_info = tune_random_forest(
        X_train, y_train, search_type="random", n_iter=args.n_iter, cv_folds=args.cv_folds
    )

    if list(model.classes_) != [0, 1]:
        raise RuntimeError(f"Thu tu lop khong dung: {model.classes_}")

    # Kiem tra cau hinh da chon tren valid
    valid_prob = model.predict_proba(X_valid)[:, 1]
    metrics = compute_metrics(y_valid, valid_prob)

    print(f"Cau hinh tot nhat: {best_params}")
    print(f"CV F1 (train): {search_info['best_cv_f1']:.4f}")
    print(
        f"Valid: F1={metrics['f1']:.4f}, Recall={metrics['recall']:.4f}, "
        f"Precision={metrics['precision']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}"
    )

    # Luu mo hinh va metadata
    joblib.dump(model, MODEL_PATH, compress=3)

    metadata = {
        "algorithm": "Random Forest",
        "hyperparameters": best_params,
        "fixed_parameters": search_info["fixed_parameters"],
        "selected_threshold": THRESHOLD,
        "target_column": target_column,
        "positive_class": 1,
        "feature_names": feature_names,
        "selection_metrics": {
            "validation_f1": metrics["f1"],
            "validation_recall": metrics["recall"],
            "validation_roc_auc": metrics["roc_auc"],
        },
        "validation_metrics": metrics,
        "hyperparameter_search": search_info,
        "train_class_distribution": {
            "churn_0": int((y_train == 0).sum()),
            "churn_1": int((y_train == 1).sum()),
        },
        "libraries": {
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(metadata), f, ensure_ascii=False, indent=4)

    # Kiem tra: tai lai mo hinh phai cho ket qua giong truoc khi luu
    loaded = joblib.load(MODEL_PATH)
    same_prob = np.allclose(loaded.predict_proba(X_valid)[:, 1], valid_prob)
    same_label = np.array_equal(loaded.predict(X_valid), model.predict(X_valid))
    if not (same_prob and same_label):
        raise RuntimeError("Mo hinh tai lai cho ket qua khac truoc khi luu!")

    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print("\nHoan tat huan luyen Random Forest!")
    print(f"Mo hinh: {MODEL_PATH} ({size_mb:.1f} MB)")
    print(f"Metadata: {METADATA_PATH}")
    print("Kiem tra tai lai mo hinh: OK")
    if size_mb > 50:
        print("Canh bao: file lon, GitHub gioi han 100 MB/file. Hay giam n_estimators/max_depth.")


if __name__ == "__main__":
    main()
