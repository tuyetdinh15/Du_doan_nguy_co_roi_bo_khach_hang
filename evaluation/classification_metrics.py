"""Cach tinh chi so thong nhat va ham doc du lieu/mo hinh dung chung cho phan danh gia.

File nay chi doc du lieu va mo hinh da luu; khong huan luyen, khong sua du lieu.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DATA_DIR = PROJECT_ROOT / "dataset" / "processed"
MODEL_DIR = PROJECT_ROOT / "saved models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Ten mo hinh (trung ten file trong saved models/) va ten hien thi thong nhat
MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost")
DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

# Ten lop dung thong nhat cho bao cao va bieu do (0 = khong roi bo, 1 = roi bo)
CLASS_LABELS = ("Không rời bỏ", "Rời bỏ")
POSITIVE_CLASS = 1
THRESHOLD = 0.5
DECIMALS = 4  # lam tron thong nhat cho bang, bao cao va bieu do

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")

# Tieu chi chon mo hinh: F1 lop Churn=1, hoa thi Recall, sau do ROC-AUC
SELECTION_ORDER = ("f1", "recall", "roc_auc")

# Dung lai chuc nang tai mo hinh/du doan do phan mo hinh ban giao
sys.path.insert(0, str(PROJECT_ROOT / "model"))
from predict_churn import load_model, predict_proba_churn  # noqa: E402


def load_schema():
    with open(DATA_DIR / "feature_schema.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_split(split_name, schema):
    """Doc train/valid/test, kiem tra ten + thu tu dac trung, tra ve (X, y)."""
    csv_path = DATA_DIR / f"{split_name}.csv"
    target = schema["target_column"]
    feature_names = schema["processed_feature_names"]

    df = pd.read_csv(csv_path)
    missing = [c for c in feature_names + [target] if c not in df.columns]
    extra = [c for c in df.columns if c not in feature_names and c != target]
    if missing or extra:
        raise ValueError(f"{csv_path.name}: thieu cot {missing}, thua cot {extra}.")
    if [c for c in df.columns if c != target] != feature_names:
        raise ValueError(f"{csv_path.name}: thu tu dac trung khac feature_schema.json.")

    X = df[feature_names]
    if X.isna().any().any():
        raise ValueError(f"{csv_path.name}: dac trung con gia tri thieu.")

    y = df[target]
    if not pd.api.types.is_numeric_dtype(y):
        y = y.map(schema["target_mapping"])
    if y.isna().any() or not set(y.unique()) <= {0, 1}:
        raise ValueError(f"{csv_path.name}: nhan {target} khong hop le.")
    return X, y.astype(int)


def check_models(schema, model_names=MODEL_NAMES):
    """Kiem tra ba mo hinh dung cung dac trung, nhan va nguong phan loai.

    Tra ve {ten_mo_hinh: metadata}. Bao loi ro rang neu thieu file hoac khong dong nhat.
    """
    metadata_all = {}
    for name in model_names:
        for ext in (".joblib", ".json"):
            path = MODEL_DIR / f"{name}{ext}"
            if not path.exists():
                raise FileNotFoundError(
                    f"Thieu {path.name} trong saved models/. "
                    "Can du ba mo hinh (Logistic Regression, Random Forest, XGBoost) truoc khi danh gia."
                )
        model, meta = load_model(name)
        if list(model.classes_) != [0, 1]:
            raise RuntimeError(f"{name}: thu tu lop khong dung {list(model.classes_)}.")
        if meta["feature_names"] != schema["processed_feature_names"]:
            raise ValueError(f"{name}: ten/thu tu dac trung trong metadata khac feature_schema.json.")
        if meta["target_column"] != schema["target_column"]:
            raise ValueError(f"{name}: cot nhan trong metadata khac feature_schema.json.")
        if meta.get("positive_class") != POSITIVE_CLASS:
            raise ValueError(f"{name}: positive_class phai la {POSITIVE_CLASS}.")
        if float(meta.get("selected_threshold", THRESHOLD)) != THRESHOLD:
            raise ValueError(f"{name}: nguong {meta.get('selected_threshold')} khac nguong thong nhat {THRESHOLD}.")
        metadata_all[name] = meta
    return metadata_all


def predict_probabilities(model_name, X):
    """Xac suat lop Churn=1 cua mo hinh da luu (dung chuc nang cua model/predict_churn.py)."""
    return np.asarray(predict_proba_churn(model_name, X))


def compute_metrics(y_true, y_prob, threshold=THRESHOLD):
    """Accuracy, Precision, Recall, F1 (lop Churn=1) tu nhan du doan theo nguong;
    ROC-AUC tinh tu xac suat (khong dung nhan du doan)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def compute_confusion(y_true, y_prob, threshold=THRESHOLD):
    """Ma tran nham lan [[TN, FP], [FN, TP]] theo nguong."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return confusion_matrix(np.asarray(y_true), y_pred, labels=[0, 1])


def selection_key(metrics):
    """Khoa so sanh mo hinh: (F1, Recall, ROC-AUC) da lam tron cung so le voi bao cao."""
    return tuple(round(metrics[k], DECIMALS) for k in SELECTION_ORDER)
