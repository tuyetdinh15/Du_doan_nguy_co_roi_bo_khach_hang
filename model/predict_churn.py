import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DATA_DIR = PROJECT_ROOT / "dataset" / "processed"
MODEL_DIR = PROJECT_ROOT / "saved models"

AVAILABLE_MODELS = ("logistic_regression", "random_forest", "xgboost")
DEFAULT_THRESHOLD = 0.5

_cache = {}


def load_model(model_name):
    """Tai (mo hinh, metadata). Ket qua duoc luu tam de khong tai lai nhieu lan."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Ten mo hinh khong hop le: {model_name}. Chon mot trong {AVAILABLE_MODELS}.")

    if model_name in _cache:
        return _cache[model_name]

    model_path = MODEL_DIR / f"{model_name}.joblib"
    metadata_path = MODEL_DIR / f"{model_name}.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Khong tim thay mo hinh: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Khong tim thay metadata: {metadata_path}")

    model = joblib.load(model_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    _cache[model_name] = (model, metadata)
    return model, metadata


def _check_features(X, metadata):
    """Kiem tra ten dac trung va sap xep dung thu tu luc huan luyen."""
    feature_names = metadata["feature_names"]
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X phai la pandas DataFrame voi ten cot dac trung.")

    missing = [c for c in feature_names if c not in X.columns]
    if missing:
        raise ValueError(f"Thieu dac trung: {missing}")

    X = X[feature_names]
    if X.isna().any().any():
        raise ValueError("Dac trung con gia tri thieu.")
    return X


def predict_proba_churn(model_name, X):
    """Xac suat khach hang roi bo (lop Churn=1), mang 1 chieu."""
    model, metadata = load_model(model_name)
    X = _check_features(X, metadata)

    classes = list(model.classes_)
    if 1 not in classes:
        raise RuntimeError("Mo hinh khong co lop 1.")
    return model.predict_proba(X)[:, classes.index(1)]


def predict_churn(model_name, X, threshold=None):
    """Tra ve DataFrame gom churn_probability va churn_prediction (0/1).

    Neu khong truyen threshold, dung selected_threshold trong metadata (mac dinh 0.5).
    """
    _, metadata = load_model(model_name)
    if threshold is None:
        threshold = metadata.get("selected_threshold", DEFAULT_THRESHOLD)

    prob = predict_proba_churn(model_name, X)
    return pd.DataFrame(
        {
            "churn_probability": prob,
            "churn_prediction": (prob >= threshold).astype(int),
        },
        index=X.index,
    )


def main():
    parser = argparse.ArgumentParser(description="Du doan thu tren vai dong dau cua valid.csv")
    parser.add_argument("--model", default="random_forest", choices=AVAILABLE_MODELS)
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()

    _, metadata = load_model(args.model)
    valid_df = pd.read_csv(DATA_DIR / "valid.csv")
    X = valid_df[metadata["feature_names"]].head(args.rows)

    result = predict_churn(args.model, X)
    result[metadata["target_column"]] = valid_df[metadata["target_column"]].head(args.rows).values

    print(f"Mo hinh: {metadata['algorithm']} (nguong {metadata['selected_threshold']})")
    print(result.to_string())


if __name__ == "__main__":
    main()