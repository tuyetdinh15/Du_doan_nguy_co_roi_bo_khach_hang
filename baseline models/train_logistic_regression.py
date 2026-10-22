
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import f1_score, recall_score, roc_auc_score

# Xác định đường dẫn dự án
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DATA_DIR = PROJECT_ROOT / "dataset" / "processed"
MODEL_DIR = PROJECT_ROOT / "saved models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Import hàm khởi tạo Logistic Regression
sys.path.insert(0, str(CURRENT_DIR))
from logistic_regression import create_model


def main():
    # Đọc dữ liệu đã tiền xử lý
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    valid_df = pd.read_csv(DATA_DIR / "valid.csv")

    with open(DATA_DIR / "feature_schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)

    target_column = schema["target_column"]
    feature_names = schema["processed_feature_names"]

    X_train = train_df[feature_names]
    y_train = train_df[target_column].replace(schema["target_mapping"])

    X_valid = valid_df[feature_names]
    y_valid = valid_df[target_column].replace(schema["target_mapping"])

    y_train = y_train.astype(int)
    y_valid = y_valid.astype(int)

    # Các cấu hình sẽ được so sánh
    candidates = [
        {"C": C, "class_weight": weight}
        for C in [0.01, 0.1, 1.0, 10.0]
        for weight in [None, "balanced"]
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model = None
    best_config = None
    best_score = None

    print("Bat dau Cross-Validation...")

    for config in candidates:
        model = create_model(**config)

        cv_result = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "f1": "f1",
                "recall": "recall",
                "roc_auc": "roc_auc",
            },
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        valid_prob = model.predict_proba(X_valid)[:, 1]
        valid_pred = (valid_prob >= 0.5).astype(int)

        f1 = f1_score(y_valid, valid_pred, zero_division=0)
        recall = recall_score(y_valid, valid_pred, zero_division=0)
        auc = roc_auc_score(y_valid, valid_prob)

        score = (
            f1,
            recall,
            auc,
        )

        print(
            f"C={config['C']}, "
            f"class_weight={config['class_weight']}, "
            f"CV F1={cv_result['test_f1'].mean():.4f}, "
            f"Valid F1={f1:.4f}, "
            f"Recall={recall:.4f}, "
            f"ROC-AUC={auc:.4f}"
        )

        if best_score is None or score > best_score:
            best_score = score
            best_config = config
            best_model = create_model(**config)

    # Huấn luyện mô hình cuối cùng trên tập train
    best_model.fit(X_train, y_train)

    model_path = MODEL_DIR / "logistic_regression.joblib"
    metadata_path = MODEL_DIR / "logistic_regression.json"

    joblib.dump(best_model, model_path)

    metadata = {
        "algorithm": "Logistic Regression",
        "hyperparameters": best_config,
        "selected_threshold": 0.5,
        "target_column": target_column,
        "positive_class": 1,
        "feature_names": feature_names,
        "selection_metrics": {
            "validation_f1": best_score[0],
            "validation_recall": best_score[1],
            "validation_roc_auc": best_score[2],
        },
        "libraries": {
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print("\nHoan tat huan luyen!")
    print(f"Mo hinh: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Cau hinh tot nhat: {best_config}")
    print(f"Validation F1: {best_score[0]:.4f}")


if __name__ == "__main__":
    main()
