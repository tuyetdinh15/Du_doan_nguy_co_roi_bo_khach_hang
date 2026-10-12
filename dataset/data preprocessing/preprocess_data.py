
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "dataset" / "processed"
OUTPUT_DIR = DATA_DIR

TARGET_COLUMN = "Churn"
ID_COLUMN = "CustomerID"

SET_FILES = {
    "train": DATA_DIR / "train.csv",
    "valid": DATA_DIR / "valid.csv",
    "test": DATA_DIR / "test.csv",
}


def load_splits():
    """Đọc ba tập dữ liệu đã được chia."""

    datasets = {}

    for name, path in SET_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Khong tim thay {path}. "
                "Hay chay split_data.py truoc."
            )

        datasets[name] = pd.read_csv(path)

    return datasets


def prepare_raw_features(df):
    """Tách nhãn và loại bỏ ID khỏi đặc trưng."""

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Thieu cot {TARGET_COLUMN}.")

    if df[TARGET_COLUMN].isnull().any():
        raise ValueError("Cot Churn co gia tri thieu.")

    # Chỉ chấp nhận hai nhãn theo quy ước của dự án
    target = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    if target.isnull().any():
        raise ValueError("Cot Churn co nhan khong hop le.")

    # Không dùng ID làm đặc trưng dự đoán
    X = df.drop(
        columns=[TARGET_COLUMN, ID_COLUMN],
        errors="ignore",
    ).copy()

    # Chuyển các cột số về kiểu số; giá trị không hợp lệ thành NaN
    expected_numeric = [
        "Age",
        "Tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    for column in expected_numeric:
        if column in X.columns:
            X[column] = pd.to_numeric(
                X[column],
                errors="coerce",
            )

    return X, target.astype("int64")


def build_preprocessor(X_train):
    """Tạo pipeline riêng cho biến số và biến phân loại."""

    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "str", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    transformers = []

    if numeric_columns:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise ValueError("Khong tim thay cot dac trung hop le.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor, numeric_columns, categorical_columns


def preprocess_data():
    """Fit trên train, transform validation/test và lưu kết quả."""

    datasets = load_splits()

    prepared = {
        name: prepare_raw_features(df)
        for name, df in datasets.items()
    }

    X_train, y_train = prepared["train"]
    X_valid, y_valid = prepared["valid"]
    X_test, y_test = prepared["test"]

    # Các tập phải có cùng tên và thứ tự cột đầu vào
    if not (
        list(X_train.columns)
        == list(X_valid.columns)
        == list(X_test.columns)
    ):
        raise ValueError(
            "Cac tap khong co cung ten va thu tu cot."
        )

    # Fit duy nhất trên train để tránh rò rỉ dữ liệu
    preprocessor, numeric_columns, categorical_columns = (
        build_preprocessor(X_train)
    )

    train_array = preprocessor.fit_transform(X_train)
    valid_array = preprocessor.transform(X_valid)
    test_array = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out().tolist()

    if not (
        train_array.shape[1]
        == valid_array.shape[1]
        == test_array.shape[1]
        == len(feature_names)
    ):
        raise ValueError(
            "So luong dac trung sau xu ly khong dong nhat."
        )

    # Đưa về DataFrame để giữ tên và thứ tự đặc trưng
    processed_sets = {
        "train": pd.DataFrame(
            train_array, columns=feature_names
        ),
        "valid": pd.DataFrame(
            valid_array, columns=feature_names
        ),
        "test": pd.DataFrame(
            test_array, columns=feature_names
        ),
    }

    labels = {
        "train": y_train,
        "valid": y_valid,
        "test": y_test,
    }

    for name in processed_sets:
        result = processed_sets[name].copy()
        result[TARGET_COLUMN] = labels[name].to_numpy()

        # Kiểm tra dữ liệu sau xử lý
        if result.isnull().any().any():
            raise ValueError(
                f"Tập {name} van con gia tri thieu."
            )

        if not np.isfinite(
            result.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Tập {name} co gia tri vo han."
            )

        result.to_csv(
            OUTPUT_DIR / f"{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"{name}: {result.shape[0]:,} dong, "
            f"{result.shape[1] - 1} dac trung + nhan Churn"
        )

    # Lưu bộ tiền xử lý để các thành viên khác tái sử dụng
    joblib.dump(
        preprocessor,
        OUTPUT_DIR / "preprocessor.joblib",
    )

    # Mô tả cấu trúc dữ liệu đầu vào và đầu ra
    schema = {
        "target_column": TARGET_COLUMN,
        "target_mapping": {
            "No": 0,
            "Yes": 1,
        },
        "id_column_excluded": ID_COLUMN,
        "raw_input_columns": list(X_train.columns),
        "raw_input_types": {
            col: str(dtype)
            for col, dtype in X_train.dtypes.items()
        },
        "valid_values": {
            "Gender": ["Female", "Male", "Other"],
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
            ],
            "PaymentMethod": [
                "Bank transfer",
                "Credit card",
                "Electronic check",
                "Mailed check",
            ],
        },
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "processed_feature_names": feature_names,
        "processed_feature_order": feature_names,
        "fit_dataset": "train",
        "random_state": 42,
        "preprocessing": {
            "numeric_imputation": "median",
            "numeric_scaling": "StandardScaler",
            "categorical_imputation": "most_frequent",
            "categorical_encoding": "OneHotEncoder",
            "unknown_categories": "ignore",
            "target_encoding": {
                "No": 0,
                "Yes": 1,
            },
        },
    }


    with open(
        OUTPUT_DIR / "feature_schema.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            schema,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\nDa tien xu ly du lieu thanh cong!")
    print(f"Ket qua: {OUTPUT_DIR}")
    print("Da luu preprocessor.joblib va feature_schema.json.")


if __name__ == "__main__":
    preprocess_data()
