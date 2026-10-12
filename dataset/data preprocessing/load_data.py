
from pathlib import Path

import pandas as pd


# Lấy đường dẫn thư mục gốc của project.
# load_data.py nằm trong dataset/data preprocessing/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Đường dẫn đến file dữ liệu gốc
DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "data"
    / "synthetic_customer_churn_100k.csv"
)

# Các cột cần có theo bộ dữ liệu của project
REQUIRED_COLUMNS = [
    "CustomerID",
    "Age",
    "Gender",
    "Tenure",
    "MonthlyCharges",
    "Contract",
    "PaymentMethod",
    "TotalCharges",
    "Churn",
]

VALID_CHURN_LABELS = {"No", "Yes"}


def load_data(file_path=DATA_PATH):
    """Đọc dữ liệu CSV và kiểm tra cấu trúc cơ bản."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file du lieu: {file_path}"
        )

    df = pd.read_csv(file_path)

    print("=" * 60)
    print("1. THONG TIN TONG QUAN")
    print("=" * 60)
    print(f"Duong dan: {file_path}")
    print(f"So dong: {df.shape[0]:,}")
    print(f"So cot: {df.shape[1]}")
    print("\nDanh sach cot:")
    print(df.columns.tolist())

    # Kiểm tra các cột bắt buộc
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Thieu cac cot bat buoc: {missing_columns}"
        )

    print("\n" + "=" * 60)
    print("2. KIEM TRA KIEU DU LIEU")
    print("=" * 60)
    print(df.dtypes)

    # Kiểm tra dữ liệu thiếu
    print("\n" + "=" * 60)
    print("3. KIEM TRA GIA TRI THIEU")
    print("=" * 60)
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0]
          if missing_values.any()
          else "Khong co gia tri thieu.")

    # Kiểm tra bản ghi trùng hoàn toàn
    duplicate_count = df.duplicated().sum()
    print("\n" + "=" * 60)
    print("4. KIEM TRA BAN GHI TRUNG")
    print("=" * 60)
    print(f"So ban ghi trung hoan toan: {duplicate_count}")

    # Kiểm tra mã khách hàng
    print("\n" + "=" * 60)
    print("5. KIEM TRA CUSTOMER ID")
    print("=" * 60)
    print(f"So CustomerID duy nhat: {df['CustomerID'].nunique()}")
    print(
        f"So CustomerID bi thieu: "
        f"{df['CustomerID'].isnull().sum()}"
    )

    # Kiểm tra nhãn mục tiêu Churn
    print("\n" + "=" * 60)
    print("6. KIEM TRA NHAN CHURN")
    print("=" * 60)
    print("Phan bo nhan Churn:")
    print(df["Churn"].value_counts(dropna=False))

    invalid_labels = set(df["Churn"].dropna().unique()) - VALID_CHURN_LABELS

    if invalid_labels:
        raise ValueError(
            f"Nhan Churn khong hop le: {invalid_labels}"
        )

    if df["Churn"].isnull().any():
        raise ValueError("Cot Churn co gia tri thieu.")

    print("\nDa doc va kiem tra du lieu thanh cong!")
    print("Luu y: Du lieu goc chua bi thay doi.")

    return df


if __name__ == "__main__":
    load_data()
