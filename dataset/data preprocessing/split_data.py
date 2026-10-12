
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "data"
    / "synthetic_customer_churn_100k.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed"

TARGET_COLUMN = "Churn"
RANDOM_STATE = 42


def split_data():
    """Chia dữ liệu thành train, validation và test theo tỷ lệ 80/10/10."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay file du lieu: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Khong tim thay cot muc tieu {TARGET_COLUMN}."
        )

    if df[TARGET_COLUMN].isnull().any():
        raise ValueError(
            "Cot Churn co gia tri thieu, khong the chia du lieu."
        )

    if not set(df[TARGET_COLUMN].unique()).issubset({"No", "Yes"}):
        raise ValueError(
            "Cot Churn co nhan khong hop le."
        )

    # Chia 80% train và 20% tạm thời
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    # Chia 20% tạm thời thành 10% validation và 10% test
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df[TARGET_COLUMN],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
        encoding="utf-8-sig",
    )
    valid_df.to_csv(
        OUTPUT_DIR / "valid.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 60)
    print("KET QUA CHIA DU LIEU")
    print("=" * 60)

    for name, subset in [
        ("Train", train_df),
        ("Validation", valid_df),
        ("Test", test_df),
    ]:
        print(f"\n{name}:")
        print(f"So dong: {len(subset):,}")
        print(f"Ty le so voi du lieu goc: {len(subset) / len(df):.0%}")
        print("Phan bo Churn (%):")
        print(
            (
                subset[TARGET_COLUMN]
                .value_counts(normalize=True)
                .mul(100)
                .round(2)
            ).to_string()
        )

    # Kiểm tra trùng CustomerID giữa các tập nếu cột này tồn tại
    if "CustomerID" in df.columns:
        train_ids = set(train_df["CustomerID"])
        valid_ids = set(valid_df["CustomerID"])
        test_ids = set(test_df["CustomerID"])

        assert train_ids.isdisjoint(valid_ids)
        assert train_ids.isdisjoint(test_ids)
        assert valid_ids.isdisjoint(test_ids)

    print("\nDa chia du lieu thanh cong!")
    print(f"Thu muc ket qua: {OUTPUT_DIR}")
    print("Luu y: Chua fit cac buoc tien xu ly tren toan bo du lieu.")


if __name__ == "__main__":
    split_data()
