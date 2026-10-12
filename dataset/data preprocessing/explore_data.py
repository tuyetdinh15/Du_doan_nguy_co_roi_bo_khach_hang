
from pathlib import Path

import matplotlib

# Dùng backend không cần mở cửa sổ biểu đồ
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Xác định thư mục gốc của project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "data"
    / "synthetic_customer_churn_100k.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "exploration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def explore_data():
    """Thống kê dữ liệu và lưu các biểu đồ khám phá."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay file du lieu: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "Churn" not in df.columns:
        raise ValueError("Du lieu khong co cot Churn.")

    print("=" * 60)
    print("1. THONG KE MO TA")
    print("=" * 60)

    # Thống kê các cột số
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    # Không đưa mã khách hàng vào thống kê đặc trưng
    numeric_features = [
        col for col in numeric_columns
        if col != "CustomerID"
    ]

    if numeric_features:
        numeric_summary = df[numeric_features].describe().T
        numeric_summary.to_csv(
            OUTPUT_DIR / "numeric_summary.csv",
            encoding="utf-8-sig",
        )
        print(numeric_summary)

    print("\n" + "=" * 60)
    print("2. PHAN BO NHAN CHURN")
    print("=" * 60)

    churn_counts = df["Churn"].value_counts(dropna=False)
    churn_percentages = (
        df["Churn"].value_counts(normalize=True, dropna=False) * 100
    )

    churn_summary = pd.DataFrame({
        "So_luong": churn_counts,
        "Ty_le_phan_tram": churn_percentages,
    })

    churn_summary.to_csv(
        OUTPUT_DIR / "churn_summary.csv",
        encoding="utf-8-sig",
    )
    print(churn_summary)

    # Biểu đồ phân bố Churn
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Churn", order=["No", "Yes"])
    plt.title("Phan bo nhan Churn")
    plt.xlabel("Churn")
    plt.ylabel("So luong khach hang")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "churn_distribution.png",
        dpi=150,
    )
    plt.close()

    print("\n" + "=" * 60)
    print("3. PHAN BO CAC BIEN SO")
    print("=" * 60)

    # Vẽ histogram cho các biến số, bỏ CustomerID
    for column in numeric_features:
        plt.figure(figsize=(7, 5))
        sns.histplot(data=df, x=column, bins=30, kde=True)
        plt.title(f"Phan bo {column}")
        plt.xlabel(column)
        plt.ylabel("Tan suat")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"distribution_{column}.png",
            dpi=150,
        )
        plt.close()

    # Thống kê tỷ lệ Churn theo từng biến phân loại
    print("\n" + "=" * 60)
    print("4. CHURN THEO CAC BIEN PHAN LOAI")
    print("=" * 60)

    categorical_columns = df.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    categorical_columns = [
        col for col in categorical_columns
        if col not in ["Churn", "CustomerID"]
    ]

    for column in categorical_columns:
        summary = pd.crosstab(
            df[column],
            df["Churn"],
            normalize="index",
        ) * 100

        summary.columns = [
            f"Churn_{label}_percent"
            for label in summary.columns
        ]

        summary.to_csv(
            OUTPUT_DIR / f"churn_by_{column}.csv",
            encoding="utf-8-sig",
        )

        print(f"\nTy le Churn theo {column} (%):")
        print(summary.round(2))

        # Biểu đồ tỷ lệ rời bỏ theo từng nhóm
        churn_rate = (
            df.assign(
                IsChurn=df["Churn"].map({"No": 0, "Yes": 1})
            )
            .groupby(column, observed=False)["IsChurn"]
            .mean()
            .mul(100)
            .reset_index(name="ChurnRate")
        )

        plt.figure(figsize=(9, 5))
        sns.barplot(
            data=churn_rate,
            x=column,
            y="ChurnRate",
        )
        plt.title(f"Ty le Churn theo {column}")
        plt.xlabel(column)
        plt.ylabel("Ty le Churn (%)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"churn_rate_by_{column}.png",
            dpi=150,
        )
        plt.close()

    print("\n" + "=" * 60)
    print("5. HOAN TAT")
    print("=" * 60)
    print(f"Ket qua da duoc luu tai: {OUTPUT_DIR}")


if __name__ == "__main__":
    explore_data()
