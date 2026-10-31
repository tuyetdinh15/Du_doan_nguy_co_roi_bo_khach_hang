"""Ve ma tran nham lan va duong ROC cho ba mo hinh tren tap test.

Co the chay rieng: python evaluation/plot_results.py (chi tai mo hinh da luu, khong huan luyen).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from classification_metrics import (
    CLASS_LABELS,
    DECIMALS,
    DISPLAY_NAMES,
    MODEL_NAMES,
    OUTPUT_DIR,
    THRESHOLD,
)

DPI = 150
MODEL_COLORS = {
    "logistic_regression": "#1f77b4",
    "random_forest": "#2ca02c",
    "xgboost": "#d62728",
}


def plot_confusion_matrix(cm, model_name, dataset_label, save_path):
    """Ve ma tran nham lan [[TN, FP], [FN, TP]]; hang = nhan that, cot = nhan du doan."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Số khách hàng")

    ax.set_xticks([0, 1], labels=CLASS_LABELS)
    ax.set_yticks([0, 1], labels=CLASS_LABELS)
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thực tế")
    ax.set_title(
        f"Ma trận nhầm lẫn - {DISPLAY_NAMES[model_name]}\n"
        f"({dataset_label}, ngưỡng {THRESHOLD})"
    )

    threshold_color = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center", fontsize=13,
                color="white" if cm[i, j] > threshold_color else "black",
            )
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)


def plot_roc_curves(y_true, probs_by_model, dataset_label, save_path):
    """Ve ROC cua cac mo hinh tren cung mot hinh, AUC tinh tu xac suat."""
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for name, prob in probs_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        ax.plot(
            fpr, tpr, lw=2, color=MODEL_COLORS[name],
            label=f"{DISPLAY_NAMES[name]} (AUC = {auc:.{DECIMALS}f})",
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ngẫu nhiên (AUC = 0.5000)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Tỷ lệ dương tính giả (FPR)")
    ax.set_ylabel("Tỷ lệ dương tính thật (TPR / Recall)")
    ax.set_title(f"Đường cong ROC - lớp Rời bỏ ({dataset_label})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)


def save_all_plots(y_true, probs_by_model, dataset_label="Test", output_dir=OUTPUT_DIR):
    from classification_metrics import compute_confusion

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, prob in probs_by_model.items():
        cm = compute_confusion(y_true, prob)
        plot_confusion_matrix(cm, name, dataset_label, output_dir / f"confusion_matrix_{name}.png")
    plot_roc_curves(y_true, probs_by_model, dataset_label, output_dir / "roc_curves.png")


def main():
    from classification_metrics import check_models, load_schema, load_split, predict_probabilities

    schema = load_schema()
    check_models(schema)
    X_test, y_test = load_split("test", schema)
    probs = {name: predict_probabilities(name, X_test) for name in MODEL_NAMES}
    save_all_plots(y_test, probs)
    print(f"Da ve bieu do tren tap test vao {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
