"""So sanh ba mo hinh va xuat ket qua danh gia vao output/.

Quy trinh (chi doc du lieu va mo hinh da luu, khong huan luyen, khong toi uu lai):
1. Kiem tra ba mo hinh dung cung dac trung, nhan va nguong.
2. So sanh tren VALID -> chon mo hinh -> ghi output/best_model.json (truoc khi dung test).
3. Danh gia cuoi cung ca ba mo hinh tren TEST (khong dung de doi mo hinh da chon).
4. Xuat model_comparison.csv, classification_report.txt, best_hyperparameters.json,
   bieu do va output/README.md (so lieu lay truc tiep tu ket qua chay).

Chay: python evaluation/compare_models.py
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from classification_metrics import (
    CLASS_LABELS,
    DECIMALS,
    DISPLAY_NAMES,
    METRIC_NAMES,
    MODEL_NAMES,
    OUTPUT_DIR,
    POSITIVE_CLASS,
    PROJECT_ROOT,
    SELECTION_ORDER,
    THRESHOLD,
    check_models,
    compute_confusion,
    compute_metrics,
    load_schema,
    load_split,
    predict_probabilities,
    selection_key,
)
from plot_results import save_all_plots

# Ten tap dung thong nhat trong bang ket qua
VALID_LABEL = "valid"
TEST_LABEL = "test"


def rel_path(path):
    return Path(path).relative_to(PROJECT_ROOT).as_posix()


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def round_metrics(metrics):
    return {k: round(v, DECIMALS) for k, v in metrics.items()}


def select_best_on_valid(valid_metrics):
    """Chon theo F1 lop Churn=1; bang nhau thi Recall, sau do ROC-AUC (chi dung valid)."""
    return max(valid_metrics, key=lambda name: selection_key(valid_metrics[name]))


def build_comparison_table(valid_metrics, test_metrics):
    rows = []
    for dataset, metrics in ((VALID_LABEL, valid_metrics), (TEST_LABEL, test_metrics)):
        for name in MODEL_NAMES:
            rows.append({"model": DISPLAY_NAMES[name], "dataset": dataset, **round_metrics(metrics[name])})
    return pd.DataFrame(rows, columns=["model", "dataset", *METRIC_NAMES])


def build_classification_report(y_valid, y_test, probs_valid, probs_test, best_name):
    lines = [
        "BAO CAO PHAN LOAI - DU DOAN KHACH HANG ROI BO (Churn)",
        f"Nguong phan loai: {THRESHOLD} | Lop duong: Churn={POSITIVE_CLASS} ({CLASS_LABELS[1]})",
        f"Mo hinh duoc chon tren valid: {DISPLAY_NAMES[best_name]}",
        "Tap valid dung de chon mo hinh; tap test chi dung danh gia cuoi cung.",
        "",
    ]
    for dataset, y_true, probs in (
        ("VALID", y_valid, probs_valid),
        ("TEST", y_test, probs_test),
    ):
        for name in MODEL_NAMES:
            prob = probs[name]
            y_pred = (prob >= THRESHOLD).astype(int)
            tn, fp, fn, tp = compute_confusion(y_true, prob).ravel()
            metrics = compute_metrics(y_true, prob)
            lines += [
                "=" * 70,
                f"{DISPLAY_NAMES[name]} - tap {dataset} ({len(y_true):,} khach hang)",
                "=" * 70,
                classification_report(
                    y_true, y_pred, target_names=list(CLASS_LABELS), digits=DECIMALS, zero_division=0
                ).rstrip(),
                "",
                f"ROC-AUC (tinh tu xac suat): {metrics['roc_auc']:.{DECIMALS}f}",
                f"Ma tran nham lan: TN={tn}, FP={fp}, FN={fn}, TP={tp}",
                "",
            ]
    return "\n".join(lines)


def collect_hyperparameters(metadata_all):
    """Chi tong hop tu metadata, khong tim kiem lai."""
    result = {}
    for name in MODEL_NAMES:
        meta = metadata_all[name]
        entry = {"algorithm": meta["algorithm"], "hyperparameters": meta["hyperparameters"]}
        if "fixed_parameters" in meta:
            entry["fixed_parameters"] = meta["fixed_parameters"]
        search = meta.get("hyperparameter_search")
        if search:
            entry["hyperparameter_search"] = {
                k: search[k]
                for k in (
                    "search_method", "scoring", "cv", "cv_folds", "random_state",
                    "n_iter", "n_candidates_evaluated", "search_space", "best_cv_f1",
                )
                if k in search
            }
        entry["libraries"] = meta.get("libraries", {})
        result[name] = entry
    return result


def pct(x):
    return f"{x * 100:.2f}%"


def md_table(df):
    header = "| Mô hình | Accuracy | Precision | Recall | F1 | ROC-AUC |\n|---|---|---|---|---|---|\n"
    body = ""
    for _, r in df.iterrows():
        body += (
            f"| {r['model']} | {r['accuracy']:.{DECIMALS}f} | {r['precision']:.{DECIMALS}f} | "
            f"{r['recall']:.{DECIMALS}f} | {r['f1']:.{DECIMALS}f} | {r['roc_auc']:.{DECIMALS}f} |\n"
        )
    return header + body


def build_readme(table, best_name, y_test, probs_test, valid_metrics, test_metrics, hyper):
    valid_df = table[table["dataset"] == VALID_LABEL]
    test_df = table[table["dataset"] == TEST_LABEL]
    n_test = len(y_test)
    n_pos = int((y_test == 1).sum())

    details = {}
    for name in MODEL_NAMES:
        tn, fp, fn, tp = compute_confusion(y_test, probs_test[name]).ravel()
        details[name] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    # Nhan xet tung mo hinh (chi dung so lieu thuc te tren test)
    per_model = ""
    for name in MODEL_NAMES:
        d, m = details[name], test_metrics[name]
        per_model += (
            f"- **{DISPLAY_NAMES[name]}**: phát hiện đúng {d['tp']:,}/{n_pos:,} khách hàng rời bỏ "
            f"(Recall = {pct(m['recall'])}), bỏ sót {d['fn']:,} khách hàng (tỷ lệ bỏ sót {pct(d['fn'] / n_pos)}). "
            f"Cảnh báo nhầm {d['fp']:,} khách hàng không rời bỏ (Precision = {pct(m['precision'])}). "
            f"Tổng số dự đoán sai = {d['fp'] + d['fn']:,}/{n_test:,} "
            f"(tỷ lệ dự đoán sai {pct((d['fp'] + d['fn']) / n_test)}).\n"
        )

    def ranking(metric):
        order = sorted(MODEL_NAMES, key=lambda n: test_metrics[n][metric], reverse=True)
        return " > ".join(f"{DISPLAY_NAMES[n]} ({test_metrics[n][metric]:.{DECIMALS}f})" for n in order)

    best_test_f1 = max(MODEL_NAMES, key=lambda n: selection_key(test_metrics[n]))
    if best_test_f1 == best_name:
        consistency = (
            f"Mô hình chọn trên valid ({DISPLAY_NAMES[best_name]}) cũng có F1 lớp Rời bỏ cao nhất trên test."
        )
    else:
        consistency = (
            f"Mô hình chọn trên valid là {DISPLAY_NAMES[best_name]}, trong khi trên test "
            f"{DISPLAY_NAMES[best_test_f1]} có F1 cao nhất. Theo quy ước, mô hình đã chọn không bị thay đổi "
            f"dựa trên kết quả test."
        )

    f1_values = [test_metrics[n]["f1"] for n in MODEL_NAMES]
    auc_values = [test_metrics[n]["roc_auc"] for n in MODEL_NAMES]
    spread = (
        f"Chênh lệch F1 giữa mô hình cao nhất và thấp nhất trên test là {max(f1_values) - min(f1_values):.{DECIMALS}f}; "
        f"chênh lệch ROC-AUC là {max(auc_values) - min(auc_values):.{DECIMALS}f}."
    )

    gap_lines = ""
    for name in MODEL_NAMES:
        gap = test_metrics[name]["f1"] - valid_metrics[name]["f1"]
        gap_lines += f"- {DISPLAY_NAMES[name]}: F1 valid = {valid_metrics[name]['f1']:.{DECIMALS}f}, F1 test = {test_metrics[name]['f1']:.{DECIMALS}f} (lệch {gap:+.{DECIMALS}f}).\n"

    return f"""# Kết quả đánh giá và so sánh mô hình

Thư mục này do phần **Đánh giá và so sánh mô hình** tạo ra. Toàn bộ số liệu bên dưới được sinh tự động từ lần chạy
`evaluation/compare_models.py` trên các mô hình đã lưu trong `saved models/`; không có số liệu nhập tay.

## 1. Quy ước đánh giá

- Dữ liệu: `dataset/processed/valid.csv` (chọn mô hình) và `test.csv` (đánh giá cuối cùng, {n_test:,} khách hàng, trong đó {n_pos:,} khách hàng rời bỏ).
- Nhãn `Churn`: 0 = {CLASS_LABELS[0]}, 1 = {CLASS_LABELS[1]}. Precision, Recall, F1 tính cho lớp `Churn=1`; ROC-AUC tính từ xác suất lớp 1.
- Ngưỡng phân loại: {THRESHOLD} cho cả ba mô hình (đã kiểm tra trong metadata).
- Tiêu chí chọn mô hình trên **valid**: F1 của lớp `Churn=1`; nếu bằng nhau ưu tiên Recall, sau đó ROC-AUC (so sánh trên giá trị đã làm tròn {DECIMALS} chữ số thập phân).
- Tập test chỉ dùng cho đánh giá cuối cùng, không dùng để chọn mô hình, đổi tham số hay chỉnh ngưỡng.
- Mọi chỉ số làm tròn {DECIMALS} chữ số thập phân; tỷ lệ phần trăm làm tròn 2 chữ số.

## 2. Mô hình được chọn

**{DISPLAY_NAMES[best_name]}** (tham chiếu trong `best_model.json`, F1 valid = {valid_metrics[best_name]['f1']:.{DECIMALS}f}, Recall valid = {valid_metrics[best_name]['recall']:.{DECIMALS}f}, ROC-AUC valid = {valid_metrics[best_name]['roc_auc']:.{DECIMALS}f}).

### Kết quả trên tập valid

{md_table(valid_df)}
### Kết quả trên tập test

{md_table(test_df)}
## 3. Nhận xét

### Khả năng phát hiện khách hàng rời bỏ và tỷ lệ dự đoán sai (tập test)

{per_model}
### Sự khác nhau giữa các mô hình (tập test)

- Xếp hạng theo F1: {ranking('f1')}.
- Xếp hạng theo Recall: {ranking('recall')}.
- Xếp hạng theo Precision: {ranking('precision')}.
- Xếp hạng theo ROC-AUC: {ranking('roc_auc')}.
- {spread}
- {consistency}

### Độ lệch giữa valid và test

{gap_lines}
## 4. Các file kết quả

| File | Nội dung |
|---|---|
| `best_model.json` | Mô hình được chọn trên valid: đường dẫn, ngưỡng, tiêu chí và chỉ số valid |
| `model_comparison.csv` | Năm chỉ số của ba mô hình, phân biệt hàng `valid` và `test` |
| `classification_report.txt` | Báo cáo phân loại chi tiết và ma trận nhầm lẫn dạng số |
| `confusion_matrix_logistic_regression.png` | Ma trận nhầm lẫn Logistic Regression (test) |
| `confusion_matrix_random_forest.png` | Ma trận nhầm lẫn Random Forest (test) |
| `confusion_matrix_xgboost.png` | Ma trận nhầm lẫn XGBoost (test) |
| `roc_curves.png` | Đường cong ROC của ba mô hình (test) |
| `best_hyperparameters.json` | Cấu hình tối ưu tổng hợp từ metadata (không tìm kiếm lại) |

## 5. Cách chạy lại

```bash
pip install -r requirements.txt   # hoặc cài pandas, scikit-learn, xgboost, matplotlib, joblib
python evaluation/compare_models.py
```

Yêu cầu có đủ `logistic_regression`, `random_forest`, `xgboost` (`.joblib` và `.json`) trong `saved models/`.
Có thể vẽ lại riêng biểu đồ bằng `python evaluation/plot_results.py`. Các script chỉ đọc dữ liệu và mô hình đã lưu,
không huấn luyện và không sửa `dataset/`, `baseline models/`, `model/`, `saved models/`.
"""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    schema = load_schema()
    metadata_all = check_models(schema)
    print("Ba mo hinh dung cung dac trung, nhan va nguong:", THRESHOLD)

    # ---- Buoc 1: so sanh tren valid va chon mo hinh (chua dung test) ----
    X_valid, y_valid = load_split(VALID_LABEL, schema)
    probs_valid = {n: predict_probabilities(n, X_valid) for n in MODEL_NAMES}
    valid_metrics = {n: compute_metrics(y_valid, probs_valid[n]) for n in MODEL_NAMES}

    best_name = select_best_on_valid(valid_metrics)
    write_json(
        OUTPUT_DIR / "best_model.json",
        {
            "model_name": best_name,
            "algorithm": DISPLAY_NAMES[best_name],
            "model_path": f"saved models/{best_name}.joblib",
            "metadata_path": f"saved models/{best_name}.json",
            "threshold": THRESHOLD,
            "positive_class": POSITIVE_CLASS,
            "selection_criteria": {
                "evaluation_set": VALID_LABEL,
                "primary": "F1 cua lop Churn=1",
                "tie_breakers": ["Recall cua lop Churn=1", "ROC-AUC"],
                "rounding_decimals": DECIMALS,
            },
            "valid_metrics": round_metrics(valid_metrics[best_name]),
            "valid_metrics_all_models": {
                DISPLAY_NAMES[n]: round_metrics(valid_metrics[n]) for n in MODEL_NAMES
            },
        },
    )
    print(f"Mo hinh duoc chon tren valid: {DISPLAY_NAMES[best_name]}")

    # ---- Buoc 2: danh gia cuoi cung tren test (khong doi mo hinh da chon) ----
    X_test, y_test = load_split(TEST_LABEL, schema)
    probs_test = {n: predict_probabilities(n, X_test) for n in MODEL_NAMES}
    test_metrics = {n: compute_metrics(y_test, probs_test[n]) for n in MODEL_NAMES}

    table = build_comparison_table(valid_metrics, test_metrics)
    table.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8")

    report = build_classification_report(y_valid, y_test, probs_valid, probs_test, best_name)
    (OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    hyper = collect_hyperparameters(metadata_all)
    write_json(OUTPUT_DIR / "best_hyperparameters.json", hyper)

    save_all_plots(y_test, probs_test, dataset_label="Test")

    readme = build_readme(table, best_name, y_test, probs_test, valid_metrics, test_metrics, hyper)
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print("\n" + table.to_string(index=False))
    print(f"\nDa luu ket qua vao {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
