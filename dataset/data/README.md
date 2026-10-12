# README - Dữ liệu và tiền xử lý

## 1. Giới thiệu

Phần này phụ trách đọc dữ liệu, khám phá dữ liệu, chia tập dữ liệu và tiền xử lý dữ liệu phục vụ bài toán **dự đoán nguy cơ rời bỏ khách hàng (Customer Churn Prediction)**.

Dữ liệu đầu vào là bộ dữ liệu tổng hợp gồm 100.000 bản ghi khách hàng.

## 2. Dữ liệu đầu vào

**Đường dẫn:** `dataset/data/synthetic_customer_churn_100k.csv`

Dữ liệu gồm 9 cột:

| Cột | Ý nghĩa |
|---|---|
| `CustomerID` | Mã định danh khách hàng |
| `Age` | Tuổi khách hàng |
| `Gender` | Giới tính |
| `Tenure` | Thời gian khách hàng sử dụng dịch vụ |
| `MonthlyCharges` | Chi phí dịch vụ hằng tháng |
| `Contract` | Loại hợp đồng |
| `PaymentMethod` | Phương thức thanh toán |
| `TotalCharges` | Tổng chi phí |
| `Churn` | Trạng thái rời bỏ dịch vụ |

Trong đó, `Churn` là biến mục tiêu:
- `No`: khách hàng không rời bỏ dịch vụ, được mã hóa thành `0`.
- `Yes`: khách hàng rời bỏ dịch vụ, được mã hóa thành `1`.

`CustomerID` chỉ dùng để định danh và kiểm tra trùng lặp giữa các tập dữ liệu, không được sử dụng làm đặc trưng đầu vào của mô hình.

## 3. Các bước thực hiện

### Bước 1. Đọc và kiểm tra dữ liệu

File: `dataset/data preprocessing/load_data.py`

- Đọc dữ liệu đầu vào.
- Kiểm tra cấu trúc và kiểu dữ liệu.
- Kiểm tra giá trị thiếu, bản ghi trùng lặp và tính duy nhất của `CustomerID`.
- Kiểm tra các giá trị của biến mục tiêu `Churn`.

### Bước 2. Khám phá dữ liệu (EDA)

File: `dataset/data preprocessing/explore_data.py`

- Thống kê các đặc trưng số.
- Thống kê phân bố của biến `Churn`.
- Phân tích tỷ lệ rời bỏ theo `Gender`, `Contract` và `PaymentMethod`.
- Tạo các biểu đồ phân bố và biểu đồ tỷ lệ rời bỏ.

Kết quả được lưu trong thư mục `dataset/exploration/`, bao gồm các file CSV thống kê và hình ảnh biểu đồ PNG.

### Bước 3. Chia tập dữ liệu

File: `dataset/data preprocessing/split_data.py`

Dữ liệu được chia theo tỷ lệ:
- Tập huấn luyện (train): 80% — 80.000 bản ghi.
- Tập xác thực (validation): 10% — 10.000 bản ghi.
- Tập kiểm tra (test): 10% — 10.000 bản ghi.

Sử dụng chia tập phân tầng theo biến `Churn` với `random_state=42` nhằm duy trì tỷ lệ các lớp tương đối đồng đều giữa các tập.

Các tập dữ liệu ban đầu được ghi vào thư mục `dataset/processed/`. Chương trình kiểm tra để bảo đảm các tập không có khách hàng trùng nhau.

### Bước 4. Tiền xử lý dữ liệu

File: `dataset/data preprocessing/preprocess_data.py`

Các bước tiền xử lý gồm:

- Mã hóa `Churn`: `No = 0`, `Yes = 1`.
- Loại bỏ cột định danh `CustomerID`.
- Với các đặc trưng số: điền giá trị thiếu bằng trung vị và chuẩn hóa bằng `StandardScaler`.
- Với các đặc trưng phân loại: điền giá trị thiếu bằng giá trị xuất hiện nhiều nhất và mã hóa One-Hot Encoding.
- Huấn luyện bộ tiền xử lý trên tập train; sử dụng chính bộ tiền xử lý đó để biến đổi tập validation và test.
- Bảo đảm tên và thứ tự các đặc trưng đầu ra nhất quán giữa ba tập.

## 4. Các file đầu ra

Thư mục `dataset/processed/` chứa:

| File | Mô tả |
|---|---|
| `train.csv` | Dữ liệu huấn luyện sau tiền xử lý |
| `valid.csv` | Dữ liệu xác thực sau tiền xử lý |
| `test.csv` | Dữ liệu kiểm tra sau tiền xử lý |
| `preprocessor.joblib` | Bộ tiền xử lý đã được huấn luyện trên tập train |
| `feature_schema.json` | Mô tả các cột đầu vào, kiểu dữ liệu, giá trị hợp lệ và thứ tự đặc trưng đầu ra |

Sau tiền xử lý, mỗi tập gồm 14 đặc trưng đầu vào và một cột mục tiêu `Churn`. Các đặc trưng đầu vào đều ở dạng số.

## 5. Thứ tự chạy chương trình

Chạy các lệnh từ thư mục gốc của dự án:

```bash
python "dataset/data preprocessing/load_data.py"
python "dataset/data preprocessing/explore_data.py"
python "dataset/data preprocessing/split_data.py"
python "dataset/data preprocessing/preprocess_data.py"
```

**Lưu ý:** Cần chạy `split_data.py` trước `preprocess_data.py`. Chương trình chia tập tạo lại các file dữ liệu đầu vào, còn chương trình tiền xử lý sẽ ghi đè các file trong `dataset/processed/` bằng dữ liệu đã xử lý. Vì vậy, khi chạy lại toàn bộ quy trình, hãy thực hiện đúng thứ tự trên.

Các thư viện Python sử dụng gồm `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` và `joblib`.

## 6. Phạm vi thực hiện

Phần này tập trung vào đọc dữ liệu, khám phá dữ liệu, chia tập và tiền xử lý dữ liệu. Không thực hiện huấn luyện, đánh giá mô hình hoặc xây dựng giao diện ứng dụng.