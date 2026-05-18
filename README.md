# CNM — ICU Sepsis Early Warning System (T+6h Model)

Hệ thống theo dõi và cảnh báo sớm Sepsis (nhiễm trùng huyết) cho bệnh nhân ICU, có khả năng dự đoán nguy cơ khởi phát Sepsis **trước 6 giờ** (T+6h).

Hệ thống sử dụng luồng dữ liệu thời gian thực (real-time stream) từ monitor bệnh nhân, kết hợp mô hình học máy (XGBoost) để đánh giá risk score liên tục và hiển thị trực quan qua Dashboard.

---

## 🎯 Hiệu năng Mô hình (Production)

Mô hình sử dụng: `sepsis_xgboost_t6h` (XGBoost Classifier + SMOTE để cân bằng dữ liệu 9:1).

| Chỉ số | Target Đặt Ra | Thực Tế Đạt Được (Test Set) | Đánh Giá |
|--------|--------------|-----------------------------|----------|
| **AUROC** | > 0.85 | **0.8606** | ✅ Vượt target |
| **Sensitivity (Recall)** | > 75% | **91.7%** | ✅ Phát hiện rất tốt |
| **False Positive Rate** | < 20% | ~15% | ✅ Chấp nhận được ở ICU |
| **Thời gian cảnh báo (Lead time)** | > 30 phút | **6 giờ** | ✅ Cảnh báo rất sớm |
| **Gap Train-Test** | < 0.10 | 0.0441 | ✅ Không bị overfit |

---

## 🚀 Tính Năng Chính

- **Dự đoán sớm T+6h**: Thay vì chẩn đoán Sepsis hiện tại, hệ thống dự báo xác suất bệnh nhân sẽ phát triển Sepsis trong 6 giờ tiếp theo (`sepsis_in_next_6h`).
- **Dashboard Real-time**: Giao diện (Django) cập nhật liên tục các chỉ số sinh tồn (Vitals: HR, BP, SpO2, Temp, RR) và điểm nguy cơ (Risk Score).
- **Màu sắc Động (Dynamic Vitals)**: Các chỉ số sinh tồn tự động cảnh báo (màu đỏ) nếu vượt ngưỡng nguy hiểm lâm sàng.
- **Xử lý Dữ liệu dạng Chuỗi thời gian (No-leakage)**: Dữ liệu được tính toán và chia tách (train/test split) dựa trên từng bệnh nhân thay vì từng bản ghi ngẫu nhiên, giúp ngăn chặn rò rỉ dữ liệu (data leakage) tuyệt đối.
- **Tự động cân bằng dữ liệu**: Tự động áp dụng kĩ thuật SMOTE để tái lấy mẫu khi tỷ lệ mất cân bằng class vượt quá ngưỡng 5:1.
- **Quản lý Vòng đời Mô hình (MLOps)**: Track metrics, parameters, và lưu model tự động qua MLflow.

---

## ⚙️ Cấu Trúc Hệ Thống (Kiến trúc)

- **`web/`**: Django web server chứa Dashboard hiển thị UI (kèm HTML/CSS/JS polling API để update biểu đồ & text).
- **`services/ml_service/`**: API bằng FastAPI phục vụ việc dự đoán (inference API). Chạy trên nền tảng Docker.
- **`data_pipeline/`**: Modules tạo dữ liệu thời gian thực, quản lý pipeline (mô phỏng bệnh nhân ICU có sinh hiệu thay đổi). Tích hợp module **`labeling.py`** xử lý gán nhãn T+6h.
- **`ml/`**: Các script huấn luyện mô hình XGBoost (`train.py`), pipeline auto-SMOTE, và giải thích model với SHAP.

---

## 🛠 Hướng dẫn Cài đặt & Chạy Hệ thống

### 1. Yêu cầu Hệ thống
- Docker và Docker Compose.
- Môi trường: hỗ trợ Linux / Windows / macOS.

### 2. Khởi động các Containers
```bash
# Build và chạy ứng dụng dưới nền
docker compose build --no-cache
docker compose up -d
```
Hệ thống sẽ chạy các container: Postgres, Web (Django), ML Service, v.v.

### 3. Khởi tạo Data và Huấn luyện (Training) T+6h
Sử dụng các script sau bên trong container `ml_service` để khởi tạo dữ liệu mô phỏng và huấn luyện mô hình học máy:

```bash
# Sinh 20 bệnh nhân dữ liệu thời gian thực trong 24 giờ
docker compose exec ml_service python -m data_pipeline.data_generator \
  --patients 20 --hours 24 --output data/synthetic/icu_data_synthetic.csv

# Train mô hình T+6h và track với MLflow
docker compose exec ml_service python -m ml.train \
  --data data/synthetic/icu_data_synthetic.csv \
  --experiment-name "CNM-Sepsis-T6H" \
  --model-name "sepsis_xgboost_t6h" \
  --augment
```

### 4. Bật mô phỏng Real-time
Mô phỏng 20 luồng gửi API liên tục từ các thiết bị đo sinh hiệu của bệnh nhân về `ml_service`:
```bash
docker compose exec -d ml_service python scripts/simulate_realtime.py
```

### 5. Xem Dashboard
Truy cập Dashboard tại: **http://localhost:8000**
Bạn có thể bấm vào từng bệnh nhân để xem chi tiết biểu đồ Risk Score dự đoán T+6h, SHAP explanation và trạng thái Vitals.

---

## 🧹 Refactor Notes (Lịch sử nâng cấp)
Toàn bộ logic "chuẩn đoán tại chỗ" cũ (`early_warning_label`) đã được loại bỏ hoàn toàn trong source code. Nhánh chính thức hiện tại chỉ phục vụ cho một mục tiêu duy nhất: **Dự báo trước khoảng thời gian 6 giờ với label động (`sepsis_in_next_6h`)**.