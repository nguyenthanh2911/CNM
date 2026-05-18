# ICU Sepsis Early Warning System 

![CI](https://github.com/nguyenthanh2911/CNM/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![MLflow](https://img.shields.io/badge/tracking-MLflow-blue)
![Docker](https://img.shields.io/badge/deploy-Docker-blue)

> Hệ thống theo dõi ICU real-time sử dụng Machine Learning nhằm cảnh báo sớm Sepsis

---

## Mục lục

1. [Giới thiệu đề tài](#1-giới-thiệu-đề-tài)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Công nghệ sử dụng](#4-công-nghệ-sử-dụng)
5. [Dữ liệu](#5-dữ-liệu)
6. [Hướng dẫn cài đặt](#6-hướng-dẫn-cài-đặt)
7. [Hướng dẫn sử dụng](#7-hướng-dẫn-sử-dụng)
8. [Quy trình ML Pipeline](#8-quy-trình-ml-pipeline)
9. [API Reference](#9-api-reference)
10. [Kết quả và đánh giá](#10-kết-quả-và-đánh-giá)

---

## 1. Giới thiệu đề tài

### Bài toán

**Sepsis** (nhiễm khuẩn huyết) là phản ứng đe dọa tính mạng của cơ thể khi nhiễm trùng, gây ra hơn **270.000 ca tử vong mỗi năm** tại Mỹ. Phát hiện sớm trong **1–6 giờ đầu** tăng tỉ lệ sống sót lên 80%, nhưng y tá ICU phải theo dõi hàng chục chỉ số liên tục cho nhiều bệnh nhân cùng lúc — dẫn đến nguy cơ bỏ sót.

### Giải pháp

Xây dựng hệ thống MLOps hoàn chỉnh theo 3 tầng:

- **Data & Training**: Sinh dữ liệu synthetic ICU, xây dựng features lâm sàng, train mô hình **XGBoost** và theo dõi thí nghiệm qua **MLflow**
- **Serving & Deployment**: Dự đoán nguy cơ sepsis mỗi 5 phút qua **FastAPI**, hiển thị **dashboard real-time** bằng Django WebSocket, đóng gói bằng **Docker Compose**, CI/CD qua **GitHub Actions**
- **Monitoring & Retraining**: Phát hiện data drift bằng **Evidently AI**, tự động retrain bằng **Prefect**, theo dõi hệ thống bằng **Prometheus + Grafana**

### Mục tiêu kỹ thuật

| Chỉ số | Mục tiêu |
|--------|----------|
| AUROC | > 0.85 |
| Sensitivity | > 75% |
| False Positive Rate | < 20% |
| Latency cảnh báo (end-to-end) | < 5 phút |
| Alert lead time trước sepsis | > 30 phút |

---

## 2. Kiến trúc hệ thống

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         DATA & TRAINING                                  │
│                                                                          │
│  data_generator.py ──► CSV ──► feature_builder.py                       │
│  (synthetic ICU)           (pandas)  (SOFA, NEWS2, qSOFA, rolling)      │
│                                     ──► train.py ──► MLflow Track        │
│                                                      ──► MLflow Registry│
│                                     ──► early_warning.py (rule engine)  │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      SERVING & DEPLOYMENT                                │
│                                                                          │
│  GitHub Actions ──► pytest ──► Build Image ──► Docker Compose            │
│                                                                          │
│  FastAPI ML Service (port 8001)                                          │
│    POST /vitals ──► preprocess ──► XGBoost predict ──► SHAP explain     │
│                  └──► EarlyWarning (trend+rate+threshold) ──► risk_score │
│    GET  /health                                                          │
│    GET  /vitals/{patient_id}/history                                     │
│    GET  /metrics (Prometheus: predictions_total, inference_seconds)      │
│                                                                          │
│  FastAPI Alert Service (port 8002)                                       │
│    POST /alerts (từ ML Service, risk ≥ 0.7)                              │
│    GET  /alerts?patient_id=&status=&limit=                               │
│    GET  /alerts/stats                                                    │
│    WebSocket ──► push critical alerts real-time                          │
│    GET  /metrics (Prometheus: active_alerts)                             │
│                                                                          │
│  Django Dashboard (port 8000) ──► WebSocket (Daphne) ──► Alerts page    │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    MONITORING & RETRAINING                               │
│                                                                          │
│  Evidently AI ──► drift_score (DataDriftPreset)                          │
│                      │                                                   │
│               drift_score > 0.7 ?                                        │
│              /               \                                           │
│     Prefect retrain_flow    No retrain                                   │
│     (train.py subprocess)                                                │
│            │                                                             │
│     New model AUROC > Production AUROC + 0.01 ?                          │
│        Yes ──► Promote to Production (archive old)                       │
│        No  ──► Keep old model                                            │
└──────────────────────────────────────────────────────────────────────────┘
```

### Luồng dữ liệu chính

```
simulate_realtime.py (20 bệnh nhân, 3 nhóm: LOW/WARN/HIGH)
  │ POST /vitals (mỗi 10s thực = 1h mô phỏng, 240 steps)
  ▼
FastAPI ML Service (:8001)
  ├── Clinical scores: SOFA, NEWS2 (từ raw vitals)
  ├── Preprocess: SimpleImputer(median) + StandardScaler
  ├── XGBoost predict → risk_score (0..1)
  ├── SHAP TreeExplainer → top-5 features
  ├── EarlyWarningPredictor (rule-based)
  │   ├── trend_score (30 phút)       [weight 30%]
  │   ├── rate_of_change_score        [weight 20%]
  │   └── threshold_score             [weight 50%]
  │   └── → early_warning_probability + level (LOW/MEDIUM/HIGH)
  └── Lưu: risk, clinical scores, raw vitals, early warning → PostgreSQL
              │
              ├── risk < 0.3  → log only
              ├── 0.3 – 0.7  → Dashboard WARNING
              ├── ≥ 0.7      → Alert Service (:8002) → WebSocket push
              └── early_warning HIGH (nhưng risk < 0.7) → Alert Service early_warning alert
                        │
                        ▼
              Django Dashboard (:8000) — WebSocket real-time
              └── Danh sách bệnh nhân + chi tiết + biểu đồ risk trend + alerts
```

---

## 3. Cấu trúc thư mục

```
CNM/
├── docker-compose.yml           # Cấu hình các dịch vụ Docker (ML service, Django, MLflow, Postgres, v.v)
├── pytest.ini                   # Cấu hình môi trường và tuỳ chọn chạy pytest
├── README.md                    # Tài liệu dự án
├── requirements.txt             # Danh sách thư viện phụ thuộc Python
├── test_vitals.json             # File JSON chứa dữ liệu mẫu dùng cho test
│
├── artifacts/                   # Thư mục lưu các artifacts sinh ra
│   └── preprocessor_t6h.joblib  # Pipeline tiền xử lý lưu ở định dạng joblib
│
├── data/                        # Tổ chức các loại dữ liệu
│   ├── processed/               # Dữ liệu sau khi đã tiền xử lý, rút trích feature
│   ├── raw/                     # Dữ liệu gốc rỗng ban đầu (không commit)
│   └── synthetic/               # Dữ liệu giải lập tự động tạo ra
│       └── icu_data_synthetic.csv
│
├── data_pipeline/               # Xử lý dữ liệu đầu vào
│   ├── __init__.py
│   ├── data_generator.py        # Sinh dữ liệu synthetic mô phỏng các bệnh nhân ICU
│   ├── labeling.py              # Tạo label T+6h (sepsis_in_next_6h), patient-based split
│   └── preprocessor.py          # Tiền xử lý dữ liệu (chuẩn hoá, điền khuyết)
│
├── docs/                        # Tài liệu dự án bổ sung
│   └── database_schema.sql      # Schema tạo cấu trúc bảng cho PostgreSQL
│
├── feature_engineering/         # Mã nguồn trích xuất và biến đổi đặc trưng
│   ├── __init__.py
│   ├── clinical_scores.py       # Tính toán các chỉ số lâm sàng (ví dụ SOFA, NEWS2, qSOFA)
│   ├── feature_builder.py       # Pipeline tổng hợp các features dữ liệu
│   └── vitals_features.py       # Trích xuất đặc trưng từ các chỉ số sinh tồn (vitals)
│
├── ml/                          # Các thành phần Machine Learning
│   ├── __init__.py
│   ├── early_warning.py         # Rule engine: trend + rate_of_change + threshold → early_warning_probability (30 phút)
│   ├── evaluate.py              # Đánh giá mô hình (AUROC, F1, Sensitivity, Specificity, confusion matrix, ROC curve)
│   ├── explain.py               # SHAP TreeExplainer giải thích top-5 features ảnh hưởng nhất
│   ├── mlflow_utils.py          # Logging params/metrics/model lên MLflow, load model từ Registry
│   ├── train.py                 # Pipeline training T+6h: patient-based split, auto SMOTE, CV 5-fold, MLflow logging
│   └── models/
│       ├── __init__.py
│       └── xgboost_model.py     # SepsisXGBModel (150 estimators, max_depth=4, scale_pos_weight, early_stopping)
│
├── monitoring/                  # Giám sát hệ thống và Data Drift
│   ├── __init__.py
│   ├── drift_detector.py        # Ứng dụng Evidently AI phát hiện data drift
│   ├── retrain_flow.py          # Pipeline Prefect kích hoạt retrain tự động
│   ├── grafana/
│   │   └── dashboards/
│   │       └── icu_dashboard.json # File cấu hình Dashboard Grafana hiển thị trạng thái
│   └── prometheus/
│       └── prometheus.yml       # Cấu hình cho việc thu thập metric của Prometheus
│
├── scripts/                     # Scripts hỗ trợ vòng đời hệ thống
│   ├── check_health.sh          # Script bash kiểm tra trạng thái sức khoẻ tự động
│   ├── ci_seed_data.py          # Script seed dữ liệu giả lập cho CI/CD test
│   ├── run_demo.sh              # Bash script để khởi chạy nhanh demo ứng dụng
│   ├── seed_patients.py         # Chèn sẵn hồ sơ bệnh nhân ảo vào PostgreSQL
│   ├── setup_db.ps1             # PowerShell script thiết lập database cho môi trường Windows
│   ├── setup_db.sh              # Bash script thiết lập database cho môi trường Linux/macOS
│   └── simulate_realtime.py     # Mô phỏng đẩy dữ liệu liên tục như sensor ICU thực tế 
│
├── services/                    # Tầng Microservices backend 
│   ├── __init__.py
│   ├── alert_service/           # Dịch vụ quản lý cảnh báo (FastAPI)
│   │   ├── __init__.py
│   │   ├── Dockerfile
│   │   ├── main.py              # Điểm khởi chạy của dịch vụ gửi cảnh báo websocket
│   │   ├── schemas.py           # Định nghĩa Pydantic schema cho Input/Output
│   │   └── websocket_manager.py # Quản lý kết nối WebSocket để push alerts realtime
│   └── ml_service/              # Dịch vụ Inferencing Machine Learning (FastAPI)
│       ├── __init__.py
│       ├── Dockerfile
│       ├── main.py              # Khởi tạo API nhận luồng vitals và trả dự đoán ML
│       ├── predictor.py         # Logic load model từ mlflow để chấm điểm (risk_score)
│       └── schemas.py           # Pydantic Models áp dụng chung (cho ML request)
│
├── tests/                       # Automated Testing (Pytest)
│   ├── __init__.py
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_pipeline.py     # Test tính liên kết của pipelines (E2E Integration)
│   └── unit/
│       ├── __init__.py
│       ├── test_api.py          # Viết test riêng cho từng endpoint của các service
│       ├── test_features.py     # Kiểm thử logic của các hàm tạo feature engineering
│       ├── test_labeling.py     # Kiểm thử logic T+6h labeling (window, split, stats)
│       └── test_model.py        # Unit test xác minh inference model đúng kết quả
│
└── web/                         # Ứng dụng Web / Frontend
    ├── Dockerfile
    ├── manage.py                # Điểm khởi chạy của Backend framework Django
    ├── config/                  # Thư mục cấu hình cốt lõi của Django 
    │   ├── __init__.py
    │   ├── asgi.py              # Interface khởi tạo app hỗ trợ dạng async WebSocket
    │   ├── settings.py          # Cấu hình chính (database, app config, secrets)
    │   ├── urls.py              # URL matching cấp website
    │   └── wsgi.py              # Interface khởi tạo server WSGI
    ├── dashboard/               # Ứng dụng con của Django thao tác Dashboard
    │   ├── __init__.py
    │   ├── consumers.py         # Django Channels nhận / trao đổi luồng WebSocket 
    │   ├── models.py            # Mô hình dữ liệu ORM nối về Postgres DB
    │   ├── routing.py           # Định nghĩa WebSockets routes
    │   ├── urls.py              # Các routes HTTP GET của giao diện màn hình
    │   ├── views.py             # View chức năng và render context (logic Frontend)
    │   └── templates/           
    │       └── dashboard/       # Tệp UI HTML với Tailwind/CSS
    │           ├── alerts.html         # Giao diện xem cảnh báo tập trung
    │           ├── base.html           # Layout chung gốc
    │           ├── patient_detail.html # Trang xem chi tiết 1 bệnh nhân
    │           └── patient_list.html   # Bảng giám sát danh sách ICU toàn diện
```

---

## 4. Công nghệ sử dụng

| Tầng | Công nghệ |
|------|-----------|
| Data & Feature | Pandas, NumPy, Scikit-learn (Pipeline, KNNImputer, StandardScaler) |
| Machine Learning | XGBoost, SHAP (TreeExplainer), EarlyWarning rule engine |
| Experiment Tracking | MLflow (Tracking Server + Model Registry) |
| Backend API | FastAPI, Pydantic v2 |
| Alerting & WebSocket | FastAPI WebSocket, Django Channels (Daphne) |
| Web Dashboard | Django 5.0, Django Channels, Postgres ORM |
| Database | PostgreSQL 15 |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions, pytest, pytest-cov |
| Monitoring | Evidently AI (DataDriftPreset), Prefect 2 (retrain_flow), Prometheus, Grafana |

---

## 5. Dữ liệu

### Nguồn dữ liệu

Dự án sử dụng **dữ liệu synthetic** được sinh bằng `data_generator.py`, mô phỏng các chỉ số sinh lý ICU thực tế:

| Nhóm | Chỉ số |
|------|--------|
| Vitals | heart_rate, systolic_bp, diastolic_bp, temperature, spo2, respiratory_rate |
| Labs | lactate, wbc, creatinine, bilirubin, platelet |
| Scores (derived) | SOFA, NEWS2, qSOFA |

### Nhãn (Label)

Dự án sử dụng **2 loại label**:

1. **`sepsis_label`** (`int` 0/1): Gán dựa trên `has_sepsis` flag sinh từ `PhysiologicalModel`. Tỉ lệ mặc định 40% bệnh nhân sepsis, 60% không sepsis.

2. **`sepsis_in_next_6h`** (`int` 0/1) — **Label chính cho training**:
   - Được tạo bởi `labeling.py:create_t6h_labels()` dựa trên cột `sepsis_onset_hour`
   - `y[t] = 1` nếu sepsis onset xảy ra trong khoảng `(t_hour, t_hour + 6h]`
   - `y[t] = 0` còn lại (bao gồm cả sau khi onset đã xảy ra)
   - Tỉ lệ positive: ~10% (imbalance ratio ~9:1)

**Cột `sepsis_onset_hour`** được sinh cùng với dữ liệu:
- Sepsis patients: onset random trong khoảng giờ 8–18 (giữa ca trực)
- Non-sepsis patients: `onset_hour = None`
- Dùng để tạo label T+6h động (không cần hardcode window giờ)

Dữ liệu synthetic có built-in **confounders** để tăng độ khó:
- Non-sepsis patient có bad spikes transient (20%)
- Sepsis patient có recovery period (15%)
- Nhiễu thiết bị (equipment noise) ngẫu nhiên 2% trên mỗi vital
- Thiếu lab values ngẫu nhiên (~5%)
- Age vitals multiplier cho bệnh nhân > 70 tuổi

### Phân chia tập dữ liệu

| Tập | Tỉ lệ | Ghi chú |
|-----|-------|---------|
| Train | ~60% | Patient-based split — mỗi patient chỉ xuất hiện trong 1 tập |
| Validation | ~20% | Tránh data leakage hoàn toàn |
| Test | ~20% | Đánh giá cuối cùng, threshold binary = 0.4 |

---

## 6. Hướng dẫn cài đặt

### Yêu cầu

- Docker & Docker Compose >= 2.0
- Git (hoặc tải source dưới dạng ZIP)

### Bước 1 — Clone và cấu hình môi trường

```bash
git clone https://github.com/nguyenthanh2911/CNM.git
cd CNM

cp .env.example .env
# Chỉnh sửa .env nếu cần (DB password, MLflow URI, ...)
```

### Bước 2 — Khởi động toàn bộ hệ thống

```bash
# Khởi động tất cả services
docker compose up -d

# (tuỳ chọn) Kiểm tra trạng thái nhanh
docker compose ps
```

Windows (không dùng WSL/Git Bash):

```powershell
docker compose up -d
# (tuỳ chọn) Kiểm tra nhanh health endpoints
Invoke-WebRequest http://localhost:8001/health -UseBasicParsing | Select-Object -ExpandProperty Content
Invoke-WebRequest http://localhost:8002/health -UseBasicParsing | Select-Object -ExpandProperty Content
```

Services sẽ chạy tại:

| Service | URL |
|---------|-----|
| Django Dashboard | http://localhost:8000 |
| FastAPI ML Service | http://localhost:8001/docs |
| FastAPI Alert Service | http://localhost:8002/docs |
| MLflow UI | http://localhost:5000 |
| Grafana | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |

### Bước 3 — Khởi tạo database

Linux / macOS:
```bash
# Khởi tạo bảng trong database
cat docs/database_schema.sql | docker compose exec -T postgres psql -U sepsis_user -d sepsis_db

# Seed dữ liệu bệnh nhân ban đầu (chạy bên trong container)
docker compose exec -T ml_service python scripts/seed_patients.py
```

Windows (PowerShell):
```powershell
# Khởi tạo bảng trong database
Get-Content docs\database_schema.sql | docker compose exec -T postgres psql -U sepsis_user -d sepsis_db

# Seed dữ liệu bệnh nhân ban đầu
docker compose exec -T ml_service python scripts/seed_patients.py
```

### Bước 4 — Chạy demo với Synthetic Data

Toàn bộ các lệnh bên dưới chạy **bên trong Docker** (không cần cài Python trên máy).
Sau thay đổi mới nhất, `PYTHONPATH=/app` đã được cấu hình sẵn trong `docker-compose.yml`, nên không cần thêm `-e PYTHONPATH=/app`.

```bash
# Sinh dữ liệu synthetic (chạy trong container)
docker compose exec -T ml_service python -m data_pipeline.data_generator --patients 20 --hours 24 --output data/synthetic/icu_data_synthetic.csv

# Train model T+6h và đăng ký lên MLflow
docker compose exec -T ml_service python -m ml.train --data data/synthetic/icu_data_synthetic.csv --experiment-name "CNM-Sepsis-T6H" --model-name "sepsis_xgboost_t6h" --augment

# Stream dữ liệu vào hệ thống (chạy ngầm mỗi 30 giây để test)
# Mô phỏng 20 bệnh nhân ICU real-time (chạy ngầm, mỗi 10s gửi vitals)
docker compose exec -d ml_service python scripts/simulate_realtime.py

# Mở dashboard: http://localhost:8000
```

---

## 7. Hướng dẫn sử dụng

### Chạy toàn bộ demo một lệnh (Linux / Git Bash)

```bash
bash scripts/run_demo.sh
```

### Train mô hình (thủ công)

```bash
# Train model T+6h (khuyên dùng)
docker compose exec -T ml_service python -m ml.train \
    --data data/synthetic/icu_data_synthetic.csv \
    --experiment-name "CNM-Sepsis-T6H" \
    --model-name "sepsis_xgboost_t6h" \
    --augment

# Model sẽ tự động thêm label T+6h và log lên MLflow
# Xem kết quả trên MLflow UI: http://localhost:5000
```

### Mô phỏng dữ liệu real-time

```bash
# Chạy mô phỏng 20 bệnh nhân ICU trong 240 bước (mỗi bước 10 giây)
# 10 BN LOW (bình thường), 4 BN WARN (dần xấu), 6 BN HIGH (dần rất xấu)
docker compose exec -d ml_service python scripts/simulate_realtime.py
```

### Chạy test

```bash
# Toàn bộ test suite
docker compose exec -T ml_service pytest tests/ -v

# Chỉ unit test
docker compose exec -T ml_service pytest tests/unit/ -v
```

### Xoá toàn bộ dữ liệu (Reset database)

Lưu ý: Dashboard đang hiển thị dữ liệu từ bảng `predictions`, nên khi reset cần xoá cả `predictions` (ngoài `prediction_results`).

Linux / macOS:

```bash
docker compose exec -T postgres psql -U sepsis_user -d sepsis_db -c "TRUNCATE TABLE alerts, prediction_results, predictions, vital_records, admissions, patients CASCADE;"
```

Windows (PowerShell):

```powershell
docker compose exec -T postgres psql -U sepsis_user -d sepsis_db -c "TRUNCATE TABLE alerts, prediction_results, predictions, vital_records, admissions, patients CASCADE;"
```

Nếu bạn đang chạy `scripts/simulate_realtime.py` thì dữ liệu sẽ được ghi lại ngay sau khi xoá; hãy dừng tiến trình đó trước (`docker compose stop ml_service`), rồi hard refresh Dashboard (`Ctrl+F5`).

---

## 8. Quy trình ML Pipeline

```
Raw Data (Synthetic ICU)             EarlyWarningPredictor
    │                                      (rule-based)
    ▼                                         │
[1] Generation   data_generator.py            │
    - 40% sepsis, 60% non-sepsis              │
    - Confounders: bad spikes, recovery,      │
      equipment noise, missing labs           │
    - sepsis_onset_hour (random 8-18h)        │
    │                                         │
    ▼                                         │
[T+6h Labeling]   labeling.py                 │
    - create_t6h_labels() dựa trên            │
      sepsis_onset_hour                       │
    - sepsis_in_next_6h = 1 nếu onset         │
      trong (t, t+6h]                         │
    - Patient-based split (no leakage)        │
    │                                         │
    ▼                                         │
[2] Preprocessing (training path)             │
    sklearn Pipeline:                         │
    - SimpleImputer(strategy='median')        │
    - StandardScaler                          │
    - Lưu: artifacts/preprocessor_t6h.joblib  │
    │                                         │
    ├── (inference path: same Pipeline,       │
    │    load từ joblib, fallback fit-on-fly) │
    │                                         │
    ▼                                         │
[3] Feature Engineering   feature_builder.py  │
    - Rolling stats: mean, std, min, max      │
      (3/12/48 intervals ~ 15/60/240 phút)   │
    - Trend: diff(1) / interval_minutes       │
    - Clinical scores: SOFA, NEWS2, qSOFA     │
    - Time-since-last-abnormal-HR             │
    - Drop raw vitals/labs columns            │
    │                                         │
    ▼                                         │
[4] Training   train.py                       │
    - Label: sepsis_in_next_6h (T+6h)        │
    - Patient-based split (~60/20/20)        │
    - scale_pos_weight = neg/pos              │
    - 5-fold StratifiedKFold cross-validation │
    - Auto-regularize nếu std_auroc > 0.08    │
    - Auto SMOTE (ratio=0.4) nếu imbalance >5│
    - XGBoost: 150 est, max_depth=4, lr=0.05  │
      subsample=0.65, reg_lambda=3, gamma=2   │
    - Early stopping: 30 rounds               │
    - Log params + metrics + model → MLflow   │
    - Register model nếu test_auroc > 0.80    │
      và gap(train-test) < 0.10 → Production  │
    │                                         │
    ▼                                         │
[5] Evaluation   evaluate.py                  │
    - AUROC, F1 (thr=0.4), Sensitivity,       │
      Specificity, Confusion Matrix           │
    - ROC curve plot (PNG)                    │
    - Confusion matrix plot (PNG)             │
    │                                         │
    ▼                                         │
[6] Registry   mlflow_utils.py                │
    - log_training_run() → params + metrics + │
      xgboost model + feature_names.json      │
    - register_model() → Production/Staging   │
    - load_production_model_with_metadata()   │
    │                                         │
    ▼                                         │
[7] Serving   ml_service/predictor.py         │
    - Load model từ MLflow Registry           │
      (ưu tiên Production → Staging → latest) │
    - Predict: 11 raw features → pipeline     │
      → XGBoost predict_proba → risk_score    │
    - Clinical scores: SOFA, NEWS2 từ raw     │
    - SHAP TreeExplainer → top-5 features     │
    - EarlyWarningPredictor (30 phút cửa sổ)  │
    - Lưu đầy đủ: raw vitals + early_warning  │
      scores vào PostgreSQL predictions table │
    │                                         │
    ▼                                         │
[8] Monitoring + Retraining                   │
    - Evidently AI (DataDriftPreset) so sánh  │
      reference (train) vs current (24h)      │
    - Prefect retrain_flow:                   │
      drift_score > 0.7 → run_training()      │
      → compare_and_promote()                 │
      (new_auroc > production_auroc + 0.01)   │
    - Prometheus metrics:                     │
      predictions_total, inference_seconds,   │
      predictions_by_risk_total, active_alerts│
```

---

## 9. API Reference

### ML Service (FastAPI — port 8001)

#### `POST /vitals` — Nhận vitals và trả về dự đoán

**Request body** (`VitalRequest` — Pydantic validation):
- `patient_id` (str, required)
- `timestamp` (datetime, required)
- `heart_rate` (float, 20–250), `systolic_bp` (40–300), `diastolic_bp` (20–200)
- `temperature` (float, 30–45), `spo2` (50–100), `respiratory_rate` (4–60)
- `lactate`, `wbc`, `creatinine`, `bilirubin`, `platelet` (float, optional)

```json
// Response
{
  "patient_id": "P001",
  "timestamp": "2024-01-15T08:30:00Z",
  "risk_score": 0.82,
  "risk_level": "CRITICAL",          // LOW / WARNING / CRITICAL
  "alert_triggered": true,
  "shap_features": [
    {"feature": "heart_rate",         "shap_value": 0.31},
    {"feature": "spo2",               "shap_value": 0.24},
    {"feature": "lactate",            "shap_value": 0.19}
  ],
  "sofa_score": 6,
  "news2_score": 9,
  "inference_time_ms": 95.0,
  "early_warning": {
    "early_warning_probability": 0.85,   // 0.0 – 1.0
    "early_warning_level": "HIGH",        // LOW / MEDIUM / HIGH
    "time_window_minutes": 30,
    "trend_score": 0.42,
    "rate_of_change_score": 0.38,
    "threshold_score": 0.91,
    "contributing_factors": [
      "Nhịp tim cao (124 bpm)",
      "Huyết áp thấp (76 mmHg)",
      "SpO2 thấp (90.0%)",
      "Lactate cao (4.2 mmol/L)"
    ]
  }
}
```

#### `GET /vitals/{patient_id}/history` — Lịch sử vitals + SHAP + early warning gần nhất

```json
{
  "latest_vitals": { "heart_rate": 112, ... },
  "top_features": [ {"feature": "heart_rate", "shap_value": 0.31}, ... ],
  "early_warning": { "early_warning_probability": 0.85, ... }
}
```

#### `GET /health` — Kiểm tra trạng thái service

```json
{
  "status": "ok",
  "model_version": "1",
  "model_auroc": 0.91,
  "uptime_seconds": 86400.0
}
```

#### `GET /metrics` — Prometheus metrics

Các metrics exposed:
- `predictions_total` (Counter) — tổng số predictions
- `predictions_by_risk_total{risk_level="LOW|WARNING|CRITICAL"}` (Counter)
- `inference_seconds` (Histogram) — latency inference

---

### Alert Service (FastAPI — port 8002)

| Method | Endpoint | Mô tả | Query Params |
|--------|----------|-------|-------------|
| `POST` | `/alerts` | Tạo alert mới (gọi từ ML Service nội bộ) | — |
| `GET` | `/alerts` | Danh sách alert | `patient_id`, `status=pending\|confirmed\|all`, `limit=1..500` |
| `GET` | `/alerts/stats` | Thống kê hôm nay | — |
| `GET` | `/health` | Health check | — |
| `GET` | `/metrics` | Prometheus metrics (`active_alerts` Gauge) | — |

Alert được push real-time qua WebSocket tại `ws://localhost:8002/ws/alerts/{patient_id}`.

---

### Django Dashboard (port 8000)

| Route | View | Mô tả |
|-------|------|-------|
| `/` | `patient_list` | Danh sách ICU real-time (latest prediction mỗi patient) |
| `/patients/{patient_id}/` | `patient_detail` | Chart 24 predictions gần nhất + vitals + SHAP + early warning |
| `/alerts/` | `alerts_page` | Trang cảnh báo tập trung |
| `/alerts/{alert_id}/acknowledge/` | `acknowledge_alert` | Xác nhận đã xử lý alert |
| `/api/patient/{patient_id}/latest/` | `patient_latest_api` | JSON API cho polling |
| WebSocket `ws://localhost:8000/ws/alerts/` | `AlertConsumer` | Push alert real-time qua Django Channels |

---

## 10. Kết quả và đánh giá

### Kết quả mô hình (test set — Synthetic ICU, label T+6h)

| Metric | Kết quả | Mục tiêu |
|--------|---------|----------|
| AUROC | **0.8270** | > 0.85 |
| Sensitivity (Recall) | **79%** | > 75% |
| Specificity | **71%** | > 80% |
| F1-score (thr=0.4) | **0.26** | > 0.75 |
| Imbalance ratio | **9:1** (→ SMOTE ratio=0.4) | — |
| Positive label ratio | **~10%** | — |

### Kết quả hệ thống

| Metric | Kết quả | Mục tiêu |
|--------|---------|----------|
| Inference latency (p95) | ~95ms | < 200ms |
| End-to-end alert latency | ~2 phút | < 5 phút |
| Concurrent patients supported | ≥ 20 | ≥ 20 |

### Hạn chế & Hướng phát triển

**Hạn chế hiện tại:**
- Dữ liệu synthetic chưa phản ánh đầy đủ độ phức tạp của ICU thực tế
- Chỉ sử dụng XGBoost, chưa khai thác thông tin chuỗi thời gian dài hạn
- Chu kỳ dự đoán 5 phút, chưa hỗ trợ alert tức thời theo giây
- Specificity chưa đạt target (>80%) do label imbalance cao

**Hướng phát triển:**
- Tích hợp MIMIC-IV để train trên dữ liệu thực
- Bổ sung LSTM để khai thác time-series
- Rút ngắn chu kỳ dự đoán xuống dưới 1 phút khi có phần cứng phù hợp
- Cải thiện specificity qua threshold tuning hoặc cost-sensitive learning

---

# CHƯƠNG 1 — PHÂN TÍCH, THIẾT KẾ

## 1.1 Mô tả bài toán

**Bối cảnh:**
Sepsis (nhiễm khuẩn huyết) là một trong những nguyên nhân tử vong hàng đầu tại các đơn vị chăm sóc đặc biệt (ICU). Phát hiện sớm trong vòng 1–6 giờ đầu có thể tăng tỉ lệ sống sót lên đến 80%. Tuy nhiên, y tá và bác sĩ ICU phải đồng thời theo dõi hàng chục chỉ số sinh lý liên tục cho nhiều bệnh nhân, dẫn đến nguy cơ bỏ sót các dấu hiệu nguy hiểm.

**Bài toán cụ thể:**
Xây dựng hệ thống tự động thu thập dữ liệu sinh lý bệnh nhân ICU theo thời gian thực, dự đoán nguy cơ sepsis mỗi 5 phút bằng mô hình học máy, giải thích kết quả dự đoán và gửi cảnh báo tức thì đến nhân viên y tế khi phát hiện nguy cơ cao.

**Đầu vào:**
- Dấu hiệu sinh tồn: nhịp tim, huyết áp, nhiệt độ, SpO2, nhịp thở
- Kết quả xét nghiệm: lactate, WBC, creatinine, bilirubin, platelet
- Thông tin bệnh nhân: tuổi, giới tính, thời gian nhập viện

**Đầu ra:**
- Risk score ∈ [0, 1] cho từng bệnh nhân tại mỗi thời điểm
- Mức độ cảnh báo: LOW / WARNING / CRITICAL
- Top 5 đặc trưng ảnh hưởng nhất đến dự đoán (SHAP)
- Dashboard real-time và thông báo WebSocket

**Ràng buộc kỹ thuật:**
- Chu kỳ dự đoán: mỗi 5 phút
- Độ trễ cảnh báo end-to-end < 5 phút
- Cảnh báo sớm hơn thời điểm sepsis thực tế ít nhất 30 phút
- AUROC > 0.85, Sensitivity > 75%

---

## 1.2 Sơ đồ chức năng tổng quát

```
                    HỆ THỐNG ICU SEPSIS EARLY WARNING
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
   [1] QUẢN LÝ               [2] DỰ ĐOÁN &          [3] GIÁM SÁT &
   BỆNH NHÂN                 CẢNH BÁO               VẬN HÀNH
        │                       │                       │
   ┌────┴────┐             ┌────┴────┐             ┌────┴────┐
   │1.1 Thêm│             │2.1 Thu  │              │3.1 Theo │
   │bệnh    │             │thập     │              │dõi drift│
   │nhân    │             │vitals   │              │dữ liệu  │
   └────────┘             │(5 phút) │              └─────────┘
   ┌────────┐             └─────────┘              ┌─────────┐
   │1.2 Xem │             ┌─────────┐              │3.2 Tự   │
   │danh    │             │2.2 Dự   │              │động     │
   │sách    │             │đoán     │              │retrain  │
   └────────┘             │risk     │              └─────────┘
   ┌────────┐             │score    │              ┌─────────┐
   │1.3 Xem │             └─────────┘              │3.3 Theo │
   │lịch sử │             ┌─────────┐              │dõi hệ   │
   │alert   │             │2.3 Giải │              │thống    │
   └────────┘             │thích    │              │Grafana  │
                          │SHAP     │              └─────────┘
                          └─────────┘
                          ┌─────────┐
                          │2.4 Gửi  │
                          │cảnh báo │
                          │WebSocket│
                          └─────────┘
                          ┌─────────┐
                          │2.5 Xác  │
                          │nhận     │
                          │alert    │
                          └─────────┘
```

---

## 1.3 Biểu đồ trường hợp sử dụng (Use Case)

```
                        ┌──────────────────────────────────────────┐
                        │          ICU Sepsis System               │
                        │                                          │
  ┌──────────┐          │  ┌────────────────────────────────────┐  │
  │          │──────────┼─►│ UC01: Xem danh sách bệnh nhân      │  │
  │          │          │  └────────────────────────────────────┘  │
  │   Y tá   │          │  ┌────────────────────────────────────┐  │
  │  (Nurse) │──────────┼─►│ UC02: Xem risk score real-time     │  │
  │          │          │  └────────────────────────────────────┘  │
  │          │          │  ┌────────────────────────────────────┐  │
  └──────────┘          │  │ UC03: Nhận cảnh báo CRITICAL       │  │
        │               │  └────────────────────────────────────┘  │
        ├───────────────┼─►                                        │
        │               │  ┌────────────────────────────────────┐  │
        │               │  │ UC04: Xem giải thích SHAP          │  │
        ├───────────────┼─►│                                    │  │
        │               │  └────────────────────────────────────┘  │
        │               │  ┌────────────────────────────────────┐  │
        │               │  │ UC05: Acknowledge alert            │  │
        └───────────────┼─►│                                    │  │
                        │  └────────────────────────────────────┘  │
                        │                                          │
  ┌──────────┐          │  ┌────────────────────────────────────┐  │
  │  Bác sĩ /│──────────┼─►│ UC06: Train / Retrain model        │  │
  │  Admin   │          │  └────────────────────────────────────┘  │
  └──────────┘          │  ┌────────────────────────────────────┐  │
        │               │  │ UC07: Xem báo cáo & metrics        │  │
        └───────────────┼─►│                                    │  │
                        │  └────────────────────────────────────┘  │
                        │                                          │
  ┌──────────┐          │  ┌────────────────────────────────────┐  │
  │ Simulator│──────────┼─►│ UC08: Gửi vitals tự động (5 phút)  │  │
  └──────────┘          │  └────────────────────────────────────┘  │
                        │  ┌────────────────────────────────────┐  │
                        │  │ UC09: Theo dõi hệ thống (Grafana)  │  │
                        │  └────────────────────────────────────┘  │
                        └──────────────────────────────────────────┘
```

---

## 1.4 Biểu đồ hoạt động

### Luồng chính: Thu thập vitals → Dự đoán → Cảnh báo (chu kỳ 5 phút)

```
  Simulator                 FastAPI ML Service              Django Dashboard
      │                            │                               │
      │   POST /vitals             │                               │
      │   (mỗi 5 phút)             │                               │
      │───────────────────────────►│                               │
      │                            │                               │
      │                     ┌──────▼──────┐                        │
      │                     │ Validate    │                        │
      │                     │ input data  │                        │
      │                     └──────┬──────┘                        │
      │                            │                               │
      │                     ┌──────▼──────┐                        │
      │                     │ Tiền xử lý  │                        │
      │                     │ imputation  │                        │
      │                     │ normalize   │                        │
      │                     └──────┬──────┘                        │
      │                            │                               │
      │                     ┌──────▼──────┐                        │
      │                     │ Feature Eng │                        │
      │                     │ rolling stats│                       │
      │                     │ SOFA, NEWS2 │                        │
      │                     └──────┬──────┘                        │
      │                            │                               │
      │                     ┌──────▼──────┐                        │
      │                     │ XGBoost     │                        │
      │                     │ predict     │                        │
      │                     │ risk score  │                        │
      │                     └──────┬──────┘                        │
      │                            │                               │
      │                     ┌──────▼──────┐                        │
      │                     │ SHAP explain│                        │
      │                     │ top-5       │                        │
      │                     └──────┬──────┘                        │
      │                            │                               │
      │                     ┌──────▼──────┐                        │
      │                     │ Lưu DB      │                        │
      │                     │ kiểm ngưỡng │                        │
      │                     └──────┬──────┘                        │
      │                            │                               │
      │              ┌─────────────┼─────────────┐                 │
      │           <0.3          0.3-0.7         ≥0.7               │
      │              │              │              │               │
      │           Log only    Warning UI    ┌──────▼──────┐        │
      │                             │       │ WebSocket   │        │
      │                             │       │ CRITICAL    │───────►│
      │                             │       │ push        │        │
      │                             │       └─────────────┘        │
      │                             │                        ┌─────▼─────┐
      │                             └───────────────────────►│ Hiển thị  │
      │                                                      │ alert +   │
      │                                                      │ SHAP chart│
      │◄──────────── response ─────────────────              └─────┬─────┘
      │   {risk_score, top_features}                               │
                                                             Y tá acknowledge
```

---

## 1.5 Biểu đồ trình tự (Sequence Diagram)

### UC02 + UC03: Nhận vitals → Dự đoán → Cảnh báo

```
Simulator   FastAPI_ML   PostgreSQL   AlertService   Django_WS    Nurse
    │             │             │             │             │        │
    │─POST/vitals►│             │             │             │        │
    │             │─validate───►│             │             │        │
    │             │◄─ok─────────│             │             │        │
    │             │             │             │             │        │
    │             │ preprocess()│             │             │        │
    │             │ feature_eng()             │             │        │
    │             │ xgb.predict()             │             │        │
    │             │ shap.explain()            │             │        │
    │             │             │             │             │        │
    │             │─INSERT risk►│             │             │        │
    │             │◄─saved──────│             │             │        │
    │             │             │             │             │        │
    │             │ [risk ≥ 0.7]│             │             │        │
    │             │─POST /alert───────────── ►│             │        │
    │             │             │             │─WS push─────►        │
    │             │             │             │             │─notify►│
    │             │             │             │             │        │
    │◄─response───│             │             │             │        │
    │  {score,    │             │             │             │        │
    │   features} │             │             │             │        │
    │             │             │             │  Nurse acknowledge──►│
    │             │             │             │◄─PATCH /alerts/{id}──│
    │             │             │─UPDATE─────►│             │        │
```

---

## 1.6 Biểu đồ Lớp (Class Diagram)

```
┌─────────────────────┐        ┌─────────────────────┐
│       Patient       │        │      Admission      │
├─────────────────────┤        ├─────────────────────┤
│- patient_id: str    │1──────►│- admission_id: str  │
│- name: str          │        │- patient_id: str    │
│- age: int           │        │- admitted_at: dt    │
│- gender: str        │        │- discharged_at: dt  │
│- ward: str          │        │- bed_number: str    │
├─────────────────────┤        └─────────────────────┘
│+ get_latest_risk()  │                  │ 1
│+ get_alert_history()│                  │
└─────────────────────┘                  ▼ *
                                ┌─────────────────────┐
                                │     VitalRecord     │
                                ├─────────────────────┤
                                │- record_id: str     │
                                │- patient_id: str    │
                                │- timestamp: dt      │
                                │- heart_rate: float  │
                                │- systolic_bp: float │
                                │- diastolic_bp: float│
                                │- temperature: float │
                                │- spo2: float        │
                                │- resp_rate: float   │
                                ├─────────────────────┤
                                │+ to_feature_vector()│
                                └─────────────────────┘
                                          │ 1
                                          ▼ 1
┌─────────────────────┐        ┌─────────────────────┐
│   SepsisPredictor   │        │   PredictionResult  │
├─────────────────────┤        ├─────────────────────┤
│- model: XGBModel    │        │- result_id: str     │
│- explainer: SHAP    │        │- patient_id: str    │
│- scaler: Scaler     │        │- timestamp: dt      │
├─────────────────────┤        │- risk_score: float  │
│+ predict(vitals)    │───────►│- risk_level: str    │
│+ explain(features)  │        │- sofa_score: int    │
│+ preprocess(data)   │        │- news2_score: int   │
└─────────────────────┘        │- top_features: list │
                               ├─────────────────────┤
                               │+ is_critical(): bool│
                               └─────────────────────┘
                                          │ 1
                                          ▼ 0..1
┌─────────────────────┐        ┌─────────────────────┐
│  WebSocketManager   │        │        Alert        │
├─────────────────────┤        ├─────────────────────┤
│- connections: dict  │◄───────│- alert_id: str      │
├─────────────────────┤        │- patient_id: str    │
│+ send(patient, msg) │        │- result_id: str     │
│+ broadcast(msg)     │        │- severity: str      │
│+ register(ws)       │        │- created_at: dt     │
└─────────────────────┘        │- acknowledged: bool │
                               │- ack_by: str        │
                               ├─────────────────────┤
                               │+ acknowledge(user)  │
                               └─────────────────────┘
```

---

## 1.7 Biểu đồ luồng dữ liệu (Data Flow Diagram)

### Mức 0 — Context Diagram

```
                       vitals (5 phút/lần)
  ┌──────────┐  ─────────────────────────► ┌──────────────────────┐
  │Simulator │                             │                      │ risk score + alert
  │/ Device  │◄─────────────────────────── │   ICU SEPSIS SYSTEM  │──────────────────► ┌────────┐
  └──────────┘        response             │                      │                     │ Nurse /│
                                           │                      │◄──────────────────  │ Doctor │
  ┌──────────┐   login / actions           │                      │    acknowledge       └────────┘
  │  Admin   │ ──────────────────────────► │                      │
  └──────────┘                             └──────────────────────┘
```

### Mức 1 — DFD chi tiết

```
                        ┌──────────────┐
  vitals                │    1.0       │  features
 ──────────────────────►│  Preprocess  ├────────────────────────►
                        │  & Feature   │
                        │  Engineer    │
                        └──────────────┘
                                              ┌──────────────┐
                               features       │    2.0       │  risk_score
                            ─────────────────►│  XGBoost     ├──────────────►
                                              │  Predict     │
                                              └──────────────┘
                                                     │
                                         ┌───────────▼──────────┐
                                         │         D1           │
                                         │  prediction_results  │
                                         │   (PostgreSQL)       │
                                         └───────────┬──────────┘
                                                     │
                                    ┌────────────────▼────────────────┐
  risk_score                        │         3.0 Alert Decision      │
 ─────────────────────────────────► │  score ≥ 0.7 → trigger alert   │
                                    └────────────────┬────────────────┘
                                                     │
                                         ┌───────────▼──────────┐
                                         │         D2           │
                                         │      alerts          │
                                         │   (PostgreSQL)       │
                                         └───────────┬──────────┘
                                                     │
                                    ┌────────────────▼────────────────┐
                                    │    4.0 WebSocket Push           │
                                    │    → Django Dashboard → Nurse   │
                                    └─────────────────────────────────┘
```

---

## 1.8 Biểu đồ mối quan hệ dữ liệu (ERD)

```
┌─────────────┐       ┌──────────────────┐       ┌──────────────────┐
│   patients  │       │    admissions    │       │  vital_records   │
├─────────────┤       ├──────────────────┤       ├──────────────────┤
│PK patient_id│1─────►│PK admission_id   │1─────►│PK record_id      │
│   name      │       │FK patient_id     │       │FK patient_id     │
│   age       │       │   admitted_at    │       │   timestamp      │
│   gender    │       │   discharged_at  │       │   heart_rate     │
│   ward      │       │   bed_number     │       │   systolic_bp    │
│   created_at│       │   status         │       │   diastolic_bp   │
└─────────────┘       └──────────────────┘       │   temperature    │
                                                  │   spo2           │
                                                  │   resp_rate      │
                                                  └────────┬─────────┘
                                                           │ 1
                                                           ▼ 1
┌─────────────┐       ┌──────────────────┐       ┌──────────────────┐
│    alerts   │       │  alert_features  │       │prediction_results│
├─────────────┤       ├──────────────────┤       ├──────────────────┤
│PK alert_id  │◄──────│FK alert_id       │       │PK result_id      │
│FK result_id │1      │   feature_name   │       │FK record_id      │
│FK patient_id│       │   shap_value     │       │FK patient_id     │
│   severity  │       │   rank           │       │   risk_score     │
│   created_at│       └──────────────────┘       │   risk_level     │
│   ack_by    │                                   │   sofa_score     │
│   ack_at    │◄──────────────────────────────────│   news2_score    │
│   status    │                            1      │   infer_ms       │
└─────────────┘                                   │   created_at     │
                                                  └──────────────────┘
```

---

## 1.9 Thiết kế giao diện

### Giao diện 1 — Dashboard chính (danh sách bệnh nhân)

```
┌──────────────────────────────────────────────────────────────────┐
│  🏥 ICU Sepsis Early Warning          [🔔 3 alerts]  [Admin ▼]  │
├──────────────────────────────────────────────────────────────────┤
│  TỔNG QUAN                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │    20    │  │    3     │  │    5     │  │    12    │        │
│  │ Bệnh nhân│  │ CRITICAL │  │ WARNING  │  │  STABLE  │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
├──────────────────────────────────────────────────────────────────┤
│  DANH SÁCH BỆNH NHÂN                        [🔍 Tìm kiếm...]    │
│  ┌──────┬──────────┬──────┬────────────┬──────────┬──────────┐  │
│  │  ID  │   Tên    │Phòng │ Risk Score │  Mức độ  │  Action  │  │
│  ├──────┼──────────┼──────┼────────────┼──────────┼──────────┤  │
│  │ P001 │ Nguyễn A │ ICU1 │ ████░ 0.82 │🔴CRITICAL│  [Xem]  │  │
│  │ P002 │ Trần B   │ ICU2 │ ███░░ 0.61 │🟡WARNING │  [Xem]  │  │
│  │ P003 │ Lê C     │ ICU1 │ █░░░░ 0.21 │🟢 STABLE │  [Xem]  │  │
│  └──────┴──────────┴──────┴────────────┴──────────┴──────────┘  │
│  Cập nhật lần cuối: 08:35:00  (chu kỳ 5 phút)                   │
└──────────────────────────────────────────────────────────────────┘
```

### Giao diện 2 — Chi tiết bệnh nhân (real-time)

```
┌──────────────────────────────────────────────────────────────────┐
│  ← Quay lại     BN: Nguyễn Văn A — ICU-1 — Giường 3            │
├─────────────────────────┬────────────────────────────────────────┤
│  RISK SCORE HIỆN TẠI    │  VITALS HIỆN TẠI                       │
│                         │  ┌─────────┬──────────┬─────────────┐  │
│       🔴  0.82          │  │HR: 112  │BP: 88/54 │  Temp: 39.1 │  │
│       CRITICAL          │  │SpO2: 93%│RR: 24   │             │  │
│  [Acknowledge Alert]    │  └─────────┴──────────┴─────────────┘  │
│                         │                                        │
│  SOFA: 6  |  NEWS2: 9   │  BIỂU ĐỒ RISK SCORE (2 giờ gần nhất) │
│                         │  1.0│              ╭──╮               │
│  TOP FEATURES (SHAP)    │  0.7│─ ─ ─ ─ ─ ─╯  ╰──             │
│  lactate_trend  ████    │  0.3│                                 │
│  spo2_min_60m   ███     │  0.0└─────────────────────────        │
│  hr_mean_15m    ██      │     -120m   -60m    -30m    now       │
│  resp_trend     ██      │                                        │
│  temp_max_60m   █       │  Cập nhật lần tới: ~5 phút            │
└─────────────────────────┴────────────────────────────────────────┘
```

### Giao diện 3 — Quản lý Alert

```
┌──────────────────────────────────────────────────────────────────┐
│  QUẢN LÝ CẢNH BÁO                   [Tất cả ▼]  [Hôm nay ▼]   │
├──────┬──────────┬───────────┬──────────┬────────────┬──────────┤
│  ID  │ Bệnh nhân│  Thời gian│ Severity │ Trạng thái │  Action  │
├──────┼──────────┼───────────┼──────────┼────────────┼──────────┤
│ A001 │   P001   │ 08:32:00  │ CRITICAL │ ⏳ Pending  │[Confirm]│
│ A002 │   P005   │ 07:45:00  │ WARNING  │ ✅ Confirmed│  [Xem]  │
│ A003 │   P012   │ 06:10:00  │ CRITICAL │ ✅ Confirmed│  [Xem]  │
└──────┴──────────┴───────────┴──────────┴────────────┴──────────┘
```

---

## 1.10 Thiết kế giải thuật

### Tổng quan mô hình đề xuất

Hệ thống sử dụng **XGBoost Classifier** kết hợp **SHAP TreeExplainer** để vừa dự đoán chính xác vừa giải thích được kết quả cho nhân viên y tế. Mỗi 5 phút, simulator gửi một bản ghi vitals mới; hệ thống tích lũy lịch sử và tính các đặc trưng cửa sổ thời gian trước khi đưa vào model.

**Lý do chọn XGBoost:**
- Xử lý tốt dữ liệu tabular với missing values
- Hỗ trợ `scale_pos_weight` phù hợp bài toán imbalanced (sepsis ~10–15%)
- Inference nhanh (< 100ms), phù hợp chu kỳ 5 phút
- Tương thích hoàn toàn với SHAP TreeExplainer

### Rút trích đặc trưng (Feature Engineering)

**Nhóm 1 — Rolling statistics** (cửa sổ 15 phút / 60 phút / 4 giờ):

```
Với mỗi vital signal x và cửa sổ thời gian w:
  mean_w    = avg(x trong w gần nhất)
  std_w     = std(x trong w gần nhất)
  min_w     = min(x trong w gần nhất)
  max_w     = max(x trong w gần nhất)
  trend_15m = (x_t - x_{t-15m}) / 15    # đạo hàm bậc 1
```

**Nhóm 2 — Clinical scores:**

```
SOFA score (0–24):
  Respiratory   : PaO2/FiO2 ratio
  Coagulation   : Platelet count
  Liver         : Bilirubin
  Cardiovascular: MAP
  Renal         : Creatinine

NEWS2 score (0–20):
  RR + SpO2 + temperature + BP + HR + consciousness

qSOFA (0–3):
  RR ≥ 22   → +1
  SBP ≤ 100 → +1
  GCS < 15  → +1
```

**Nhóm 3 — Time-since-abnormal:**

```
Với mỗi chỉ số có ngưỡng lâm sàng:
  time_since_abnormal = t_now - t_last_abnormal  (tính bằng phút)
```

**Tổng số features: ~85 features**

### Giải thuật học (XGBoost)

```
Input : Feature vector X (85 chiều), label y ∈ {0, 1}
Output: Mô hình F(x) = P(sepsis=1 | x)

Hyperparameters:
  n_estimators      = 300
  max_depth         = 6
  learning_rate     = 0.05
  subsample         = 0.8
  colsample_bytree  = 0.8
  scale_pos_weight  = n_negative / n_positive   # xử lý imbalance
  eval_metric       = ['auc', 'logloss']
  early_stopping    = 20 rounds

Training process:
  1. Chronological split (tránh data leakage theo thời gian)
  2. StandardScaler fit trên train, transform val/test
  3. xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
  4. Chọn best iteration theo val AUROC
  5. Log toàn bộ params + metrics + model lên MLflow

Threshold tuning (trên val set):
  Mặc định  : 0.5
  Thực tế   : tune để đạt Sensitivity > 75%
  Thường dùng: threshold ≈ 0.35–0.45
```

### Giải thích bằng SHAP

```python
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_instance)

# Top-5 features có |shap_value| lớn nhất
top_features = sorted(
    zip(feature_names, shap_values),
    key=lambda x: abs(x[1]),
    reverse=True
)[:5]
```

---

## 1.11 Thiết kế Test

### Chiến lược kiểm thử

| Loại test | Công cụ | Phạm vi |
|-----------|---------|---------|
| Unit test | pytest | Từng hàm / class riêng lẻ |
| Integration test | pytest + TestClient | API endpoints |
| Model test | pytest | Metrics ngưỡng tối thiểu |
| Load test | locust | Hiệu năng đồng thời |

### Unit Test

```
tests/unit/
├── test_features.py
│     - test_sofa_score_calculation()       # giá trị biên, chuẩn lâm sàng
│     - test_news2_score_calculation()
│     - test_rolling_mean_15m()
│     - test_forward_fill_imputation()
│     - test_outlier_removal_iqr()
│
├── test_model.py
│     - test_model_output_range_0_1()       # risk score ∈ [0,1]
│     - test_model_auroc_above_threshold()  # AUROC > 0.80 trên test set
│     - test_shap_top5_features_returned()
│     - test_inference_time_under_200ms()
│
└── test_api.py
      - test_post_vitals_valid_input()       # HTTP 200 + đúng schema
      - test_post_vitals_missing_field()     # HTTP 422
      - test_post_vitals_out_of_range()      # HTTP 422
      - test_get_health_returns_healthy()
      - test_alert_acknowledge_flow()
```

### Integration Test

```
tests/integration/
└── test_pipeline.py
      - test_end_to_end_low_risk()           # score < 0.3 → không tạo alert
      - test_end_to_end_critical_risk()      # score ≥ 0.7 → tạo alert + WS push
      - test_drift_detector_trigger()        # Evidently phát hiện drift
      - test_retrain_flow_execution()        # Prefect flow chạy đến hết
```

### CI/CD Test Pipeline (GitHub Actions)

```yaml
on: [push, pull_request]

jobs:
  test:
    steps:
      - pytest tests/unit/ -v --cov=.
      - pytest tests/integration/ -v
      - Check coverage >= 70%
      - Build Docker image
      - docker compose up → bash scripts/check_health.sh
```

---

# CHƯƠNG 2 — HIỆN THỰC

## 2.1 Công nghệ sử dụng

**Frontend:** Django Templates + Bootstrap 5 + Chart.js cho dashboard real-time; Django Channels (ASGI) xử lý WebSocket để đẩy cảnh báo tức thì đến trình duyệt mà không cần reload trang.

**Dữ liệu:** DuckDB để truy vấn nhanh file Parquet trong quá trình training; PostgreSQL lưu trữ bệnh nhân, vitals và alerts ở production; Pandas + NumPy xử lý feature engineering.

**Học máy:** XGBoost làm mô hình chính; Scikit-learn cho preprocessing (StandardScaler, KNN imputation); SHAP cho giải thích kết quả; MLflow tracking thí nghiệm và quản lý Model Registry.

**Framework:** FastAPI (inference API + alert API); Django (web dashboard); Prefect (orchestration pipeline retrain); Evidently AI (data drift detection).

**Các thư viện khác:** Docker + Docker Compose (containerization); GitHub Actions (CI/CD); Prometheus + Grafana (monitoring hệ thống); pytest (kiểm thử).

---

## 2.2 Kết quả đạt được

### 2.2.1 Chức năng sinh dữ liệu và huấn luyện mô hình

Hệ thống sinh dữ liệu synthetic mô phỏng 20 bệnh nhân ICU trong 24 giờ với đầy đủ vitals và lab values theo chu kỳ 5 phút. Pipeline feature engineering tính toán ~85 đặc trưng bao gồm rolling statistics (15 phút / 60 phút / 4 giờ), SOFA score, NEWS2 score và qSOFA. Mô hình XGBoost được huấn luyện và toàn bộ thí nghiệm (params, metrics, artifacts) được ghi lại tự động trên MLflow UI tại `http://localhost:5000`. Model tốt nhất được promote lên Production stage trong MLflow Registry.

*(Chèn ảnh: MLflow experiment list, training metrics, model registry)*

### 2.2.2 Chức năng dự đoán real-time (FastAPI ML Service)

FastAPI ML Service tại `http://localhost:8001` nhận vitals qua `POST /vitals` mỗi 5 phút, thực hiện preprocessing → feature engineering → XGBoost inference → SHAP explanation trong một pipeline liên tục. Kết quả trả về gồm risk score, risk level, top-5 SHAP features, SOFA/NEWS2 score và inference time. Endpoint `GET /health` trả về trạng thái model và AUROC hiện tại. Toàn bộ API được tài liệu hóa tự động tại `/docs`.

*(Chèn ảnh: Swagger UI /docs, response mẫu với risk score và SHAP features)*

### 2.2.3 Chức năng dashboard và cảnh báo real-time (Django)

Dashboard Django tại `http://localhost:8000` hiển thị danh sách bệnh nhân với risk score cập nhật mỗi 5 phút, màu sắc phân biệt theo mức độ (xanh / vàng / đỏ). Trang chi tiết bệnh nhân hiển thị biểu đồ risk score 2 giờ gần nhất, vitals hiện tại và top-5 SHAP features dạng bar chart. Khi risk score ≥ 0.7, hệ thống tự động đẩy cảnh báo CRITICAL qua WebSocket đến tất cả client đang kết nối mà không cần reload trang. Y tá có thể acknowledge alert trực tiếp từ dashboard, trạng thái cập nhật realtime cho toàn bộ người dùng

*(Chèn ảnh: trang danh sách bệnh nhân, trang chi tiết, popup cảnh báo CRITICAL)*

### 2.2.4 Chức năng monitoring và tự động retrain

Evidently AI chạy định kỳ so sánh phân phối dữ liệu mới với reference data, tính drift score theo PSI. Nếu drift score > 0.7, Prefect tự động trigger flow retrain: load dữ liệu mới → feature engineering → train XGBoost → evaluate → nếu AUROC cao hơn model hiện tại thì promote lên Production, ngược lại giữ model cũ. Prometheus thu thập metrics từ FastAPI (request count, latency, prediction distribution) và Grafana hiển thị dashboard hệ thống tại `http://localhost:3000`.

*(Chèn ảnh: Grafana dashboard metrics, Prefect flow run history, Evidently drift report)*
