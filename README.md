# ICU Sepsis Early Warning System

![CI](https://github.com/nguyenthanh2911/CNM/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![MLflow](https://img.shields.io/badge/tracking-MLflow-blue)
![Docker](https://img.shields.io/badge/deploy-Docker-blue)

> Hệ thống theo dõi ICU real-time sử dụng Machine Learning nhằm cảnh báo sớm Sepsis
> Đồ án môn học | Khoa Công nghệ Thông tin

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
┌──────────────────────────────────────────────────────────────────────┐
│                         DATA & TRAINING                              │
│                                                                      │
│  data_generator.py ──► CSV / Parquet ──► feature_builder.py        │
│  (synthetic ICU)        (DuckDB)         (SOFA, NEWS2, rolling)     │
│                                     ──► train.py ──► MLflow Track   │
│                                                  ──► MLflow Registry│
└──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      SERVING & DEPLOYMENT                            │
│                                                                      │
│  GitHub Actions ──► pytest ──► Build Image ──► Docker Compose      │
│                                                                      │
│  FastAPI (port 8001)                                                 │
│    POST /vitals ──► XGBoost predict ──► SHAP ──► risk score        │
│    GET  /health                                                      │
│    ──► Request logging ──► Prometheus ──► Grafana                   │
│                                                                      │
│  Django (port 8000) ──► Dashboard real-time (WebSocket)            │
└──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    MONITORING & RETRAINING                           │
│                                                                      │
│  Evidently AI ──► drift score                                        │
│                      │                                               │
│               weight > 0.7 ?                                         │
│              /               \                                       │
│        Prefect flow         No retrain                               │
│     (retrain + register)                                             │
│            │                                                         │
│     Decision on Production Rules                                     │
│        Pass ──► Promote to Production                               │
│        Fail ──► Keep old model                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Luồng dữ liệu chính

```
Simulator (data_generator.py)
      │
      ▼ POST /vitals (mỗi 5 phút)
FastAPI ML Service
  ├── Tiền xử lý (imputation, chuẩn hóa)
  ├── Feature Engineering (rolling stats, SOFA, NEWS2)
  ├── XGBoost predict → risk score
  ├── SHAP explain (top-5 features)
  └── Lưu risk score → PostgreSQL
              │
              ├── score < 0.3  → log only
              ├── 0.3 – 0.7    → Dashboard warning
              └── ≥ 0.7        → CRITICAL: WebSocket push
                        │
                        ▼
              Django Dashboard (real-time WebSocket)
              Y tá / Bác sĩ ICU
```

---

## 3. Cấu trúc thư mục

```
CNM/
├── README.md
├── docker-compose.yml
├── .env.example
├── requirements.txt
│
├── docs/
│   ├── architecture.md
│   ├── api_spec.yaml
│   ├── database_schema.sql
│   └── diagrams/
│       ├── usecase.png
│       ├── activity.png
│       ├── sequence.png
│       ├── class_diagram.png
│       └── erd.png
│
├── data/
│   ├── raw/
│   │   └── (không commit — dữ liệu thô)
│   ├── processed/
│   │   ├── features_train.parquet
│   │   ├── features_val.parquet
│   │   └── features_test.parquet
│   └── synthetic/
│       └── icu_data_synthetic.csv
│
├── data_pipeline/
│   ├── __init__.py
│   ├── data_generator.py        # Sinh dữ liệu synthetic ICU
│   │                            #   - PhysiologicalModel: mô phỏng vitals
│   │                            #   - ICUSepsisGenerator: CSV / stream mode
│   └── preprocessor.py          # Imputation (forward-fill, KNN) + normalize
│
├── feature_engineering/
│   ├── __init__.py
│   ├── vitals_features.py       # Rolling mean/std/min/max, trend
│   ├── lab_features.py          # Tốc độ thay đổi lactate, WBC, flag bất thường
│   ├── clinical_scores.py       # SOFA, NEWS2, qSOFA
│   └── feature_builder.py       # Pipeline tổng hợp toàn bộ features
│
├── ml/
│   ├── __init__.py
│   ├── models/
│   │   └── xgboost_model.py     # XGBoost classifier (class_weight balanced)
│   ├── train.py                 # Load parquet → train → log MLflow
│   ├── evaluate.py              # AUROC, F1, Sensitivity, ROC curve
│   ├── explain.py               # SHAP TreeExplainer, top-5 features
│   └── mlflow_utils.py          # Log params/metrics, Model Registry
│
├── services/
│   ├── ml_service/              # FastAPI — Inference Engine (port 8001)
│   │   ├── main.py
│   │   ├── predictor.py         # Load model từ MLflow Registry, predict
│   │   └── schemas.py           # Pydantic request/response schemas
│   └── alert_service/           # FastAPI — Alert Manager (port 8002)
│       ├── main.py
│       └── websocket_manager.py # WebSocket push khi risk ≥ 0.7
│
├── web/                         # Django Dashboard (port 8000)
│   ├── manage.py
│   ├── dashboard/
│   │   ├── views.py
│   │   ├── consumers.py         # Django Channels WebSocket consumer
│   │   └── templates/
│   │       └── dashboard.html
│   └── config/
│       └── settings.py
│
├── monitoring/
│   ├── drift_detector.py        # Evidently AI — phát hiện data drift
│   ├── retrain_flow.py          # Prefect flow — trigger retrain khi drift > 0.7
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       └── dashboards/
│           └── icu_dashboard.json
│
├── tests/
│   ├── unit/
│   │   ├── test_features.py
│   │   ├── test_model.py
│   │   └── test_api.py
│   └── integration/
│       └── test_pipeline.py
│
└── scripts/
    ├── setup_db.sh
    ├── seed_patients.py
    ├── check_health.sh
    └── run_demo.sh
```

---

## 4. Công nghệ sử dụng

| Tầng | Công nghệ |
|------|-----------|
| Data & Feature | DuckDB, Pandas, Parquet |
| Machine Learning | XGBoost, SHAP, Scikit-learn |
| Experiment Tracking | MLflow |
| Backend API | FastAPI, Pydantic |
| Web Dashboard | Django, Django Channels (WebSocket) |
| Database | PostgreSQL |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions, pytest |
| Monitoring | Evidently AI, Prefect, Prometheus, Grafana |

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

Sử dụng **Sepsis-3 criteria**: bệnh nhân được gán nhãn `sepsis=1` khi SOFA ≥ 2 kèm nghi ngờ nhiễm trùng trong cửa sổ thời gian tương ứng.

### Phân chia tập dữ liệu

| Tập | Tỉ lệ | Ghi chú |
|-----|-------|---------|
| Train | 70% | Chronological split |
| Validation | 15% | Tune threshold |
| Test | 15% | Đánh giá cuối cùng |

---

## 6. Hướng dẫn cài đặt

### Yêu cầu

- Docker & Docker Compose >= 2.0
- Python >= 3.10
- Git

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

# Kiểm tra trạng thái
bash scripts/check_health.sh
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

```bash
bash scripts/setup_db.sh
python scripts/seed_patients.py
```

### Bước 4 — Chạy demo với Synthetic Data

```bash
# Sinh dữ liệu synthetic
python data_pipeline/data_generator.py --mode csv --patients 20 --hours 24

# Train model
python ml/train.py --data data/synthetic/icu_data_synthetic.csv

# Stream dữ liệu vào hệ thống (mỗi 5 phút 1 lần)
python data_pipeline/data_generator.py --mode stream --interval 300

# Mở dashboard: http://localhost:8000
```

---

## 7. Hướng dẫn sử dụng

### Chạy toàn bộ demo một lệnh

```bash
bash scripts/run_demo.sh
```

### Train mô hình

```bash
python ml/train.py \
    --data data/synthetic/icu_data_synthetic.csv \
    --experiment-name "sepsis_v1" \
    --model-name "sepsis_xgboost"

# Xem kết quả trên MLflow UI
open http://localhost:5000
```

### Evaluate và xuất báo cáo

```bash
python ml/evaluate.py \
    --model-version 1 \
    --test-data data/processed/features_test.parquet \
    --output reports/evaluation_v1/
```

### Chạy test

```bash
# Toàn bộ test suite
pytest tests/ -v

# Chỉ unit test
pytest tests/unit/ -v
```

---

## 8. Quy trình ML Pipeline

```
Raw Data (Synthetic ICU)
    │
    ▼
[1] Generation   data_generator.py
    - Mô phỏng vitals + labs theo PhysiologicalModel
    - Gán Sepsis-3 label tự động
    │
    ▼
[2] Preprocessing   preprocessor.py
    - Forward-fill missing vitals (< 10 phút gap)
    - KNN imputation cho lab values
    - Loại outlier thiết bị (IQR method)
    - StandardScaler normalize
    │
    ▼
[3] Feature Engineering   feature_builder.py
    - Rolling stats: mean, std, min, max (15 phút / 60 phút / 4h)
    - Trend: gradient 15 phút
    - Clinical scores: SOFA, NEWS2, qSOFA
    - Time-since-abnormal features
    │
    ▼
[4] Training   train.py
    - Chronological split train / val / test
    - class_weight='balanced' xử lý imbalance
    - Train XGBoost
    - Log params + metrics + model lên MLflow
    │
    ▼
[5] Evaluation   evaluate.py
    - AUROC, AUPRC, F1, Sensitivity, Specificity
    - Calibration plot
    - Alert lead time analysis
    │
    ▼
[6] Registry   mlflow_utils.py
    - Promote best model → Production stage
    - Versioning tự động
    │
    ▼
[7] Serving   ml_service/predictor.py
    - Load model từ MLflow Registry
    - Real-time inference mỗi 5 phút
    │
    ▼
[8] Monitoring + Retraining
    - Evidently AI phát hiện drift định kỳ
    - Prefect trigger retrain nếu drift score > 0.7
    - Model mới được promote nếu AUROC cao hơn
```

---

## 9. API Reference

### ML Service (FastAPI — port 8001)

#### `POST /vitals` — Nhận và dự đoán

```json
// Request
{
  "patient_id": "P001",
  "timestamp": "2024-01-15T08:30:00",
  "heart_rate": 112,
  "systolic_bp": 88,
  "diastolic_bp": 54,
  "temperature": 39.1,
  "spo2": 93,
  "respiratory_rate": 24
}

// Response
{
  "patient_id": "P001",
  "risk_score": 0.82,
  "risk_level": "CRITICAL",
  "alert_triggered": true,
  "top_features": [
    {"feature": "lactate_trend_15m",  "shap_value": 0.31},
    {"feature": "spo2_min_60m",       "shap_value": 0.24},
    {"feature": "heart_rate_mean_15m","shap_value": 0.19}
  ],
  "sofa_score": 6,
  "news2_score": 9,
  "inference_time_ms": 95
}
```

#### `GET /health` — Kiểm tra trạng thái

```json
{
  "status": "healthy",
  "model_version": "1.0",
  "model_auroc": 0.91,
  "uptime_seconds": 86400
}
```

### Alert Service (FastAPI — port 8002)

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/alerts` | Danh sách alert (filter theo status, patient) |
| `GET` | `/alerts/{id}` | Chi tiết 1 alert |
| `PATCH` | `/alerts/{id}/acknowledge` | Xác nhận đã xử lý |
| `GET` | `/alerts/stats` | Thống kê alert theo ca trực |

---

## 10. Kết quả và đánh giá

### Kết quả mô hình (test set — Synthetic ICU)

| Metric | Kết quả | Mục tiêu |
|--------|---------|----------|
| AUROC | — | > 0.85 |
| Sensitivity | — | > 75% |
| Specificity | — | > 80% |
| F1-score | — | > 0.75 |
| Alert lead time | — | > 30 phút |

### Kết quả hệ thống

| Metric | Kết quả | Mục tiêu |
|--------|---------|----------|
| Inference latency (p95) | — | < 200ms |
| End-to-end alert latency | — | < 5 phút |
| Concurrent patients supported | — | ≥ 20 |

### Hạn chế & Hướng phát triển

**Hạn chế hiện tại:**
- Dữ liệu synthetic chưa phản ánh đầy đủ độ phức tạp của ICU thực tế
- Chỉ sử dụng XGBoost, chưa khai thác thông tin chuỗi thời gian dài hạn
- Chu kỳ dự đoán 5 phút, chưa hỗ trợ alert tức thời theo giây

**Hướng phát triển:**
- Tích hợp MIMIC-IV để train trên dữ liệu thực
- Bổ sung LSTM để khai thác time-series
- Rút ngắn chu kỳ dự đoán xuống dưới 1 phút khi có phần cứng phù hợp

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
   │1.1 Thêm│             │2.1 Thu  │             │3.1 Theo │
   │bệnh    │             │thập     │             │dõi drift│
   │nhân    │             │vitals   │             │dữ liệu  │
   └────────┘             │(5 phút) │             └─────────┘
   ┌────────┐             └─────────┘             ┌─────────┐
   │1.2 Xem │             ┌─────────┐             │3.2 Tự   │
   │danh    │             │2.2 Dự   │             │động     │
   │sách    │             │đoán     │             │retrain  │
   └────────┘             │risk     │             └─────────┘
   ┌────────┐             │score    │             ┌─────────┐
   │1.3 Xem │             └─────────┘             │3.3 Theo │
   │lịch sử │             ┌─────────┐             │dõi hệ   │
   │alert   │             │2.3 Giải │             │thống    │
   └────────┘             │thích    │             │Grafana  │
                          │SHAP     │             └─────────┘
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
  │          │──────────┼─►│ UC01: Xem danh sách bệnh nhân     │  │
  │          │          │  └────────────────────────────────────┘  │
  │   Y tá   │          │  ┌────────────────────────────────────┐  │
  │  (Nurse) │──────────┼─►│ UC02: Xem risk score real-time    │  │
  │          │          │  └────────────────────────────────────┘  │
  │          │          │  ┌────────────────────────────────────┐  │
  └──────────┘          │  │ UC03: Nhận cảnh báo CRITICAL      │  │
        │               │  └────────────────────────────────────┘  │
        ├───────────────┼─►                                         │
        │               │  ┌────────────────────────────────────┐  │
        │               │  │ UC04: Xem giải thích SHAP         │  │
        ├───────────────┼─►│                                    │  │
        │               │  └────────────────────────────────────┘  │
        │               │  ┌────────────────────────────────────┐  │
        │               │  │ UC05: Acknowledge alert            │  │
        └───────────────┼─►│                                    │  │
                        │  └────────────────────────────────────┘  │
                        │                                          │
  ┌──────────┐          │  ┌────────────────────────────────────┐  │
  │  Bác sĩ /│──────────┼─►│ UC06: Train / Retrain model       │  │
  │  Admin   │          │  └────────────────────────────────────┘  │
  └──────────┘          │  ┌────────────────────────────────────┐  │
        │               │  │ UC07: Xem báo cáo & metrics       │  │
        └───────────────┼─►│                                    │  │
                        │  └────────────────────────────────────┘  │
                        │                                          │
  ┌──────────┐          │  ┌────────────────────────────────────┐  │
  │ Simulator│──────────┼─►│ UC08: Gửi vitals tự động (5 phút) │  │
  └──────────┘          │  └────────────────────────────────────┘  │
                        │  ┌────────────────────────────────────┐  │
                        │  │ UC09: Theo dõi hệ thống (Grafana) │  │
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
      │   (mỗi 5 phút)            │                               │
      │───────────────────────────►│                               │
      │                            │                               │
      │                     ┌──────▼──────┐                       │
      │                     │ Validate    │                       │
      │                     │ input data  │                       │
      │                     └──────┬──────┘                       │
      │                            │                               │
      │                     ┌──────▼──────┐                       │
      │                     │ Tiền xử lý  │                       │
      │                     │ imputation  │                       │
      │                     │ normalize   │                       │
      │                     └──────┬──────┘                       │
      │                            │                               │
      │                     ┌──────▼──────┐                       │
      │                     │ Feature Eng │                       │
      │                     │ rolling stats│                      │
      │                     │ SOFA, NEWS2 │                       │
      │                     └──────┬──────┘                       │
      │                            │                               │
      │                     ┌──────▼──────┐                       │
      │                     │ XGBoost     │                       │
      │                     │ predict     │                       │
      │                     │ risk score  │                       │
      │                     └──────┬──────┘                       │
      │                            │                               │
      │                     ┌──────▼──────┐                       │
      │                     │ SHAP explain│                       │
      │                     │ top-5       │                       │
      │                     └──────┬──────┘                       │
      │                            │                               │
      │                     ┌──────▼──────┐                       │
      │                     │ Lưu DB      │                       │
      │                     │ kiểm ngưỡng │                       │
      │                     └──────┬──────┘                       │
      │                            │                               │
      │              ┌─────────────┼─────────────┐                │
      │           <0.3          0.3-0.7         ≥0.7              │
      │              │              │              │               │
      │           Log only    Warning UI    ┌──────▼──────┐       │
      │                             │       │ WebSocket   │       │
      │                             │       │ CRITICAL    │──────►│
      │                             │       │ push        │       │
      │                             │       └─────────────┘       │
      │                             │                        ┌─────▼─────┐
      │                             └───────────────────────►│ Hiển thị  │
      │                                                       │ alert +   │
      │                                                       │ SHAP chart│
      │◄──────────── response ─────────────────               └─────┬─────┘
      │   {risk_score, top_features}                                │
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
    │             │─POST /alert────────────── ►│             │        │
    │             │             │             │─WS push─────►        │
    │             │             │             │             │─notify►│
    │             │             │             │             │        │
    │◄─response───│             │             │             │        │
    │  {score,    │             │             │             │        │
    │   features} │             │             │             │        │
    │             │             │             │   Nurse acknowledge──►│
    │             │             │             │◄─PATCH /alerts/{id}──│
    │             │             │─UPDATE──────►│             │        │
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

Dashboard Django tại `http://localhost:8000` hiển thị danh sách bệnh nhân với risk score cập nhật mỗi 5 phút, màu sắc phân biệt theo mức độ (xanh / vàng / đỏ). Trang chi tiết bệnh nhân hiển thị biểu đồ risk score 2 giờ gần nhất, vitals hiện tại và top-5 SHAP features dạng bar chart. Khi risk score ≥ 0.7, hệ thống tự động đẩy cảnh báo CRITICAL qua WebSocket đến tất cả client đang kết nối mà không cần reload trang. Y tá có thể acknowledge alert trực tiếp từ dashboard, trạng thái cập nhật realtime cho toàn bộ người dùng.

*(Chèn ảnh: trang danh sách bệnh nhân, trang chi tiết, popup cảnh báo CRITICAL)*

### 2.2.4 Chức năng monitoring và tự động retrain

Evidently AI chạy định kỳ so sánh phân phối dữ liệu mới với reference data, tính drift score theo PSI. Nếu drift score > 0.7, Prefect tự động trigger flow retrain: load dữ liệu mới → feature engineering → train XGBoost → evaluate → nếu AUROC cao hơn model hiện tại thì promote lên Production, ngược lại giữ model cũ. Prometheus thu thập metrics từ FastAPI (request count, latency, prediction distribution) và Grafana hiển thị dashboard hệ thống tại `http://localhost:3000`.

*(Chèn ảnh: Grafana dashboard metrics, Prefect flow run history, Evidently drift report)*
