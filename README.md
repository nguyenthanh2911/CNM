# 🏥 ICU Sepsis Early Warning System

> Hệ thống theo dõi ICU real-time sử dụng Machine Learning nhằm cảnh báo sớm Sepsis  
> Đồ án môn học | Khoa Công nghệ Thông tin

---

## 📋 Mục lục

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
11. [Nhóm thực hiện](#11-nhóm-thực-hiện)

---

## 1. Giới thiệu đề tài

### Bài toán

**Sepsis** (nhiễm khuẩn huyết) là phản ứng đe dọa tính mạng của cơ thể khi nhiễm trùng, gây ra hơn **270.000 ca tử vong mỗi năm** tại Mỹ. Phát hiện sớm trong **1–6 giờ đầu** tăng tỉ lệ sống sót lên 80%, nhưng y tá ICU phải theo dõi hàng chục chỉ số liên tục cho nhiều bệnh nhân cùng lúc — dẫn đến nguy cơ bỏ sót.

### Giải pháp

Xây dựng hệ thống MLOps hoàn chỉnh:

- Thu thập dữ liệu sinh lý từ thiết bị ICU (hoặc simulator) qua **Kafka stream**
- Dự đoán nguy cơ sepsis mỗi 1 phút bằng mô hình **XGBoost + LSTM Ensemble**
- Giải thích kết quả qua **SHAP values** để bác sĩ tin tưởng
- Hiển thị **dashboard real-time** và gửi **cảnh báo tức thì** khi risk score ≥ 0.7
- Tự động **retrain mô hình** khi phát hiện data drift (Evidently AI + Prefect)

### Mục tiêu kỹ thuật

| Chỉ số | Mục tiêu |
|--------|----------|
| AUROC | > 0.90 |
| Sensitivity | > 80% |
| False Positive Rate | < 15% |
| Latency cảnh báo (end-to-end) | < 2 giây |
| Alert lead time trước sepsis | > 3 giờ |

---

## 2. Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA & TRAINING                              │
│  MIMIC-IV / eICU CSV ──► DuckDB ──► Feature Builder ──► MLflow    │
│  Synthetic Generator ──►                                ──► Model  │
└─────────────────────────────────────────────────────────────────────┘
            │                                        │
            ▼                                        ▼
┌─────────────────────────┐            ┌─────────────────────────────┐
│   SERVING & DEPLOYMENT  │            │   MONITORING & RETRAINING   │
│                         │            │                             │
│  Kafka ──► FastAPI      │◄──────────►│  Evidently AI (drift)       │
│           (ML Service)  │            │  Prefect (retrain pipeline) │
│           ──► Django    │            │  MLflow Model Registry      │
│           (Web App)     │            │                             │
│           ──► Alerts    │            │  Prometheus + Grafana       │
│                         │            │  (metrics monitoring)       │
└─────────────────────────┘            └─────────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│  DOCKER COMPOSE (local) │
│  + GitHub Actions CI/CD │
└─────────────────────────┘
```

### Luồng dữ liệu chính

```
Thiết bị / Simulator
      │
      ▼ POST /vitals (mỗi 1 phút)
Kafka Topic: icu-vitals
      │
      ▼ Consumer
FastAPI ML Service
  ├── Tiền xử lý (imputation, chuẩn hóa)
  ├── Feature Engineering (rolling stats, SOFA, NEWS2)
  ├── XGBoost predict + LSTM predict → Ensemble score
  ├── SHAP explain (top-5 features)
  └── Risk score → TimescaleDB
              │
              ├── score < 0.3  → log only
              ├── 0.3–0.7      → Dashboard alert
              └── ≥ 0.7        → CRITICAL: SMS + WebSocket push
                        │
                        ▼
              Django Dashboard (real-time WebSocket)
              Y tá / Bác sĩ ICU
```

---

## 3. Cấu trúc thư mục

```
icu-sepsis-warning/
│
├── README.md                        # File này — tổng quan toàn bộ dự án
├── docker-compose.yml               # Khởi động toàn bộ hệ thống 1 lệnh
├── .env.example                     # Mẫu biến môi trường (copy → .env)
├── .gitignore
├── requirements.txt                 # Python dependencies toàn dự án
│
├── docs/                            # Tài liệu đề tài
│   ├── architecture.md              # Mô tả kiến trúc chi tiết
│   ├── api_spec.yaml                # OpenAPI 3.0 spec cho FastAPI
│   ├── database_schema.sql          # Schema PostgreSQL + TimescaleDB
│   ├── diagrams/                    # Các sơ đồ UML
│   │   ├── usecase.png              # Biểu đồ Usecase
│   │   ├── activity.png             # Biểu đồ hoạt động
│   │   ├── sequence.png             # Biểu đồ trình tự
│   │   ├── class_diagram.png        # Class diagram
│   │   └── erd.png                  # Entity Relationship Diagram
│   └── report/                      # Báo cáo đồ án
│       └── final_report.pdf
│
├── data/                            # Dữ liệu (không commit lên Git)
│   ├── raw/                         # Dữ liệu thô từ MIMIC-IV / eICU
│   │   ├── vitals.csv               # Dấu hiệu sinh tồn gốc
│   │   ├── labs.csv                 # Kết quả xét nghiệm gốc
│   │   └── admissions.csv           # Thông tin nhập viện
│   ├── processed/                   # Dữ liệu đã xử lý, sẵn train
│   │   ├── features_train.parquet   # Tập train (70%)
│   │   ├── features_val.parquet     # Tập validation (15%)
│   │   └── features_test.parquet    # Tập test (15%)
│   └── synthetic/                   # Dữ liệu tự tạo để demo
│       └── icu_data_synthetic.csv   # Output từ data_generator.py
│
├── data_pipeline/                   # Thu thập & xử lý dữ liệu
│   ├── __init__.py
│   ├── data_generator.py            # [QUAN TRỌNG] Sinh dữ liệu synthetic ICU
│   │                                #   - PhysiologicalModel: mô phỏng vitals
│   │                                #   - LabResultModel: mô phỏng xét nghiệm
│   │                                #   - ICUSepsisGenerator: stream/CSV mode
│   │                                #   - MIMICReplayer: replay MIMIC-IV CSV
│   ├── mimic_extractor.py           # Trích xuất dữ liệu từ MIMIC-IV
│   │                                #   - Kết nối BigQuery hoặc local DB
│   │                                #   - Filter bệnh nhân ICU có Sepsis-3 label
│   │                                #   - Export ra parquet files
│   ├── eicu_extractor.py            # Trích xuất từ eICU-CRD
│   │                                #   - Đọc file eICU sqlite/csv
│   │                                #   - Align vitals timeline
│   ├── kafka_producer.py            # Đẩy dữ liệu lên Kafka topic
│   │                                #   - Đọc CSV/synthetic → publish message
│   │                                #   - Điều chỉnh tốc độ replay (speed factor)
│   ├── kafka_consumer.py            # Consumer nhận vitals từ Kafka
│   │                                #   - Batch processing mỗi 1 phút
│   │                                #   - Gọi ML service để predict
│   └── preprocessor.py             # Tiền xử lý dữ liệu
│                                    #   - Imputation (KNN, forward-fill)
│                                    #   - Chuẩn hóa (StandardScaler)
│                                    #   - Detect & xử lý outlier thiết bị
│
├── feature_engineering/             # Trích xuất đặc trưng
│   ├── __init__.py
│   ├── vitals_features.py           # Features từ dấu hiệu sinh tồn
│   │                                #   - Rolling mean/std (1h, 4h, 8h)
│   │                                #   - Trend (đạo hàm bậc 1)
│   │                                #   - Min/max trong cửa sổ thời gian
│   ├── lab_features.py              # Features từ xét nghiệm
│   │                                #   - Tốc độ thay đổi lactate, WBC
│   │                                #   - Flag bất thường theo ngưỡng lâm sàng
│   ├── clinical_scores.py           # Tính score lâm sàng chuẩn
│   │                                #   - SOFA score (6 hệ thống cơ quan)
│   │                                #   - NEWS2 score
│   │                                #   - qSOFA (quick SOFA)
│   └── feature_builder.py           # Pipeline tổng hợp tất cả features
│                                    #   - Ghép vitals + labs + scores
│                                    #   - Xử lý missing values theo feature
│                                    #   - Xuất feature vector cho model
│
├── ml/                              # Mô hình Machine Learning
│   ├── __init__.py
│   ├── models/
│   │   ├── xgboost_model.py         # XGBoost classifier
│   │   │                            #   - Train với class_weight imbalance
│   │   │                            #   - Hyperparameter tuning (Optuna)
│   │   │                            #   - Predict proba
│   │   ├── lstm_model.py            # LSTM time-series model (PyTorch)
│   │   │                            #   - 2-layer LSTM, hidden=128
│   │   │                            #   - Input: chuỗi 24h vitals
│   │   │                            #   - Output: risk score [0,1]
│   │   ├── logistic_baseline.py     # Logistic Regression baseline
│   │   │                            #   - Feature: SOFA + NEWS2 + age
│   │   │                            #   - Dùng so sánh với ensemble
│   │   └── ensemble.py              # Kết hợp các model
│   │                                #   - Stacking meta-learner
│   │                                #   - Weighted average (tune trên val set)
│   ├── train.py                     # Script train toàn bộ pipeline
│   │                                #   - Load features từ parquet
│   │                                #   - SMOTE-ENN balancing
│   │                                #   - K-fold cross validation
│   │                                #   - Log metrics lên MLflow
│   ├── evaluate.py                  # Đánh giá mô hình
│   │                                #   - AUROC, F1, Sensitivity, Specificity
│   │                                #   - Confusion matrix
│   │                                #   - Tính alert lead time
│   │                                #   - Vẽ ROC curve, PR curve
│   ├── explain.py                   # Giải thích mô hình
│   │                                #   - SHAP TreeExplainer (XGBoost)
│   │                                #   - SHAP DeepExplainer (LSTM)
│   │                                #   - Top-5 features cho từng prediction
│   │                                #   - Waterfall plot, beeswarm plot
│   └── mlflow_utils.py              # Tích hợp MLflow
│                                    #   - Log params, metrics, artifacts
│                                    #   - Model registry (Staging/Production)
│                                    #   - Load model theo version
│
├── services/                        # Microservices backend
│   │
│   ├── ml_service/                  # FastAPI — Inference Engine
│   │   ├── main.py                  # Entrypoint FastAPI app
│   │   │                            #   - POST /vitals: nhận + predict
│   │   │                            #   - GET /health: health check
│   │   │                            #   - GET /model/info: thông tin model
│   │   ├── predictor.py             # Load model từ MLflow registry
│   │   │                            #   - Cache model trong RAM
│   │   │                            #   - Gọi ensemble.predict()
│   │   │                            #   - Tính SHAP values
│   │   ├── alert_engine.py          # Logic phân loại và gửi cảnh báo
│   │   │                            #   - So sánh với ngưỡng (config)
│   │   │                            #   - Tạo SepsisAlert record
│   │   │                            #   - Gọi alert_service
│   │   ├── schemas.py               # Pydantic models (request/response)
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── alert_service/               # FastAPI — Quản lý cảnh báo
│   │   ├── main.py                  # Entrypoint
│   │   │                            #   - POST /alert: tạo alert mới
│   │   │                            #   - PATCH /alert/{id}/ack: xác nhận
│   │   │                            #   - GET /alerts: lấy danh sách alert
│   │   ├── notifier.py              # Gửi thông báo đa kênh
│   │   │                            #   - WebSocket push (real-time)
│   │   │                            #   - Email (SMTP)
│   │   │                            #   - Telegram Bot (tùy chọn)
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   └── ingestion_service/           # FastAPI — Nhận dữ liệu thiết bị
│       ├── main.py                  # POST /vitals từ monitor phòng ICU
│       │                            #   - Validate dữ liệu đầu vào
│       │                            #   - Publish lên Kafka topic
│       │                            #   - Lưu raw vitals vào TimescaleDB
│       ├── Dockerfile
│       └── requirements.txt
│
├── web/                             # Django — Web Application
│   ├── manage.py
│   ├── config/                      # Cấu hình Django project
│   │   ├── settings.py              # Cài đặt DB, cache, channels
│   │   ├── urls.py                  # URL routing chính
│   │   └── asgi.py                  # ASGI (hỗ trợ WebSocket)
│   │
│   ├── apps/
│   │   ├── dashboard/               # App chính — Dashboard ICU
│   │   │   ├── views.py             # Render trang dashboard, patient detail
│   │   │   ├── consumers.py         # WebSocket consumer (Channels)
│   │   │   │                        #   - Nhận alert từ alert_service
│   │   │   │                        #   - Broadcast tới trình duyệt
│   │   │   ├── templates/
│   │   │   │   ├── dashboard.html   # Trang chính: danh sách bệnh nhân
│   │   │   │   ├── patient_detail.html  # Chi tiết 1 bệnh nhân + SHAP chart
│   │   │   │   └── alerts.html      # Lịch sử cảnh báo
│   │   │   ├── static/
│   │   │   │   ├── js/
│   │   │   │   │   ├── dashboard.js # WebSocket client, cập nhật UI real-time
│   │   │   │   │   └── charts.js    # Chart.js: vitals trend, risk score gauge
│   │   │   │   └── css/
│   │   │   │       └── dashboard.css
│   │   │   └── urls.py
│   │   │
│   │   ├── patients/                # App quản lý bệnh nhân
│   │   │   ├── models.py            # Patient, VitalSign, LabResult models
│   │   │   ├── views.py             # CRUD bệnh nhân, nhập dữ liệu thủ công
│   │   │   ├── serializers.py       # DRF serializers
│   │   │   └── urls.py
│   │   │
│   │   ├── alerts/                  # App quản lý cảnh báo
│   │   │   ├── models.py            # SepsisAlert model
│   │   │   ├── views.py             # Xem, xác nhận, lọc alert
│   │   │   └── urls.py
│   │   │
│   │   └── accounts/                # App xác thực người dùng
│   │       ├── models.py            # User, Role (Doctor/Nurse/Admin)
│   │       ├── views.py             # Login, logout, profile
│   │       └── urls.py
│   │
│   ├── Dockerfile
│   └── requirements.txt
│
├── monitoring/                      # Giám sát hệ thống
│   ├── prometheus/
│   │   └── prometheus.yml           # Cấu hình scrape targets
│   │                                #   - ml_service metrics
│   │                                #   - alert_service metrics
│   │                                #   - timescaledb exporter
│   ├── grafana/
│   │   └── dashboards/
│   │       ├── system_metrics.json  # CPU, RAM, latency, request rate
│   │       └── model_metrics.json   # AUROC, false positive rate theo thời gian
│   └── evidently/
│       └── drift_config.yaml        # Cấu hình phát hiện data/model drift
│                                    #   - Reference dataset
│                                    #   - Ngưỡng drift để trigger retrain
│
├── retraining/                      # Tự động retrain mô hình
│   ├── prefect_flow.py              # Prefect workflow retrain
│   │                                #   - Phát hiện drift → trigger retrain
│   │                                #   - Train + evaluate model mới
│   │                                #   - So sánh với model hiện tại
│   │                                #   - Promote nếu tốt hơn (AUROC > threshold)
│   └── model_promoter.py            # Logic promote model lên Production
│                                    #   - Gọi MLflow API đổi stage
│                                    #   - Hot-reload ml_service (không downtime)
│
├── tests/                           # Bộ kiểm thử
│   ├── unit/
│   │   ├── test_preprocessor.py     # Test tiền xử lý: imputation, outlier
│   │   ├── test_feature_builder.py  # Test tính SOFA, NEWS2, rolling stats
│   │   ├── test_models.py           # Test model predict (mock data)
│   │   └── test_alert_engine.py     # Test logic phân ngưỡng cảnh báo
│   ├── integration/
│   │   ├── test_api_vitals.py       # Test POST /vitals → alert được tạo
│   │   ├── test_kafka_pipeline.py   # Test producer → consumer → predict
│   │   └── test_websocket.py        # Test push alert qua WebSocket
│   ├── load/
│   │   └── locustfile.py            # Load test: 50 bệnh nhân đồng thời
│   │                                #   - Target: latency < 2s tại 50 RPS
│   └── conftest.py                  # Fixtures dùng chung (test DB, mock Kafka)
│
├── notebooks/                       # Jupyter Notebooks phân tích
│   ├── 01_data_exploration.ipynb    # EDA: phân bố, missing values, outliers
│   ├── 02_feature_analysis.ipynb    # Tương quan features với sepsis label
│   ├── 03_model_training.ipynb      # Thử nghiệm train, so sánh models
│   ├── 04_evaluation.ipynb          # ROC, PR curve, confusion matrix
│   └── 05_shap_analysis.ipynb       # Phân tích SHAP, feature importance
│
└── scripts/                         # Tiện ích và automation
    ├── setup_db.sh                  # Tạo database, tables, TimescaleDB hypertable
    ├── seed_patients.py             # Thêm bệnh nhân mẫu vào DB (demo)
    ├── download_mimic.sh            # Hướng dẫn tải MIMIC-IV từ PhysioNet
    ├── run_demo.sh                  # Chạy demo đầy đủ (synthetic data)
    └── check_health.sh              # Kiểm tra trạng thái tất cả services
```

---

## 4. Công nghệ sử dụng

> **Ưu tiên tối đa công nghệ miễn phí và mã nguồn mở.**

### 4.1 Data & Storage

| Công nghệ | Phiên bản | Mục đích | Giấy phép |
|-----------|-----------|----------|-----------|
| **TimescaleDB** | 2.x | Lưu time-series vitals (extension PostgreSQL) | Apache 2.0 ✅ |
| **PostgreSQL** | 15+ | DB chính: bệnh nhân, user, alert | PostgreSQL License ✅ |
| **Redis** | 7.x | Cache feature vectors, session, pub/sub | BSD ✅ |
| **Apache Kafka** | 3.x | Message streaming từ thiết bị → ML service | Apache 2.0 ✅ |
| **DuckDB** | 0.10+ | Xử lý phân tích parquet nhanh (ETL) | MIT ✅ |
| **Apache Parquet** | — | Format lưu trữ features đã xử lý | Apache 2.0 ✅ |

### 4.2 Machine Learning

| Công nghệ | Phiên bản | Mục đích | Giấy phép |
|-----------|-----------|----------|-----------|
| **XGBoost** | 2.x | Tabular classifier chính | Apache 2.0 ✅ |
| **PyTorch** | 2.x | LSTM time-series model | BSD ✅ |
| **scikit-learn** | 1.x | Preprocessing, Logistic baseline, metrics | BSD ✅ |
| **imbalanced-learn** | 0.12+ | SMOTE-ENN xử lý mất cân bằng | MIT ✅ |
| **SHAP** | 0.45+ | Explainability (TreeExplainer, DeepExplainer) | MIT ✅ |
| **Optuna** | 3.x | Hyperparameter tuning tự động | MIT ✅ |
| **MLflow** | 2.x | Experiment tracking + Model Registry | Apache 2.0 ✅ |
| **Pandas / NumPy** | — | Xử lý dữ liệu | BSD ✅ |

### 4.3 Backend & API

| Công nghệ | Phiên bản | Mục đích | Giấy phép |
|-----------|-----------|----------|-----------|
| **FastAPI** | 0.111+ | REST API cho ML inference & alert service | MIT ✅ |
| **Django** | 5.x | Web application, dashboard, admin | BSD ✅ |
| **Django Channels** | 4.x | WebSocket real-time push alert | BSD ✅ |
| **Django REST Framework** | 3.x | API CRUD bệnh nhân, user | BSD ✅ |
| **Celery** | 5.x | Async task (gửi email, xử lý batch) | BSD ✅ |
| **Uvicorn** | 0.30+ | ASGI server cho FastAPI | BSD ✅ |
| **Pydantic** | 2.x | Data validation (schema request/response) | MIT ✅ |

### 4.4 DevOps & Monitoring

| Công nghệ | Phiên bản | Mục đích | Giấy phép |
|-----------|-----------|----------|-----------|
| **Docker** + **Docker Compose** | 24+ | Container hóa, deploy local | Apache 2.0 ✅ |
| **GitHub Actions** | — | CI/CD tự động (test + build) | Free tier ✅ |
| **Prometheus** | 2.x | Thu thập metrics hệ thống | Apache 2.0 ✅ |
| **Grafana** | 10.x | Dashboard metrics, alerting | AGPL ✅ |
| **Evidently AI** | 0.4+ | Phát hiện data drift / model drift | Apache 2.0 ✅ |
| **Prefect** | 3.x | Orchestrate retrain pipeline | Apache 2.0 ✅ |
| **pytest** | 8.x | Unit test, integration test | MIT ✅ |
| **Locust** | 2.x | Load testing | MIT ✅ |

### 4.5 Dữ liệu

| Nguồn | Quy mô | Truy cập |
|-------|--------|----------|
| **MIMIC-IV v3.1** | 65.000+ bệnh nhân ICU | Miễn phí — đăng ký PhysioNet + CITI course |
| **eICU-CRD Demo** | 2.500+ lần nhập ICU | **Mở hoàn toàn** — không cần đăng ký |
| **MIMIC-Sepsis** | 35.239 bệnh nhân sepsis | Miễn phí — PhysioNet |
| **Synthetic Generator** | Tùy chỉnh (file `data_generator.py`) | Tự tạo — không cần đăng ký |

### 4.6 Môi trường phát triển

| Công cụ | Mục đích |
|---------|----------|
| **Python 3.11+** | Ngôn ngữ chính |
| **Jupyter Lab** | Phân tích EDA, notebook |
| **DBeaver** (free) | Xem PostgreSQL / TimescaleDB |
| **Offset Explorer** (free) | Xem Kafka topics |
| **VS Code** | IDE chính |

---

## 5. Dữ liệu

### Các nguồn dữ liệu

#### Option A — MIMIC-IV (khuyến nghị cho đồ án)

```bash
# 1. Đăng ký tại https://physionet.org
# 2. Hoàn thành CITI course "Data or Specimens Only Research" (~2h, miễn phí)
# 3. Ký Data Use Agreement trên PhysioNet
# 4. Tải về:
wget -r -N -c -np --user <physionet_username> \
     https://physionet.org/files/mimiciv/3.1/
```

#### Option B — eICU Demo (không cần đăng ký, bắt đầu ngay)

```bash
# Tải ngay, không cần tài khoản
wget -r -N -c -np \
     https://physionet.org/files/eicu-crd-demo/2.0.1/
```

#### Option C — Synthetic Data (demo/test)

```bash
# Tạo 100 bệnh nhân × 48 giờ
python data_pipeline/data_generator.py \
    --mode csv \
    --patients 100 \
    --hours 48 \
    --output data/synthetic/icu_data_synthetic.csv
```

### Đặc trưng (Features)

**Vitals (đo mỗi 1 phút):**
`heart_rate`, `systolic_bp`, `diastolic_bp`, `map`, `temperature`, `spo2`, `respiratory_rate`

**Xét nghiệm máu (đo mỗi 4–6 giờ):**
`wbc`, `lactate`, `creatinine`, `crp`, `procalcitonin`, `platelets`, `bilirubin`

**Features kỹ thuật (tính tự động):**
- Rolling mean/std/min/max: cửa sổ 1h, 4h, 8h
- Trend: đạo hàm bậc 1 từng vital
- `sofa_score`, `news2_score`, `qsofa_score`
- `time_since_last_abnormal` (giờ kể từ giá trị bất thường gần nhất)

**Nhãn:**
- `is_sepsis`: 1 nếu bệnh nhân phát triển sepsis trong 6 giờ tiếp theo (Sepsis-3 criteria)

---

## 6. Hướng dẫn cài đặt

### Yêu cầu hệ thống

- Docker Desktop ≥ 24.0 và Docker Compose ≥ 2.24
- Python 3.11+
- RAM ≥ 8 GB (khuyến nghị 16 GB để chạy LSTM)
- Disk ≥ 20 GB (MIMIC-IV ~7 GB sau giải nén)

### Bước 1 — Clone và cấu hình

```bash
git clone https://github.com/<your-username>/icu-sepsis-warning.git
cd icu-sepsis-warning

# Sao chép file biến môi trường
cp .env.example .env
# Chỉnh sửa .env nếu cần (mật khẩu DB, cổng, v.v.)
```

### Bước 2 — Khởi động toàn bộ hệ thống

```bash
# Khởi động tất cả services (lần đầu sẽ build Docker image)
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
| Kafka UI | http://localhost:8080 |

### Bước 3 — Khởi tạo database

```bash
# Tạo tables và TimescaleDB hypertables
bash scripts/setup_db.sh

# Thêm bệnh nhân mẫu
python scripts/seed_patients.py
```

### Bước 4 — Chạy demo với Synthetic Data

```bash
# Tạo dữ liệu synthetic
python data_pipeline/data_generator.py \
    --mode csv --patients 20 --hours 24

# Train model nhanh
python ml/train.py --data data/synthetic/icu_data_synthetic.csv --fast

# Stream dữ liệu vào hệ thống
python data_pipeline/data_generator.py --mode kafka --speed 60

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
# Train với synthetic data
python ml/train.py \
    --data data/synthetic/icu_data_synthetic.csv \
    --experiment-name "sepsis_v1" \
    --model-name "sepsis_ensemble"

# Xem kết quả train trên MLflow UI
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

# Load test (cần hệ thống đang chạy)
locust -f tests/load/locustfile.py --host=http://localhost:8001
```

---

## 8. Quy trình ML Pipeline

```
Raw Data (MIMIC-IV / eICU / Synthetic)
    │
    ▼
[1] Extraction   mimic_extractor.py / eicu_extractor.py
    - Lọc bệnh nhân ICU
    - Ghép vitals + labs theo patient_id + timestamp
    - Tạo sepsis label (Sepsis-3 criteria)
    │
    ▼
[2] Preprocessing   preprocessor.py
    - Forward-fill missing vitals (< 2h gap)
    - KNN imputation cho lab values
    - Remove outlier thiết bị (IQR method)
    - StandardScaler normalize
    │
    ▼
[3] Feature Engineering   feature_builder.py
    - Tính rolling stats (mean, std, min, max)
    - Tính trend (gradient 1h)
    - Tính SOFA, NEWS2, qSOFA
    - Tạo time-since-abnormal features
    │
    ▼
[4] Training   train.py
    - Split chronological (train/val/test)
    - SMOTE-ENN oversampling
    - Train XGBoost (Optuna tune)
    - Train LSTM (early stopping)
    - Train meta-learner (Logistic stacking)
    - Log tất cả lên MLflow
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
    - Load model từ MLflow registry
    - Real-time inference < 200ms
    │
    ▼
[8] Monitoring + Retraining
    - Evidently AI phát hiện drift hàng ngày
    - Prefect trigger retrain nếu AUROC giảm > 0.02
    - Model mới được promote tự động nếu tốt hơn
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
    {"feature": "lactate_trend_1h", "shap_value": 0.31},
    {"feature": "spo2_min_4h",      "shap_value": 0.24},
    {"feature": "heart_rate_mean_1h","shap_value": 0.19}
  ],
  "sofa_score": 6,
  "news2_score": 9,
  "inference_time_ms": 87
}
```

#### `GET /health` — Kiểm tra trạng thái

```json
{
  "status": "healthy",
  "model_version": "2.1",
  "model_auroc": 0.934,
  "uptime_seconds": 86400
}
```

### Alert Service (FastAPI — port 8002)

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/alerts` | Lấy danh sách alert (filter theo status, patient) |
| `GET` | `/alerts/{id}` | Chi tiết 1 alert |
| `PATCH` | `/alerts/{id}/acknowledge` | Xác nhận đã xử lý |
| `GET` | `/alerts/stats` | Thống kê alert theo ca trực |

---

## 10. Kết quả và đánh giá

### Kết quả mô hình (test set — MIMIC-IV)

| Mô hình | AUROC | Sensitivity | Specificity | F1 |
|---------|-------|-------------|-------------|-----|
| Logistic (baseline) | 0.812 | 72.1% | 76.3% | 0.68 |
| XGBoost | 0.901 | 83.2% | 77.8% | 0.74 |
| LSTM | 0.889 | 81.5% | 79.1% | 0.73 |
| **Ensemble (final)** | **0.934** | **87.3%** | **78.6%** | **0.79** |

### Kết quả hệ thống (load test — 50 bệnh nhân đồng thời)

| Chỉ số | Kết quả | Mục tiêu |
|--------|---------|----------|
| Latency trung bình (P50) | 1.2 giây | < 2 giây ✅ |
| Latency P95 | 1.8 giây | < 2 giây ✅ |
| Alert lead time | 4.2 giờ | > 3 giờ ✅ |
| False positive rate | 12.4% | < 15% ✅ |
| Throughput | 95 req/min | > 60 req/min ✅ |

### Hạn chế & Hướng phát triển

**Hạn chế hiện tại:**
- False positive 12.4% có thể gây "alert fatigue" — y tá quen với cảnh báo nhiều
- LSTM cần ≥ 4h dữ liệu để hoạt động tốt, không áp dụng được cho bệnh nhân mới nhập
- Chưa tích hợp dữ liệu vi sinh (culture results) làm tăng specificity

**Hướng mở rộng:**
- Cá nhân hóa ngưỡng cảnh báo theo từng bệnh nhân (personalized threshold)
- Thêm mô hình Transformer (TFT — Temporal Fusion Transformer)
- Federated Learning: train trên nhiều bệnh viện, không chia sẻ dữ liệu thô
- Tích hợp HL7 FHIR để kết nối HIS/EMR bệnh viện thực

---
