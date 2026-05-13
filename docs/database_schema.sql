-- CNM ICU Sepsis Database Schema (PostgreSQL)

-- Required for gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS patients (
  patient_id VARCHAR(20) PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  age INT,
  gender VARCHAR(10),
  ward VARCHAR(50),
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS admissions (
  admission_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id VARCHAR(20) REFERENCES patients(patient_id),
  admitted_at TIMESTAMP NOT NULL,
  discharged_at TIMESTAMP,
  bed_number VARCHAR(20),
  status VARCHAR(20) DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS vital_records (
  record_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id VARCHAR(20) REFERENCES patients(patient_id),
  timestamp TIMESTAMP NOT NULL,
  heart_rate FLOAT, systolic_bp FLOAT, diastolic_bp FLOAT,
  temperature FLOAT, spo2 FLOAT, respiratory_rate FLOAT,
  lactate FLOAT, wbc FLOAT, creatinine FLOAT,
  bilirubin FLOAT, platelet FLOAT
);
CREATE INDEX IF NOT EXISTS idx_vital_patient_time ON vital_records(patient_id, timestamp DESC);

-- Table: predictions (synced with PredictionORM in services/ml_service/main.py)
CREATE TABLE IF NOT EXISTS predictions (
  id                        SERIAL PRIMARY KEY,
  patient_id                VARCHAR(64) NOT NULL,
  timestamp                 TIMESTAMPTZ NOT NULL,
  risk_score                DOUBLE PRECISION NOT NULL,
  risk_level                VARCHAR(16) NOT NULL,
  alert_triggered           BOOLEAN NOT NULL,
  sofa_score                INT NOT NULL,
  news2_score               INT NOT NULL,
  inference_time_ms         DOUBLE PRECISION NOT NULL,
  -- raw vitals (nullable)
  heart_rate                DOUBLE PRECISION,
  systolic_bp               DOUBLE PRECISION,
  diastolic_bp              DOUBLE PRECISION,
  temperature               DOUBLE PRECISION,
  spo2                      DOUBLE PRECISION,
  respiratory_rate          DOUBLE PRECISION,
  lactate                   DOUBLE PRECISION,
  wbc                       DOUBLE PRECISION,
  creatinine                DOUBLE PRECISION,
  bilirubin                 DOUBLE PRECISION,
  platelet                  DOUBLE PRECISION,
  -- early warning scores (nullable)
  early_warning_probability DOUBLE PRECISION,
  early_warning_level       VARCHAR(16),
  trend_score               DOUBLE PRECISION,
  rate_of_change_score      DOUBLE PRECISION,
  threshold_score           DOUBLE PRECISION,
  created_at                TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_patient_time ON predictions(patient_id, timestamp DESC);

-- DEPRECATED: prediction_results kept for backward compatibility.
-- Use 'predictions' table instead (matching PredictionORM).
CREATE TABLE IF NOT EXISTS prediction_results (
  result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id VARCHAR(20) REFERENCES patients(patient_id),
  timestamp TIMESTAMP NOT NULL,
  risk_score FLOAT NOT NULL,
  risk_level VARCHAR(20) NOT NULL,
  sofa_score INT,
  news2_score INT,
  inference_ms FLOAT,
  created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_pred_patient_time ON prediction_results(patient_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS alerts (
  alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id VARCHAR(20) REFERENCES patients(patient_id),
  risk_score DOUBLE PRECISION NOT NULL,
  risk_level VARCHAR(16) NOT NULL,
  alert_type VARCHAR(32) DEFAULT 'sepsis',
  top_features JSONB,
  sofa_score INT NOT NULL DEFAULT 0,
  news2_score INT NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL,
  acknowledged BOOLEAN DEFAULT FALSE,
  ack_by VARCHAR(128),
  ack_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_alerts_patient ON alerts(patient_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_unacked ON alerts(acknowledged) WHERE acknowledged = FALSE;
