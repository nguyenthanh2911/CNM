-- CNM ICU Sepsis Database Schema (PostgreSQL)

-- Required for gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE patients (
  patient_id VARCHAR(20) PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  age INT,
  gender VARCHAR(10),
  ward VARCHAR(50),
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE admissions (
  admission_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id VARCHAR(20) REFERENCES patients(patient_id),
  admitted_at TIMESTAMP NOT NULL,
  discharged_at TIMESTAMP,
  bed_number VARCHAR(20),
  status VARCHAR(20) DEFAULT 'active'
);

CREATE TABLE vital_records (
  record_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id VARCHAR(20) REFERENCES patients(patient_id),
  timestamp TIMESTAMP NOT NULL,
  heart_rate FLOAT, systolic_bp FLOAT, diastolic_bp FLOAT,
  temperature FLOAT, spo2 FLOAT, respiratory_rate FLOAT,
  lactate FLOAT, wbc FLOAT, creatinine FLOAT,
  bilirubin FLOAT, platelet FLOAT
);
CREATE INDEX idx_vital_patient_time ON vital_records(patient_id, timestamp DESC);

CREATE TABLE prediction_results (
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
CREATE INDEX idx_pred_patient_time ON prediction_results(patient_id, timestamp DESC);

CREATE TABLE alerts (
  alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id VARCHAR(20) REFERENCES patients(patient_id),
  result_id UUID REFERENCES prediction_results(result_id),
  severity VARCHAR(20) NOT NULL,
  top_features JSONB,
  created_at TIMESTAMP DEFAULT NOW(),
  acknowledged BOOLEAN DEFAULT FALSE,
  ack_by VARCHAR(100),
  ack_at TIMESTAMP
);
CREATE INDEX idx_alerts_patient ON alerts(patient_id, created_at DESC);
CREATE INDEX idx_alerts_unacked ON alerts(acknowledged) WHERE acknowledged = FALSE;
