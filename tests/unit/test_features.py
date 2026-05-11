"""Unit tests for feature engineering: clinical scores and rolling vitals features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering.clinical_scores import (
    calculate_news2,
    calculate_qsofa,
    calculate_sofa,
)


# ---------------------------------------------------------------------------
# TestSOFA
# ---------------------------------------------------------------------------

class TestSOFA:
    """Tests for calculate_sofa()."""

    def test_sofa_normal_patient(self):
        """All normal values → SOFA == 0."""
        row = {
            "spo2": 98,
            "platelet": 200,
            "bilirubin": 0.8,
            "creatinine": 0.9,
            "systolic_bp": 120,
        }
        assert calculate_sofa(row) == 0

    def test_sofa_critical_patient(self):
        """Severely abnormal values → SOFA ≥ 16."""
        row = {
            "spo2": 88,       # respiratory +4
            "platelet": 40,   # coagulation +4
            "bilirubin": 15,  # liver +4
            "creatinine": 6,  # renal +4
            "systolic_bp": 65,  # cardiovascular +4
        }
        assert calculate_sofa(row) >= 16

    def test_sofa_returns_int(self):
        """Return type must be int."""
        row = {
            "spo2": 95,
            "platelet": 150,
            "bilirubin": 1.0,
            "creatinine": 1.0,
            "systolic_bp": 110,
        }
        assert isinstance(calculate_sofa(row), int)

    def test_sofa_partial_missing(self):
        """Missing keys should be treated as None → no points added for that component."""
        row = {"spo2": 98}  # only SpO2 provided
        result = calculate_sofa(row)
        assert isinstance(result, int)
        assert result == 0  # spo2=98 → 0 points, rest missing → 0

    def test_sofa_boundary_spo2(self):
        """spo2 == 94 → respiratory score +3 (< 94 threshold)."""
        row = {"spo2": 93, "platelet": None, "bilirubin": None, "creatinine": None, "systolic_bp": None}
        score = calculate_sofa(row)
        assert score == 3

    def test_sofa_moderate_patient(self):
        """Moderately abnormal values → intermediate SOFA."""
        row = {
            "spo2": 95,       # +2
            "platelet": 120,  # +2
            "bilirubin": 3.0, # +2
            "creatinine": 2.5,# +2
            "systolic_bp": 85,# +2
        }
        assert calculate_sofa(row) == 10


# ---------------------------------------------------------------------------
# TestNEWS2
# ---------------------------------------------------------------------------

class TestNEWS2:
    """Tests for calculate_news2()."""

    def test_news2_normal(self):
        """All normal vital signs → NEWS2 == 0."""
        row = {
            "respiratory_rate": 16,
            "spo2": 97,
            "temperature": 37.0,
            "systolic_bp": 120,
            "heart_rate": 75,
        }
        assert calculate_news2(row) == 0

    def test_news2_high_risk(self):
        """Critically abnormal vitals → NEWS2 ≥ 12."""
        row = {
            "respiratory_rate": 26,  # +3
            "spo2": 90,              # +3
            "temperature": 39.5,     # +2
            "systolic_bp": 88,       # +3
            "heart_rate": 135,       # +3
        }
        assert calculate_news2(row) >= 12

    def test_news2_returns_int(self):
        row = {"respiratory_rate": 14, "spo2": 96, "temperature": 37.0,
               "systolic_bp": 115, "heart_rate": 80}
        assert isinstance(calculate_news2(row), int)

    def test_news2_partial_missing(self):
        """Only heart_rate provided → only heart_rate component scored."""
        row = {"heart_rate": 80}
        score = calculate_news2(row)
        assert isinstance(score, int)
        assert score == 0  # HR 80 → 0 points

    def test_news2_low_heart_rate(self):
        """Heart rate ≤ 40 → +3."""
        row = {"heart_rate": 38}
        assert calculate_news2(row) == 3

    def test_news2_very_low_bp(self):
        """Systolic BP ≤ 90 → +3."""
        row = {"systolic_bp": 85}
        assert calculate_news2(row) == 3


# ---------------------------------------------------------------------------
# TestQSOFA
# ---------------------------------------------------------------------------

class TestQSOFA:
    """Tests for calculate_qsofa()."""

    def test_qsofa_zero_normal(self):
        row = {"respiratory_rate": 16, "systolic_bp": 120}
        assert calculate_qsofa(row) == 0

    def test_qsofa_max_two(self):
        """Both criteria met → 2 (GCS not in dataset, max is 2)."""
        row = {"respiratory_rate": 24, "systolic_bp": 95}
        assert calculate_qsofa(row) == 2

    def test_qsofa_returns_int(self):
        assert isinstance(calculate_qsofa({"respiratory_rate": 18, "systolic_bp": 110}), int)


# ---------------------------------------------------------------------------
# TestRollingFeatures
# ---------------------------------------------------------------------------

class TestRollingFeatures:
    """Tests for add_rolling_features()."""

    @pytest.fixture
    def sample_df(self):
        """A minimal dataframe with 10 records for one patient."""
        return pd.DataFrame(
            {
                "patient_id": ["P001"] * 10,
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="5min"),
                "heart_rate": np.random.normal(80, 5, 10),
                "systolic_bp": np.random.normal(120, 10, 10),
                "diastolic_bp": np.random.normal(80, 5, 10),
                "temperature": np.random.normal(37, 0.3, 10),
                "spo2": np.random.normal(98, 1, 10),
                "respiratory_rate": np.random.normal(16, 2, 10),
            }
        )

    def test_rolling_adds_mean_columns(self, sample_df):
        from feature_engineering.vitals_features import add_rolling_features

        result = add_rolling_features(sample_df)
        assert "heart_rate_mean_15m" in result.columns

    def test_rolling_adds_std_columns(self, sample_df):
        from feature_engineering.vitals_features import add_rolling_features

        result = add_rolling_features(sample_df)
        assert "heart_rate_std_60m" in result.columns

    def test_rolling_adds_trend_column(self, sample_df):
        from feature_engineering.vitals_features import add_rolling_features

        result = add_rolling_features(sample_df)
        assert "heart_rate_trend_15m" in result.columns

    def test_rolling_no_rows_lost(self, sample_df):
        from feature_engineering.vitals_features import add_rolling_features

        result = add_rolling_features(sample_df)
        assert len(result) == len(sample_df)

    def test_rolling_all_vitals_have_mean(self, sample_df):
        from feature_engineering.vitals_features import add_rolling_features

        result = add_rolling_features(sample_df)
        for vital in ["heart_rate", "systolic_bp", "diastolic_bp", "temperature", "spo2", "respiratory_rate"]:
            assert f"{vital}_mean_15m" in result.columns, f"Missing {vital}_mean_15m"

    def test_rolling_values_are_finite(self, sample_df):
        from feature_engineering.vitals_features import add_rolling_features

        result = add_rolling_features(sample_df)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert not result[numeric_cols].isin([np.inf, -np.inf]).any().any()
