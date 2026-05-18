"""Integration tests for the full data → feature → prediction pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# TestFullFeaturePipeline
# ---------------------------------------------------------------------------

class TestFullFeaturePipeline:
    """End-to-end test: generate synthetic data → FeatureBuilder → assert columns."""

    @pytest.fixture(scope="class")
    def pipeline_result(self):
        """Run the full pipeline once and share the result across tests."""
        from data_pipeline.data_generator import ICUSepsisGenerator
        from feature_engineering.feature_builder import FeatureBuilder

        gen = ICUSepsisGenerator(n_patients=2, hours=1, interval_minutes=5)
        df = gen.generate_dataframe()

        builder = FeatureBuilder()
        result = builder.build(df)
        return result

    def test_result_not_empty(self, pipeline_result):
        assert not pipeline_result.empty

    def test_result_has_rows(self, pipeline_result):
        assert len(pipeline_result) > 0

    def test_sofa_score_column_present(self, pipeline_result):
        assert "sofa_score" in pipeline_result.columns

    def test_news2_score_column_present(self, pipeline_result):
        assert "news2_score" in pipeline_result.columns

    def test_qsofa_score_column_present(self, pipeline_result):
        assert "qsofa_score" in pipeline_result.columns

    def test_rolling_mean_column_present(self, pipeline_result):
        assert "heart_rate_mean_15m" in pipeline_result.columns

    def test_sofa_score_non_negative(self, pipeline_result):
        assert (pipeline_result["sofa_score"] >= 0).all()

    def test_news2_score_non_negative(self, pipeline_result):
        assert (pipeline_result["news2_score"] >= 0).all()

    def test_no_inf_values(self, pipeline_result):
        num_cols = pipeline_result.select_dtypes(include=[np.number]).columns
        assert not pipeline_result[num_cols].isin([np.inf, -np.inf]).any().any()

    def test_patient_id_preserved(self, pipeline_result):
        assert "patient_id" in pipeline_result.columns
        assert pipeline_result["patient_id"].nunique() == 2

    def test_raw_vitals_dropped(self, pipeline_result):
        """FeatureBuilder drops raw vitals columns after feature extraction."""
        dropped = ["heart_rate", "systolic_bp", "diastolic_bp",
                   "temperature", "spo2", "respiratory_rate"]
        for col in dropped:
            assert col not in pipeline_result.columns, f"Column '{col}' should have been dropped"


# ---------------------------------------------------------------------------
# TestGeneratorDataframe
# ---------------------------------------------------------------------------

class TestGeneratorDataframe:
    """Tests for ICUSepsisGenerator.generate_dataframe()."""

    def test_generates_expected_row_count(self):
        from data_pipeline.data_generator import ICUSepsisGenerator

        gen = ICUSepsisGenerator(n_patients=2, hours=1, interval_minutes=5)
        df = gen.generate_dataframe()
        # 2 patients × (60min / 5min) = 24 rows
        assert len(df) == 24

    def test_all_required_columns_present(self):
        from data_pipeline.data_generator import ICUSepsisGenerator

        gen = ICUSepsisGenerator(n_patients=1, hours=1, interval_minutes=5)
        df = gen.generate_dataframe()
        required = [
            "patient_id", "timestamp", "heart_rate", "systolic_bp",
            "diastolic_bp", "temperature", "spo2", "respiratory_rate",
            "lactate", "wbc", "creatinine", "bilirubin", "platelet",
            "sepsis_label",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_sepsis_label_binary(self):
        from data_pipeline.data_generator import ICUSepsisGenerator

        gen = ICUSepsisGenerator(n_patients=5, hours=1, interval_minutes=5)
        df = gen.generate_dataframe()
        assert set(df["sepsis_label"].unique()).issubset({0, 1})

    def test_vitals_in_physiological_range(self):
        from data_pipeline.data_generator import ICUSepsisGenerator

        gen = ICUSepsisGenerator(n_patients=3, hours=2, interval_minutes=5)
        df = gen.generate_dataframe()
        assert df["heart_rate"].between(20, 250).all(), "heart_rate out of range"
        assert df["spo2"].between(50, 100).all(), "spo2 out of range"
        assert df["temperature"].between(30, 45).all(), "temperature out of range"


# ---------------------------------------------------------------------------
# TestT6HLabelingIntegration
# ---------------------------------------------------------------------------

class TestT6HLabelingIntegration:
    """Integration test: data_generator → create_t6h_labels → split_by_patient."""

    @pytest.fixture(scope="class")
    def t6h_data(self):
        from data_pipeline.data_generator import ICUSepsisGenerator
        from data_pipeline.labeling import create_t6h_labels, split_by_patient

        gen = ICUSepsisGenerator(n_patients=10, hours=24, interval_minutes=5)
        df_raw = gen.generate_dataframe()
        df_labeled = create_t6h_labels(df_raw)
        train, val, test = split_by_patient(df_labeled)
        return df_labeled, train, val, test

    def test_label_col_exists(self, t6h_data):
        from data_pipeline.labeling import LABEL_COL_T6H
        df_labeled = t6h_data[0]
        assert LABEL_COL_T6H in df_labeled.columns

    def test_has_positive_labels(self, t6h_data):
        from data_pipeline.labeling import LABEL_COL_T6H
        df_labeled = t6h_data[0]
        assert df_labeled[LABEL_COL_T6H].sum() > 0

    def test_no_leakage_train_val(self, t6h_data):
        _, train, val, _ = t6h_data
        overlap = set(train["patient_id"]) & set(val["patient_id"])
        assert len(overlap) == 0

    def test_no_leakage_train_test(self, t6h_data):
        _, train, _, test = t6h_data
        overlap = set(train["patient_id"]) & set(test["patient_id"])
        assert len(overlap) == 0

    def test_all_rows_covered(self, t6h_data):
        df_labeled, train, val, test = t6h_data
        total = len(train) + len(val) + len(test)
        assert total == len(df_labeled)

    def test_label_binary_values(self, t6h_data):
        from data_pipeline.labeling import LABEL_COL_T6H
        df_labeled = t6h_data[0]
        unique = set(df_labeled[LABEL_COL_T6H].unique())
        assert unique.issubset({0, 1})
