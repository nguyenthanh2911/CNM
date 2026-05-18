"""Unit tests cho data_pipeline/labeling.py — T+6h label logic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipeline.labeling import (
    LABEL_COL_T6H,
    create_t6h_labels,
    get_label_stats,
    split_by_patient,
)


def _make_df(
    n_patients: int = 4,
    hours: int = 24,
    interval_minutes: int = 60,
    sepsis_onset_hours: dict | None = None,
) -> pd.DataFrame:
    """
    Tạo DataFrame giả lập để test.
    sepsis_onset_hours: {patient_id: onset_hour} — None = non-sepsis
    """
    from datetime import datetime, timezone, timedelta

    if sepsis_onset_hours is None:
        sepsis_onset_hours = {}

    records = []
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    n_steps = int(hours * 60 / interval_minutes)

    for i in range(n_patients):
        pid = f"P{i+1:04d}"
        onset = sepsis_onset_hours.get(pid)
        has_sepsis = onset is not None

        for step in range(n_steps):
            ts = start + timedelta(minutes=step * interval_minutes)
            records.append({
                "patient_id": pid,
                "timestamp": ts.isoformat(),
                "heart_rate": float(np.random.uniform(60, 120)),
                "systolic_bp": float(np.random.uniform(80, 140)),
                "diastolic_bp": float(np.random.uniform(50, 90)),
                "temperature": float(np.random.uniform(36, 39)),
                "spo2": float(np.random.uniform(88, 100)),
                "respiratory_rate": float(np.random.uniform(12, 30)),
                "lactate": float(np.random.uniform(0.5, 4.0)),
                "wbc": float(np.random.uniform(4, 15)),
                "creatinine": float(np.random.uniform(0.6, 3.0)),
                "bilirubin": float(np.random.uniform(0.1, 3.0)),
                "platelet": float(np.random.uniform(80, 400)),
                "sepsis_label": int(has_sepsis),
                "sepsis_onset_hour": float(onset) if onset is not None else float("nan"),
            })

    return pd.DataFrame(records)


# ────────────────────────────────────────────────────────────────────
class TestCreateT6HLabels:

    def test_nonsepsis_all_zero(self):
        """Non-sepsis patient: toàn bộ label phải = 0."""
        df = _make_df(
            n_patients=2,
            sepsis_onset_hours={}  # không có sepsis
        )
        df_labeled = create_t6h_labels(df)
        assert df_labeled[LABEL_COL_T6H].sum() == 0

    def test_sepsis_has_positive_labels(self):
        """Sepsis patient: phải có ít nhất 1 label = 1."""
        df = _make_df(
            n_patients=2,
            sepsis_onset_hours={"P0001": 12}
        )
        df_labeled = create_t6h_labels(df)
        p1 = df_labeled[df_labeled["patient_id"] == "P0001"]
        assert p1[LABEL_COL_T6H].sum() > 0

    def test_label_window_correct(self):
        """
        Với onset_hour=12, horizon=6:
        - Giờ 6–12: label = 1 (onset trong cửa sổ 6h tới)
        - Giờ 0–5: label = 0 (onset quá xa)
        - Giờ >= 12: label = 0 (đã qua onset)
        """
        df = _make_df(
            n_patients=1,
            hours=24,
            interval_minutes=60,
            sepsis_onset_hours={"P0001": 12}
        )
        df_labeled = create_t6h_labels(df, horizon_hours=6)
        p1 = df_labeled[df_labeled["patient_id"] == "P0001"].copy()
        p1["hour"] = range(len(p1))

        # Giờ 6–11: phải là 1 (trong cửa sổ 6h trước onset=12)
        window_rows = p1[p1["hour"].between(6, 11)]
        assert window_rows[LABEL_COL_T6H].all(), \
            f"Hours 6-11 should be 1, got: {window_rows[LABEL_COL_T6H].tolist()}"

        # Giờ 0–5: phải là 0 (onset quá xa)
        before_rows = p1[p1["hour"] < 6]
        assert (before_rows[LABEL_COL_T6H] == 0).all(), \
            f"Hours 0-5 should be 0"

        # Giờ >= 12: phải là 0 (đã qua onset)
        after_rows = p1[p1["hour"] >= 12]
        assert (after_rows[LABEL_COL_T6H] == 0).all(), \
            f"Hours >= 12 should be 0"

    def test_no_future_leakage(self):
        """
        Label tại giờ T không được dùng dữ liệu tương lai.
        Kiểm tra: label=1 chỉ xuất hiện TRƯỚC onset, không phải SAU.
        """
        df = _make_df(
            n_patients=1,
            hours=24,
            interval_minutes=60,
            sepsis_onset_hours={"P0001": 15}
        )
        df_labeled = create_t6h_labels(df, horizon_hours=6)
        p1 = df_labeled[df_labeled["patient_id"] == "P0001"].copy()
        p1["hour"] = range(len(p1))

        # Không có label=1 tại hoặc sau onset
        after_onset = p1[p1["hour"] >= 15]
        assert (after_onset[LABEL_COL_T6H] == 0).all(), \
            "No label=1 should appear at or after onset"

    def test_label_col_exists(self):
        """Cột LABEL_COL_T6H phải tồn tại sau khi label."""
        df = _make_df(n_patients=2)
        df_labeled = create_t6h_labels(df)
        assert LABEL_COL_T6H in df_labeled.columns

    def test_label_binary(self):
        """Label chỉ có giá trị 0 hoặc 1."""
        df = _make_df(
            n_patients=4,
            sepsis_onset_hours={"P0001": 10, "P0002": 16}
        )
        df_labeled = create_t6h_labels(df)
        unique_vals = set(df_labeled[LABEL_COL_T6H].unique())
        assert unique_vals.issubset({0, 1}), \
            f"Labels should be 0 or 1, got: {unique_vals}"

    def test_row_count_preserved(self):
        """Số dòng không thay đổi sau khi label."""
        df = _make_df(n_patients=4)
        df_labeled = create_t6h_labels(df)
        assert len(df_labeled) == len(df)

    def test_different_horizons(self):
        """Horizon khác nhau cho kết quả label khác nhau."""
        df = _make_df(
            n_patients=1,
            hours=24,
            interval_minutes=60,
            sepsis_onset_hours={"P0001": 12}
        )
        df_3h = create_t6h_labels(df, horizon_hours=3)
        df_6h = create_t6h_labels(df, horizon_hours=6)
        df_12h = create_t6h_labels(df, horizon_hours=12)

        pos_3h  = df_3h[LABEL_COL_T6H].sum()
        pos_6h  = df_6h[LABEL_COL_T6H].sum()
        pos_12h = df_12h[LABEL_COL_T6H].sum()

        # Horizon lớn hơn → nhiều label=1 hơn
        assert pos_3h <= pos_6h <= pos_12h, \
            f"Expected pos_3h({pos_3h}) <= pos_6h({pos_6h}) <= pos_12h({pos_12h})"


# ────────────────────────────────────────────────────────────────────
class TestSplitByPatient:

    def _make_labeled_df(self) -> pd.DataFrame:
        df = _make_df(
            n_patients=10,
            sepsis_onset_hours={
                "P0001": 10, "P0002": 14, "P0003": 8,
                "P0004": 16, "P0005": 12,
            }
        )
        return create_t6h_labels(df)

    def test_no_patient_overlap_train_val(self):
        """Không có patient nào vừa ở train vừa ở val."""
        df = self._make_labeled_df()
        train, val, test = split_by_patient(df)
        overlap = set(train["patient_id"]) & set(val["patient_id"])
        assert len(overlap) == 0, f"Train/val overlap: {overlap}"

    def test_no_patient_overlap_train_test(self):
        """Không có patient nào vừa ở train vừa ở test."""
        df = self._make_labeled_df()
        train, val, test = split_by_patient(df)
        overlap = set(train["patient_id"]) & set(test["patient_id"])
        assert len(overlap) == 0, f"Train/test overlap: {overlap}"

    def test_no_patient_overlap_val_test(self):
        """Không có patient nào vừa ở val vừa ở test."""
        df = self._make_labeled_df()
        train, val, test = split_by_patient(df)
        overlap = set(val["patient_id"]) & set(test["patient_id"])
        assert len(overlap) == 0, f"Val/test overlap: {overlap}"

    def test_all_patients_covered(self):
        """Tất cả patient phải xuất hiện trong 1 trong 3 tập."""
        df = self._make_labeled_df()
        train, val, test = split_by_patient(df)
        all_pids = set(df["patient_id"].unique())
        covered  = (
            set(train["patient_id"]) |
            set(val["patient_id"]) |
            set(test["patient_id"])
        )
        assert all_pids == covered, \
            f"Missing patients: {all_pids - covered}"

    def test_total_rows_preserved(self):
        """Tổng số dòng sau split = tổng ban đầu."""
        df = self._make_labeled_df()
        train, val, test = split_by_patient(df)
        assert len(train) + len(val) + len(test) == len(df)

    def test_train_larger_than_val_test(self):
        """Train set phải lớn hơn val và test."""
        df = self._make_labeled_df()
        train, val, test = split_by_patient(df)
        assert len(train) > len(val)
        assert len(train) > len(test)


# ────────────────────────────────────────────────────────────────────
class TestGetLabelStats:

    def test_returns_required_keys(self):
        """Kết quả phải có đủ 5 keys."""
        df = _make_df(n_patients=2)
        df = create_t6h_labels(df)
        stats = get_label_stats(df)
        for key in ["total", "positive", "negative",
                    "positive_ratio", "imbalance_ratio"]:
            assert key in stats, f"Missing key: {key}"

    def test_total_equals_rows(self):
        df = _make_df(n_patients=2)
        df = create_t6h_labels(df)
        stats = get_label_stats(df)
        assert stats["total"] == len(df)

    def test_positive_plus_negative_equals_total(self):
        df = _make_df(
            n_patients=2,
            sepsis_onset_hours={"P0001": 10}
        )
        df = create_t6h_labels(df)
        stats = get_label_stats(df)
        assert stats["positive"] + stats["negative"] == stats["total"]

    def test_positive_ratio_between_0_and_1(self):
        df = _make_df(
            n_patients=4,
            sepsis_onset_hours={"P0001": 10}
        )
        df = create_t6h_labels(df)
        stats = get_label_stats(df)
        assert 0.0 <= stats["positive_ratio"] <= 1.0
