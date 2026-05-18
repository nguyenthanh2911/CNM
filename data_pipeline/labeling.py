"""
T+6h Labeling module cho bài toán dự đoán sớm Sepsis.

Công thức:
    y[t] = 1  nếu sepsis_onset xảy ra trong khoảng (t_hour, t_hour + horizon_hours]
    y[t] = 0  còn lại (bao gồm cả sau khi onset đã xảy ra)

Tránh leakage:
    - Features tại T chỉ dùng dữ liệu từ T trở về trước
    - Split theo patient_id, không phải theo dòng
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


HORIZON_HOURS: int = 6          # Dự đoán trước bao nhiêu giờ
LABEL_COL_T6H: str = "sepsis_in_next_6h"


def create_t6h_labels(
    df: pd.DataFrame,
    horizon_hours: int = HORIZON_HOURS,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    onset_col: str = "sepsis_onset_hour",
) -> pd.DataFrame:
    """
    Tạo label sepsis_in_next_6h cho từng dòng dữ liệu.

    Args:
        df: DataFrame có cột patient_id, timestamp, sepsis_onset_hour
        horizon_hours: Cửa sổ dự đoán (mặc định 6h)
        patient_col: Tên cột patient id
        time_col: Tên cột timestamp
        onset_col: Tên cột giờ onset sepsis

    Returns:
        DataFrame gốc + cột sepsis_in_next_6h (0 hoặc 1)
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values([patient_col, time_col]).reset_index(drop=True)
    df[LABEL_COL_T6H] = 0

    for pid, group in df.groupby(patient_col, sort=False):
        onset_hour = group[onset_col].iloc[0]

        # Non-sepsis hoặc không có onset → toàn bộ label = 0
        if pd.isna(onset_hour):
            continue

        onset_hour = float(onset_hour)

        # Tính giờ hiện tại của mỗi dòng từ timestamp đầu tiên
        t0 = group[time_col].iloc[0]
        hours_elapsed = (group[time_col] - t0).dt.total_seconds() / 3600.0

        # Label = 1 nếu: onset nằm trong (current_hour, current_hour + horizon]
        # Tức là: current_hour < onset_hour <= current_hour + horizon_hours
        # Và: chưa xảy ra onset (current_hour < onset_hour)
        mask = (
            (hours_elapsed < onset_hour) &                          # chưa onset
            (hours_elapsed + horizon_hours >= onset_hour)           # onset trong cửa sổ
        )
        df.loc[group.index[mask], LABEL_COL_T6H] = 1

    return df


def split_by_patient(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    patient_col: str = "patient_id",
    label_col: str = LABEL_COL_T6H,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split train/val/test theo patient_id để tránh data leakage.

    Mỗi bệnh nhân chỉ xuất hiện trong 1 trong 3 tập.
    Cố gắng giữ tỷ lệ sepsis/non-sepsis cân bằng giữa các tập.

    Returns:
        (train_df, val_df, test_df)
    """
    rng = np.random.default_rng(seed=random_seed)

    # Lấy danh sách patient + có sepsis không
    patient_info = (
        df.groupby(patient_col)["sepsis_label"]
        .max()
        .reset_index()
        .rename(columns={"sepsis_label": "has_sepsis"})
    )

    sepsis_patients    = patient_info[patient_info["has_sepsis"] == 1][patient_col].values
    nonsepsis_patients = patient_info[patient_info["has_sepsis"] == 0][patient_col].values

    def _split_group(patients: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        patients = rng.permutation(patients)
        n = len(patients)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))
        test   = patients[:n_test]
        val    = patients[n_test:n_test + n_val]
        train  = patients[n_test + n_val:]
        return train, val, test

    sep_train, sep_val, sep_test       = _split_group(sepsis_patients)
    non_train, non_val, non_test       = _split_group(nonsepsis_patients)

    train_ids = np.concatenate([sep_train, non_train])
    val_ids   = np.concatenate([sep_val,   non_val])
    test_ids  = np.concatenate([sep_test,  non_test])

    train_df = df[df[patient_col].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df[patient_col].isin(val_ids)].reset_index(drop=True)
    test_df  = df[df[patient_col].isin(test_ids)].reset_index(drop=True)

    # In thống kê
    def _stats(name: str, d: pd.DataFrame) -> None:
        n_pat = d[patient_col].nunique()
        ratio = d[label_col].mean() if label_col in d.columns else float("nan")
        print(f"{name:6s}: {len(d):5,} rows | {n_pat:2} patients | "
              f"label_ratio={ratio:.3f}")

    print("=== Patient-based split ===")
    _stats("Train", train_df)
    _stats("Val",   val_df)
    _stats("Test",  test_df)

    return train_df, val_df, test_df


def get_label_stats(df: pd.DataFrame, label_col: str = LABEL_COL_T6H) -> dict:
    """Trả về thống kê label để kiểm tra imbalance."""
    total  = len(df)
    pos    = int(df[label_col].sum())
    neg    = total - pos
    ratio  = pos / total if total > 0 else 0.0
    return {
        "total": total,
        "positive": pos,
        "negative": neg,
        "positive_ratio": round(ratio, 4),
        "imbalance_ratio": round(neg / max(pos, 1), 2),
    }
