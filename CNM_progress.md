# CNM — ICU Sepsis Early Warning System

## 🔬 Hướng 3 — T+6h Early Warning Labeling (nhánh: update_t6h)

### Mục tiêu
Chuyển từ label tĩnh per-patient sang dự đoán thực sự sớm:
`sepsis_in_next_6h = 1` nếu Sepsis onset xảy ra trong 6h tới.

### Các bước đã hoàn thành

| Bước | File | Thay đổi | Commit |
|------|------|----------|--------|
| 1 | `data_pipeline/data_generator.py` | Thêm `sepsis_onset_hour` vào PhysiologicalModel, gán random onset 8–18h, cập nhật `_severity_factor()` dùng onset động | feat(t6h-step1) |
| 2 | `data_pipeline/labeling.py` | Tạo mới: `create_t6h_labels()`, `split_by_patient()`, `get_label_stats()` | feat(t6h-step2) |
| 3 | `ml/train.py` | Rewrite: dùng `LABEL_COL_T6H`, patient-based split, auto SMOTE ratio=0.4, DataFrame fix cho XGBoost | feat(t6h-step3) |
| 4 | `tests/unit/test_labeling.py` | Tạo mới: 18 tests — window logic, no-leakage, binary labels, horizons | feat(t6h-step4) |
| 4 | `tests/integration/test_pipeline.py` | Thêm `TestT6HLabelingIntegration` — 6 tests end-to-end | feat(t6h-step4) |

### Kết quả model T+6h

| Metric | Giá trị |
|--------|---------|
| CV AUROC | 0.8486 ± 0.0144 |
| Train AUROC | 0.8969 |
| Val AUROC | 0.6316 |
| Test AUROC | 0.8270 |
| Gap train-test | 0.0699 (dưới ngưỡng) |
| Sensitivity | 79% |
| Specificity | 71% |
| Label positive ratio | 10% |
| Imbalance ratio | 9:1 → SMOTE applied |

### So sánh với label cũ

| | Label cũ (`sepsis_label`) | Label mới (`sepsis_in_next_6h`) |
|---|---|---|
| Loại | Per-patient tĩnh | Per-timestep động |
| Ý nghĩa | "BN này có Sepsis không?" | "BN này sẽ bị Sepsis trong 6h không?" |
| Train/test split | Random rows | Patient-based (no leakage) |
| Imbalance | ~40% positive | ~10% positive |
| SMOTE | Optional | Auto-apply khi ratio > 5 |
| Cảnh báo sớm | ❌ Không thực sự sớm | ✅ Trước onset 6h |

### Files thay đổi trên nhánh update_t6h
- `data_pipeline/data_generator.py`
- `data_pipeline/labeling.py` *(mới)*
- `ml/train.py`
- `tests/unit/test_labeling.py` *(mới)*
- `tests/integration/test_pipeline.py`
