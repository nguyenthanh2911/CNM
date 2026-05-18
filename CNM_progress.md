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

---

### Bước 6 — Xóa code logic cũ (Refactor)

**Mục tiêu:** Dọn sạch toàn bộ code liên quan đến logic cũ,
chỉ giữ lại logic T+6h.

| Xóa | File | Chi tiết |
|-----|------|----------|
| Logic `early_warning_label` trong `_build_record()` | `data_generator.py` | Xóa 8 dòng rule-based cứng giờ 10–12 |
| `early_warning_label` khỏi `desired_cols` x2 | `data_generator.py` | Xóa khỏi `generate_csv()` và `generate_dataframe()` |
| `LABEL_COL_LEGACY = "sepsis_label"` | `train.py` | Xóa reference không dùng |

**Kết quả sau refactor:**
- DataFrame chỉ còn `sepsis_label` + `sepsis_onset_hour` — không còn `early_warning_label`
- `train.py` chỉ có 1 label duy nhất: `LABEL_COL = LABEL_COL_T6H`
- Tất cả 69 tests vẫn pass

---

## 📊 Tổng kết nhánh này — T+6h Implementation

### Toàn bộ commits theo thứ tự

| Commit | Message | Nội dung |
|--------|---------|----------|
| feat(t6h-step1) | add sepsis_onset_hour | PhysiologicalModel có onset động 8–18h, severity_factor dùng onset thực |
| feat(t6h-step2) | add labeling.py | create_t6h_labels, split_by_patient, get_label_stats |
| feat(t6h-step3) | rewrite train.py | T+6h label, patient-based split, auto SMOTE ratio=0.4 |
| feat(t6h-step4) | add tests | 18 unit tests + 6 integration tests |
| feat(t6h-step5) | docs | Progress log |
| refactor(t6h-step6) | remove legacy code | Xóa early_warning_label, LABEL_COL_LEGACY |

### Files thay đổi so với main

| File | Loại |
|------|------|
| `data_pipeline/data_generator.py` | Modified |
| `data_pipeline/labeling.py` | **New** |
| `ml/train.py` | Modified |
| `tests/unit/test_labeling.py` | **New** |
| `tests/integration/test_pipeline.py` | Modified |
| `CNM_progress.md` | Modified |

### Kết quả model T+6h vs label cũ

| | Label cũ | Label T+6h |
|---|---|---|
| Loại label | Per-patient tĩnh | Per-timestep động |
| Ý nghĩa | Bệnh nhân có Sepsis không? | Sẽ bị Sepsis trong 6h không? |
| Split | Random rows | Patient-based (no leakage) |
| Positive ratio | ~40% | ~10% |
| SMOTE | Optional | Auto khi ratio > 5:1 |
| Test AUROC | ~0.85+ | 0.8270 |
| Sensitivity | - | 79% |
| Cảnh báo sớm | ❌ | ✅ trước onset 6h |
