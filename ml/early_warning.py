from __future__ import annotations
from collections import deque
from typing import Any, Dict, List, Optional
import numpy as np


class EarlyWarningPredictor:
    """
    Dự đoán nguy cơ sepsis trong 30 phút tới dựa trên 3 yếu tố:
    1. Trend score         : xu hướng vitals đang xấu đi
    2. Rate of change score: tốc độ thay đổi các chỉ số
    3. Threshold score     : mức độ gần ngưỡng nguy hiểm lâm sàng
    """

    DANGER_THRESHOLDS = {
        'heart_rate':        {'high': 100,  'critical': 120},
        'systolic_bp':       {'low':  100,  'critical': 85},
        'temperature':       {'high': 38.3, 'critical': 39.0},
        'spo2':              {'low':  95,   'critical': 92},
        'respiratory_rate':  {'high': 20,   'critical': 25},
        'lactate':           {'high': 2.0,  'critical': 4.0},
        'wbc':               {'high': 12,   'critical': 18},
        'creatinine':        {'high': 1.2,  'critical': 2.0},
        'platelet':          {'low':  150,  'critical': 100},
    }

    def __init__(self) -> None:
        self.history: Dict[str, deque] = {}
        self.maxlen: int = 6  # 6 records x 5 phút = 30 phút history

    def update(self, patient_id: str, vitals: Dict[str, Any]) -> None:
        """Thêm record vitals mới vào history của bệnh nhân."""
        if patient_id not in self.history:
            self.history[patient_id] = deque(maxlen=self.maxlen)
        self.history[patient_id].append(vitals)

    def _safe_float(self, value: Any) -> Optional[float]:
        """Chuyển đổi an toàn sang float, trả về None nếu không hợp lệ."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _trend_score(self, patient_id: str) -> float:
        """
        Tính xu hướng xấu đi bằng cách so sánh record đầu và cuối
        trong 30 phút qua.

        Trả về: 0.0 (không đổi) đến 1.0 (xấu hoàn toàn)
        """
        records = list(self.history.get(patient_id, []))
        if len(records) < 2:
            return 0.0

        first = records[0]
        last  = records[-1]
        bad_trends = 0
        total      = 0

        # Heart rate tăng → xấu
        hr_first = self._safe_float(first.get('heart_rate'))
        hr_last  = self._safe_float(last.get('heart_rate'))
        if hr_first is not None and hr_last is not None:
            if hr_last > hr_first + 5:
                bad_trends += 1
            total += 1

        # Systolic BP giảm → xấu
        sbp_first = self._safe_float(first.get('systolic_bp'))
        sbp_last  = self._safe_float(last.get('systolic_bp'))
        if sbp_first is not None and sbp_last is not None:
            if sbp_last < sbp_first - 5:
                bad_trends += 1
            total += 1

        # Temperature tăng → xấu
        temp_first = self._safe_float(first.get('temperature'))
        temp_last  = self._safe_float(last.get('temperature'))
        if temp_first is not None and temp_last is not None:
            if temp_last > temp_first + 0.3:
                bad_trends += 1
            total += 1

        # SpO2 giảm → xấu
        spo2_first = self._safe_float(first.get('spo2'))
        spo2_last  = self._safe_float(last.get('spo2'))
        if spo2_first is not None and spo2_last is not None:
            if spo2_last < spo2_first - 1:
                bad_trends += 1
            total += 1

        # Respiratory rate tăng → xấu
        rr_first = self._safe_float(first.get('respiratory_rate'))
        rr_last  = self._safe_float(last.get('respiratory_rate'))
        if rr_first is not None and rr_last is not None:
            if rr_last > rr_first + 2:
                bad_trends += 1
            total += 1

        # Lactate tăng → xấu
        lac_first = self._safe_float(first.get('lactate'))
        lac_last  = self._safe_float(last.get('lactate'))
        if lac_first is not None and lac_last is not None:
            if lac_last > lac_first + 0.3:
                bad_trends += 1
            total += 1

        # Platelet giảm → xấu
        plt_first = self._safe_float(first.get('platelet'))
        plt_last  = self._safe_float(last.get('platelet'))
        if plt_first is not None and plt_last is not None:
            if plt_last < plt_first - 20:
                bad_trends += 1
            total += 1

        return float(bad_trends / max(total, 1))

    def _rate_of_change_score(self, patient_id: str) -> float:
        """
        Tính tốc độ thay đổi nguy hiểm (đạo hàm bậc 1).

        Trả về: 0.0 (ổn định) đến 1.0 (thay đổi cực nhanh)
        """
        records = list(self.history.get(patient_id, []))
        if len(records) < 2:
            return 0.0

        scores: List[float] = []

        # Heart rate: thay đổi > 10 bpm/interval là nguy hiểm
        hrs = [self._safe_float(r.get('heart_rate')) for r in records]
        hrs = [v for v in hrs if v is not None]
        if len(hrs) >= 2:
            roc = abs(hrs[-1] - hrs[-2])
            scores.append(min(roc / 20.0, 1.0))

        # Systolic BP: thay đổi > 10 mmHg/interval là nguy hiểm
        sbps = [self._safe_float(r.get('systolic_bp')) for r in records]
        sbps = [v for v in sbps if v is not None]
        if len(sbps) >= 2:
            roc = abs(sbps[-1] - sbps[-2])
            scores.append(min(roc / 20.0, 1.0))

        # SpO2: thay đổi > 2%/interval là nguy hiểm
        spo2s = [self._safe_float(r.get('spo2')) for r in records]
        spo2s = [v for v in spo2s if v is not None]
        if len(spo2s) >= 2:
            roc = abs(spo2s[-1] - spo2s[-2])
            scores.append(min(roc / 5.0, 1.0))

        # Lactate: thay đổi > 0.5/interval là nguy hiểm
        lacs = [self._safe_float(r.get('lactate')) for r in records]
        lacs = [v for v in lacs if v is not None]
        if len(lacs) >= 2:
            roc = abs(lacs[-1] - lacs[-2])
            scores.append(min(roc / 2.0, 1.0))

        # Respiratory rate: thay đổi > 4/interval là nguy hiểm
        rrs = [self._safe_float(r.get('respiratory_rate')) for r in records]
        rrs = [v for v in rrs if v is not None]
        if len(rrs) >= 2:
            roc = abs(rrs[-1] - rrs[-2])
            scores.append(min(roc / 10.0, 1.0))

        # Temperature: thay đổi > 0.5°C/interval là nguy hiểm
        temps = [self._safe_float(r.get('temperature')) for r in records]
        temps = [v for v in temps if v is not None]
        if len(temps) >= 2:
            roc = abs(temps[-1] - temps[-2])
            scores.append(min(roc / 1.0, 1.0))

        return float(np.mean(scores)) if scores else 0.0

    def _threshold_score(self, vitals: Dict[str, Any]) -> float:
        """
        Tính mức độ gần ngưỡng nguy hiểm lâm sàng.

        Trả về: 0.0 (an toàn) đến 1.0 (vượt ngưỡng critical)
        """
        scores: List[float] = []

        hr = self._safe_float(vitals.get('heart_rate'))
        if hr is not None:
            if hr >= 120:    scores.append(1.0)
            elif hr >= 100:  scores.append(0.5)
            else:            scores.append(0.0)

        sbp = self._safe_float(vitals.get('systolic_bp'))
        if sbp is not None:
            if sbp <= 85:    scores.append(1.0)
            elif sbp <= 100: scores.append(0.5)
            else:            scores.append(0.0)

        temp = self._safe_float(vitals.get('temperature'))
        if temp is not None:
            if temp >= 39.0:   scores.append(1.0)
            elif temp >= 38.3: scores.append(0.5)
            else:              scores.append(0.0)

        spo2 = self._safe_float(vitals.get('spo2'))
        if spo2 is not None:
            if spo2 <= 92:  scores.append(1.0)
            elif spo2 <= 95: scores.append(0.5)
            else:            scores.append(0.0)

        rr = self._safe_float(vitals.get('respiratory_rate'))
        if rr is not None:
            if rr >= 25:  scores.append(1.0)
            elif rr >= 20: scores.append(0.5)
            else:          scores.append(0.0)

        lac = self._safe_float(vitals.get('lactate'))
        if lac is not None:
            if lac >= 4.0:  scores.append(1.0)
            elif lac >= 2.0: scores.append(0.5)
            else:            scores.append(0.0)

        wbc = self._safe_float(vitals.get('wbc'))
        if wbc is not None:
            if wbc >= 18:  scores.append(1.0)
            elif wbc >= 12: scores.append(0.5)
            else:           scores.append(0.0)

        creat = self._safe_float(vitals.get('creatinine'))
        if creat is not None:
            if creat >= 2.0:  scores.append(1.0)
            elif creat >= 1.2: scores.append(0.5)
            else:              scores.append(0.0)

        plt = self._safe_float(vitals.get('platelet'))
        if plt is not None:
            if plt <= 100:  scores.append(1.0)
            elif plt <= 150: scores.append(0.5)
            else:            scores.append(0.0)

        return float(np.mean(scores)) if scores else 0.0

    def _get_contributing_factors(
        self, patient_id: str, current_vitals: Dict[str, Any]
    ) -> List[str]:
        """
        Phân tích và liệt kê các yếu tố nguy cơ cụ thể.
        """
        factors: List[str] = []
        records = list(self.history.get(patient_id, []))

        # Kiểm tra ngưỡng hiện tại
        lac = self._safe_float(current_vitals.get('lactate'))
        if lac is not None and lac >= 2.0:
            factors.append(f"Lactate cao ({lac:.1f} mmol/L)")

        spo2 = self._safe_float(current_vitals.get('spo2'))
        if spo2 is not None and spo2 <= 95:
            factors.append(f"SpO2 thấp ({spo2:.1f}%)")

        hr = self._safe_float(current_vitals.get('heart_rate'))
        if hr is not None and hr >= 100:
            factors.append(f"Nhịp tim cao ({hr:.0f} bpm)")

        sbp = self._safe_float(current_vitals.get('systolic_bp'))
        if sbp is not None and sbp <= 100:
            factors.append(f"Huyết áp thấp ({sbp:.0f} mmHg)")

        temp = self._safe_float(current_vitals.get('temperature'))
        if temp is not None and temp >= 38.3:
            factors.append(f"Sốt cao ({temp:.1f}°C)")

        rr = self._safe_float(current_vitals.get('respiratory_rate'))
        if rr is not None and rr >= 20:
            factors.append(f"Nhịp thở tăng ({rr:.0f} lần/phút)")

        wbc = self._safe_float(current_vitals.get('wbc'))
        if wbc is not None and wbc >= 12:
            factors.append(f"Bạch cầu tăng ({wbc:.1f} K/μL)")

        # Kiểm tra xu hướng (so sánh đầu và cuối)
        if len(records) >= 2:
            first = records[0]
            last  = records[-1]

            lac_f = self._safe_float(first.get('lactate'))
            lac_l = self._safe_float(last.get('lactate'))
            if lac_f and lac_l and lac_l > lac_f + 0.5:
                factors.append(f"Lactate tăng nhanh (+{lac_l - lac_f:.1f} trong 30 phút)")

            spo2_f = self._safe_float(first.get('spo2'))
            spo2_l = self._safe_float(last.get('spo2'))
            if spo2_f and spo2_l and spo2_l < spo2_f - 2:
                factors.append(f"SpO2 giảm liên tục ({spo2_f:.1f}% → {spo2_l:.1f}%)")

            hr_f = self._safe_float(first.get('heart_rate'))
            hr_l = self._safe_float(last.get('heart_rate'))
            if hr_f and hr_l and hr_l > hr_f + 10:
                factors.append(f"Nhịp tim tăng nhanh (+{hr_l - hr_f:.0f} trong 30 phút)")

            sbp_f = self._safe_float(first.get('systolic_bp'))
            sbp_l = self._safe_float(last.get('systolic_bp'))
            if sbp_f and sbp_l and sbp_l < sbp_f - 10:
                factors.append(f"Huyết áp tụt nhanh (-{sbp_f - sbp_l:.0f} trong 30 phút)")

        # Loại bỏ trùng lặp, giữ tối đa 5
        seen = set()
        unique_factors = []
        for f in factors:
            if f not in seen:
                seen.add(f)
                unique_factors.append(f)
            if len(unique_factors) >= 5:
                break

        return unique_factors

    def predict_early_warning(
        self,
        patient_id: str,
        current_vitals: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Kết hợp 3 scores để tính xác suất sepsis trong 30 phút tới.

        Returns dict:
        {
            "early_warning_probability": float,   # 0.0 - 1.0
            "early_warning_level": str,           # LOW / MEDIUM / HIGH
            "time_window_minutes": int,           # = 30
            "trend_score": float,
            "rate_of_change_score": float,
            "threshold_score": float,
            "contributing_factors": List[str],
        }
        """
        # Cập nhật history trước
        self.update(patient_id, current_vitals)

        # Tính 3 scores độc lập
        trend_score  = self._trend_score(patient_id)
        roc_score    = self._rate_of_change_score(patient_id)
        thresh_score = self._threshold_score(current_vitals)

        # Weighted combination:
        # threshold quan trọng nhất (0.5) vì phản ánh trạng thái hiện tại
        # trend quan trọng thứ 2 (0.3) vì phản ánh hướng thay đổi
        # rate of change ít nhất (0.2) vì có thể nhiễu
        probability = (
            thresh_score * 0.50 +
            trend_score  * 0.30 +
            roc_score    * 0.20
        )
        probability = float(np.clip(probability, 0.0, 1.0))

        # Phân mức độ
        if probability >= 0.70:
            level = "HIGH"
        elif probability >= 0.40:
            level = "MEDIUM"
        else:
            level = "LOW"

        # Lấy các yếu tố nguy cơ
        factors = self._get_contributing_factors(patient_id, current_vitals)

        return {
            "early_warning_probability": round(probability, 4),
            "early_warning_level":       level,
            "time_window_minutes":       30,
            "trend_score":               round(trend_score, 4),
            "rate_of_change_score":      round(roc_score, 4),
            "threshold_score":           round(thresh_score, 4),
            "contributing_factors":      factors,
        }
