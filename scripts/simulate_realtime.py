#!/usr/bin/env python3
""" ICU Realtime Simulation — Dựa trên dữ liệu từ CSV.
Đọc file data/raw/real_time_icu.csv, trong đó mỗi giá trị `timestamp` (0, 1, 2...)
tương ứng với các mốc giờ 0h, 1h, 2h... mô phỏng.
"""

import csv
import time
import concurrent.futures
from datetime import datetime, timezone
import httpx
from pathlib import Path
from collections import defaultdict

ML_SERVICE_URL = "http://ml_service:8001/vitals"
STEP_SLEEP     = 10  # Mỗi bước cách nhau 10 giây trong thực tế
DATA_FILE      = Path(__file__).parent.parent / "data" / "raw" / "real_time_icu.csv"

def send_patient(pid: str, vitals: dict, real_timestamp: str) -> str:
    payload = {
        "patient_id": pid,
        "timestamp": real_timestamp,
        **vitals
    }
    
    try:
        with httpx.Client(timeout=8.0) as c:
            r = c.post(ML_SERVICE_URL, json=payload)
            if r.status_code == 200:
                d  = r.json()
                ew = d.get("early_warning", {})
                return (f"{pid:5s} | {d.get('risk_level','?'):8s} "
                        f"score={d.get('risk_score',0):.3f} "
                        f"EW={ew.get('early_warning_level','?')}"
                        f"({ew.get('early_warning_probability',0)*100:.0f}%)")
            return f"{pid:5s} | HTTP {r.status_code}"
    except Exception as e:
        return f"{pid:5s} | ERROR: {e}"

def send_all(patients_data: list, step: int):
    real_timestamp = datetime.now(timezone.utc).isoformat()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        futs = []
        for row in patients_data:
            pid = row["patient_id"]
            
            # Đồng bộ ID: P0001 -> P001 để khớp với danh sách bệnh nhân đã seed
            if pid.startswith("P") and len(pid) > 4:
                try:
                    pid = f"P{int(pid[1:]):03d}"
                except ValueError:
                    pass
            
            # Trích xuất vitals
            vitals = {}
            for k, v in row.items():
                if k not in ["patient_id", "timestamp"]:
                    try:
                        vitals[k] = float(v)
                    except ValueError:
                        pass # Bỏ qua nếu dữ liệu trống hoặc không phải dạng số

            futs.append(ex.submit(send_patient, pid, vitals, real_timestamp))
            
        for f in concurrent.futures.as_completed(futs):
            print(" ", f.result())

def main():
    if not DATA_FILE.exists():
        print(f"File not found: {DATA_FILE}")
        return

    print("=" * 65)
    print(f" Loading ICU Data from {DATA_FILE.name}")
    
    # Nhóm dữ liệu theo timestamp
    data_by_ts = defaultdict(list)
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "timestamp" in row:
                try:
                    ts = int(float(row["timestamp"]))
                    data_by_ts[ts].append(row)
                except ValueError:
                    pass

    if not data_by_ts:
        print("Error: No data or missing 'timestamp' column.")
        return
        
    timestamps = sorted(data_by_ts.keys())
    total_steps = len(timestamps)
    
    print(f" Found {total_steps} unique timestamps (hours).")
    print(f" Total simulated runtime: {total_steps} steps x {STEP_SLEEP}s = {total_steps*STEP_SLEEP}s")
    print("=" * 65)

    for i, ts in enumerate(timestamps):
        patients_data = data_by_ts[ts]
        print(f"\n[Step {i+1:03d}/{total_steps} | Simulated Hour: {ts}h | {datetime.now().strftime('%H:%M:%S')}]")
        
        send_all(patients_data, ts)
        
        if i < total_steps - 1:
            time.sleep(STEP_SLEEP)

    print("\nSimulation complete.")

if __name__ == "__main__":
    main()
