import asyncio
import httpx
import random
from datetime import datetime, timezone

ML_URL = "http://localhost:8001/vitals"
PATIENTS = [f"P{i:04d}" for i in range(1, 21)]
ROUNDS = 10

# Seed cố định để tái lập kết quả
RNG = random.Random(42)
async def send_vitals(client: httpx.AsyncClient, patient_id: str, round_idx: int):
    progress = round_idx / ROUNDS  # 0.0 → 1.0

    # Mỗi bệnh nhân có offset riêng dựa trên patient_id
    pid_num = int(patient_id[1:])  # P0001 → 1, P0020 → 20
    offset = (pid_num - 10) / 20   # -0.45 → +0.5

    # Thêm nhiễu ngẫu nhiên ±10% riêng cho mỗi bệnh nhân
    noise = RNG.uniform(-0.1, 0.1)

    payload = {
        "patient_id": patient_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "heart_rate": 100 + (progress + offset + noise) * 30,
        "systolic_bp": 90 - (progress + offset + noise) * 15,
        "diastolic_bp": 60 - (progress + offset + noise) * 15,
        "temperature": 38.5 + (progress + offset + noise) * 1.0,
        "spo2": 94 - (progress + offset + noise) * 6,
        "respiratory_rate": 22 + (progress + offset + noise) * 6,
        "lactate": 3.0 + (progress + offset + noise) * 1.5,
        "wbc": 14.0 + (progress + offset + noise) * 4.0,
        "creatinine": 2.0 + (progress + offset + noise) * 1.5,
        "bilirubin": 2.5 + (progress + offset + noise) * 1.0,
        "platelet": 120 - (progress + offset + noise) * 40,
    }

    try:
        resp = await client.post(ML_URL, json=payload, timeout=5.0)
        data = resp.json()
        risk = data.get("risk_score", 0)
        ew = data.get("early_warning", {})
        level = ew.get("early_warning_level", "?")
        prob = ew.get("early_warning_probability", 0)
        print(f"  {patient_id} round {round_idx+1:02d}: risk={risk:.3f}  ew={level} ({prob*100:.0f}%)")
    except Exception as e:
        print(f"  {patient_id} ERROR: {e}")


async def main():
    print("=== Seeding Early Warning Demo Data ===")
    print(f"20 patients x {ROUNDS} rounds worsening vitals\n")

    async with httpx.AsyncClient() as client:
        for round_idx in range(ROUNDS):
            print(f"\n--- Round {round_idx+1}/{ROUNDS} ---")
            tasks = [send_vitals(client, pid, round_idx) for pid in PATIENTS]
            await asyncio.gather(*tasks)
            await asyncio.sleep(1.5)

    print("\n=== Done! Open http://localhost:8000 to see results ===")


if __name__ == "__main__":
    asyncio.run(main())

