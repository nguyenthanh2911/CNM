from __future__ import annotations

import os
import random
from typing import List, Tuple

import psycopg2
from dotenv import load_dotenv


VIETNAMESE_NAMES: List[str] = [
    "Nguyễn Văn An",
    "Trần Thị Bình",
    "Lê Văn Cường",
    "Phạm Thị Dung",
    "Hoàng Văn Đức",
    "Vũ Thị Hạnh",
    "Đặng Văn Hùng",
    "Bùi Thị Lan",
    "Ngô Văn Minh",
    "Đỗ Thị Nga",
    "Hồ Văn Phúc",
    "Dương Thị Quỳnh",
    "Lý Văn Sơn",
    "Mai Thị Trang",
    "Đinh Văn Tuấn",
    "Chu Thị Vân",
    "Tạ Văn Xuyên",
    "Ninh Thị Yến",
    "Cao Văn Khoa",
    "Hà Thị Liên",
    "Tôn Văn Nam",
    "Lâm Thị Oanh",
    "Kiều Văn Quang",
    "Phan Thị Thu",
]


def _conn_params_from_env() -> dict:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    user = os.getenv("POSTGRES_USER", "sepsis_user")
    password = os.getenv("POSTGRES_PASSWORD", "sepsis_pass")
    db = os.getenv("POSTGRES_DB", "sepsis_db")
    return {"host": host, "port": port, "user": user, "password": password, "dbname": db}


def main() -> None:
    load_dotenv()

    params = _conn_params_from_env()

    patients: List[Tuple[str, str, int, str, str]] = []
    for i in range(1, 21):
        pid = f"P{i:03d}"
        name = random.choice(VIETNAMESE_NAMES)
        age = random.randint(45, 80)
        gender = random.choice(["M", "F"])
        ward = random.choice(["ICU-1", "ICU-2"])
        patients.append((pid, name, age, gender, ward))

    with psycopg2.connect(**params) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO patients (patient_id, name, age, gender, ward)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (patient_id) DO NOTHING
                """,
                patients,
            )
        conn.commit()

    print("Seeded 20 patients (P001..P020)")


if __name__ == "__main__":
    main()
