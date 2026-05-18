#!/bin/bash
set -euo pipefail

# Load .env if present
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

docker compose up -d

sleep 15

bash scripts/setup_db.sh
docker compose exec -T ml_service python scripts/seed_patients.py

docker compose exec -T ml_service python -m data_pipeline.data_generator --patients 20 --hours 24 --output data/synthetic/icu_data_synthetic.csv
docker compose exec -T ml_service python -m ml.train --data data/synthetic/icu_data_synthetic.csv --experiment-name CNM-Sepsis-T6H --model-name sepsis_xgboost_t6h --augment

echo "Starting realtime simulation (20 patients)..."
docker compose exec -d ml_service python scripts/simulate_realtime.py
echo "Demo running! Open http://localhost:8000"
echo "View logs: docker compose logs -f ml_service"

