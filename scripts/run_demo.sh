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
python scripts/seed_patients.py

python data_pipeline/data_generator.py --mode csv --patients 20 --hours 24
python ml/train.py --data data/synthetic/icu_data_synthetic.csv --experiment-name demo

echo "Starting data stream..."
python data_pipeline/data_generator.py --mode stream --interval 300 &

echo "Demo running! Open http://localhost:8000"
