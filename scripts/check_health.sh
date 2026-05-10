#!/bin/bash
set -euo pipefail

ok=0

total=4

if curl -fsS http://localhost:8001/health >/dev/null; then
  echo "ML Service: OK"
  ok=$((ok+1))
else
  echo "ML Service: FAIL"
fi

if curl -fsS http://localhost:8002/health >/dev/null; then
  echo "Alert Service: OK"
  ok=$((ok+1))
else
  echo "Alert Service: FAIL"
fi

if curl -fsS http://localhost:8000/ >/dev/null; then
  echo "Django Dashboard: OK"
  ok=$((ok+1))
else
  echo "Django Dashboard: FAIL"
fi

if curl -fsS http://localhost:5000/health >/dev/null; then
  echo "MLflow: OK"
  ok=$((ok+1))
else
  echo "MLflow: FAIL"
fi

echo "${ok}/${total} services healthy"
