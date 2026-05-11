#!/bin/bash
set -euo pipefail

# Load .env if present
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

POSTGRES_USER=${POSTGRES_USER:-sepsis_user}
POSTGRES_DB=${POSTGRES_DB:-sepsis_db}
POSTGRES_SERVICE=${POSTGRES_SERVICE:-postgres}

attempts=0
max_attempts=30

if ! docker compose ps -q "$POSTGRES_SERVICE" >/dev/null 2>&1; then
  echo "docker compose not available or not in PATH" >&2
  exit 1
fi

if [ -z "$(docker compose ps -q "$POSTGRES_SERVICE")" ]; then
  docker compose up -d "$POSTGRES_SERVICE"
fi

until docker compose exec -T "$POSTGRES_SERVICE" pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; do
  attempts=$((attempts+1))
  if [ "$attempts" -ge "$max_attempts" ]; then
    echo "Postgres not ready after ${max_attempts} attempts" >&2
    exit 1
  fi
  sleep 1
done

docker compose exec -T "$POSTGRES_SERVICE" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 < docs/database_schema.sql

echo "Database setup complete!"
