#!/bin/bash
# AutoMLOps startup — launches Airflow scheduler then the Flask app
set -e

echo "===== AutoMLOps Startup at $(date -u '+%Y-%m-%d %H:%M:%S') ====="

# ── Airflow scheduler ─────────────────────────────────────────────────────────
echo "[startup] Starting Apache Airflow scheduler..."
airflow scheduler &
SCHEDULER_PID=$!
echo "[startup] Scheduler PID: ${SCHEDULER_PID}"

# Brief pause so the scheduler can parse DAGs before first web request
sleep 4

# ── Flask application ─────────────────────────────────────────────────────────
echo "[startup] Starting Flask application on :7860..."
exec gunicorn app:app \
    --bind 0.0.0.0:7860 \
    --workers 1 \
    --threads 4 \
    --worker-class gthread \
    --timeout 300 \
    --log-level info
