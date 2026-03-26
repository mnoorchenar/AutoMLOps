#!/bin/bash
# AutoMLOps startup — launches Airflow scheduler then the Flask app
set -e

echo "===== AutoMLOps Startup at $(date -u '+%Y-%m-%d %H:%M:%S') ====="

# ── Airflow scheduler ─────────────────────────────────────────────────────────
echo "[startup] Starting Apache Airflow scheduler..."
airflow scheduler &
SCHEDULER_PID=$!
echo "[startup] Scheduler PID: ${SCHEDULER_PID}"

# Wait until the Airflow DB is reachable and scheduler has parsed DAGs.
# Polls every 2 s, gives up after 90 s (HuggingFace Space health-check window).
echo "[startup] Waiting for Airflow scheduler to become ready..."
for i in $(seq 1 45); do
    if python - <<'EOF' 2>/dev/null
import sys
try:
    from airflow.utils.session import create_session
    from airflow.models import DagBag
    with create_session() as s:
        pass
    db = DagBag(dag_folder="/app/dags", include_examples=False, read_dags_from_db=False)
    if db.import_errors:
        sys.exit(1)
    sys.exit(0)
except Exception:
    sys.exit(1)
EOF
    then
        echo "[startup] Airflow ready after $((i*2))s"
        break
    fi
    sleep 2
done

# ── Flask application ─────────────────────────────────────────────────────────
echo "[startup] Starting Flask application on :7860..."
exec gunicorn app:app \
    --bind 0.0.0.0:7860 \
    --workers 1 \
    --threads 4 \
    --worker-class gthread \
    --timeout 300 \
    --log-level info
