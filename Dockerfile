FROM python:3.11-slim

# ── System packages ───────────────────────────────────────────────────────────
# libgomp1  : required by LightGBM (OpenMP runtime)
# git       : required by MLflow's git-hash logging (suppressed below if absent)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (HuggingFace Spaces requirement) ────────────────────────────
RUN useradd -m -u 1000 user

# ── Environment ───────────────────────────────────────────────────────────────
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    GIT_PYTHON_REFRESH=quiet \
    # Apache Airflow
    AIRFLOW_HOME=/home/user/airflow \
    AIRFLOW__CORE__DAGS_FOLDER=/app/dags \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__CORE__EXECUTOR=SequentialExecutor \
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////home/user/airflow/airflow.db \
    AIRFLOW__SCHEDULER__DAG_DIR_LIST_INTERVAL=15 \
    AIRFLOW__LOGGING__BASE_LOG_FOLDER=/home/user/airflow/logs \
    AIRFLOW__WEBSERVER__SECRET_KEY=automlops-hf-secret \
    # MLflow — absolute path so Airflow tasks (different CWD) share the same DB
    MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db

USER user
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Install app dependencies first (faster layer caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Apache Airflow with its official constraint file to avoid conflicts
ARG AIRFLOW_VERSION=2.10.4
RUN pip install --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.11.txt"

# ── Application code ──────────────────────────────────────────────────────────
COPY --chown=user . .

# Create directories needed at runtime and ensure start.sh is executable
RUN mkdir -p mlruns logs /home/user/airflow/logs && chmod +x /app/start.sh

# Initialise Airflow metadata DB (SQLite — no external DB needed)
RUN airflow db migrate

EXPOSE 7860

CMD ["/app/start.sh"]
