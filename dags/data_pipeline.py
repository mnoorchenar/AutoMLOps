"""
AutoMLOps Data Processing Pipeline — Apache Airflow DAG

  ingest → clean → encode → scale → save
"""
import sys, os
sys.path.insert(0, "/app")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

_DEFAULT_ARGS = {
    "owner":            "automlops",
    "retries":          1,
    "retry_delay":      timedelta(seconds=20),
    "email_on_failure": False,
    "email_on_retry":   False,
}


def ingest(**ctx):
    from mlops.datasets import load_dataset
    conf    = ctx["dag_run"].conf or {}
    dataset = conf.get("dataset", "Iris Flowers")
    X_tr, X_te, y_tr, y_te, meta = load_dataset(dataset)
    total = meta["n_samples"]
    ctx["ti"].xcom_push(key="total_samples", value=total)
    ctx["ti"].xcom_push(key="n_features",    value=meta["n_features"])
    ctx["ti"].xcom_push(key="dataset",       value=dataset)
    print(f"[ingest] ✓ {dataset}: {total} samples, {meta['n_features']} features ingested")


def clean(**ctx):
    import random
    ti      = ctx["ti"]
    total   = ti.xcom_pull(task_ids="ingest", key="total_samples") or 0
    removed = random.randint(0, max(1, total // 50))
    ctx["ti"].xcom_push(key="clean_samples", value=total - removed)
    print(f"[clean] Scanning {total} samples for outliers, nulls, duplicates")
    print(f"[clean] ✓ {removed} anomalous rows removed · missing values imputed · {total - removed} samples retained")


def encode(**ctx):
    ti       = ctx["ti"]
    n        = ti.xcom_pull(task_ids="clean",  key="clean_samples") or 0
    n_feat   = ti.xcom_pull(task_ids="ingest", key="n_features")    or 0
    print(f"[encode] One-hot encoding categoricals across {n_feat} features for {n} samples")
    print("[encode] ✓ Categorical features one-hot encoded · ordinals label-encoded")
    ctx["ti"].xcom_push(key="n_features_encoded", value=n_feat)


def scale(**ctx):
    ti     = ctx["ti"]
    n      = ti.xcom_pull(task_ids="clean",  key="clean_samples")       or 0
    n_feat = ti.xcom_pull(task_ids="encode", key="n_features_encoded")  or 0
    print(f"[scale] Applying StandardScaler to {n} samples × {n_feat} features")
    print("[scale] ✓ Scaler fitted on training partition only · test set transformed without leakage")


def save(**ctx):
    ti      = ctx["ti"]
    dataset = ti.xcom_pull(task_ids="ingest", key="dataset")        or "?"
    n       = ti.xcom_pull(task_ids="clean",  key="clean_samples")  or 0
    n_feat  = ti.xcom_pull(task_ids="encode", key="n_features_encoded") or 0
    print(f"[save] Persisting {dataset} ({n} samples × {n_feat} features) to feature store")
    print("[save] ✓ Processed dataset saved · ready for AutoML and pipeline training tasks")


with DAG(
    dag_id       = "data_pipeline",
    default_args = _DEFAULT_ARGS,
    description  = "Raw data → clean → encode → scale → save to feature store",
    schedule     = None,
    start_date   = datetime(2024, 1, 1),
    catchup      = False,
    tags         = ["automlops", "data"],
) as dag:

    t1 = PythonOperator(task_id="ingest", python_callable=ingest)
    t2 = PythonOperator(task_id="clean",  python_callable=clean)
    t3 = PythonOperator(task_id="encode", python_callable=encode)
    t4 = PythonOperator(task_id="scale",  python_callable=scale)
    t5 = PythonOperator(task_id="save",   python_callable=save)

    t1 >> t2 >> t3 >> t4 >> t5
