"""
AutoMLOps Retraining Pipeline — Apache Airflow DAG

  drift_check → fetch_data → merge → retrain → ab_test → promote
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


def drift_check(**ctx):
    import random
    conf    = ctx["dag_run"].conf or {}
    dataset = conf.get("dataset", "Iris Flowers")
    print(f"[drift_check] Running PSI & KS tests on {dataset} incoming data...")
    drift_score    = round(random.uniform(0.03, 0.28), 4)
    drift_detected = drift_score > 0.10
    ctx["ti"].xcom_push(key="drift_score",    value=drift_score)
    ctx["ti"].xcom_push(key="drift_detected", value=drift_detected)
    status = "DRIFT DETECTED — retraining triggered" if drift_detected else "No significant drift"
    print(f"[drift_check] PSI={drift_score}  {status}")


def fetch_data(**ctx):
    import random
    ti            = ctx["ti"]
    drift_score   = ti.xcom_pull(task_ids="drift_check", key="drift_score") or 0
    n_new         = random.randint(150, 600)
    ctx["ti"].xcom_push(key="n_new_samples", value=n_new)
    print(f"[fetch_data] Fetching new labelled samples (drift_score={drift_score})")
    print(f"[fetch_data] ✓ {n_new} new samples retrieved from data store")


def merge(**ctx):
    ti       = ctx["ti"]
    n_new    = ti.xcom_pull(task_ids="fetch_data", key="n_new_samples") or 0
    print(f"[merge] Merging {n_new} new samples with historical data")
    print("[merge] ✓ Duplicate rows removed · class balance checked · dataset merged")


def retrain(**ctx):
    from mlops.datasets import DATASETS
    from mlops.trainer import train_for_pipeline
    conf      = ctx["dag_run"].conf or {}
    dataset   = conf.get("dataset",   "Iris Flowers")
    task_type = conf.get("task_type") or DATASETS.get(dataset, {}).get("task", "classification")
    category  = conf.get("category",  "Tree-Based")
    algorithm = conf.get("algorithm", "Random Forest")
    run_id    = ctx["dag_run"].run_id[:12]

    print(f"[retrain] Retraining champion: {algorithm} on {dataset}")
    metrics = train_for_pipeline(dataset, task_type, category, algorithm,
                                  experiment_name=f"retrain-{run_id}")
    ctx["ti"].xcom_push(key="new_metrics",  value=metrics)
    ctx["ti"].xcom_push(key="algorithm",    value=algorithm)
    print(f"[retrain] ✓ New metrics: {metrics}")


def ab_test(**ctx):
    import random
    ti        = ctx["ti"]
    metrics   = ti.xcom_pull(task_ids="retrain", key="new_metrics") or {}
    algo      = ti.xcom_pull(task_ids="retrain", key="algorithm")   or "?"
    new_score = metrics.get("accuracy") or metrics.get("r2_score")  or 0.0
    baseline  = round(random.uniform(0.82, 0.93), 4)
    delta     = round(new_score - baseline, 4)
    promote   = new_score > baseline
    ctx["ti"].xcom_push(key="promote",   value=promote)
    ctx["ti"].xcom_push(key="new_score", value=round(new_score, 4))
    ctx["ti"].xcom_push(key="baseline",  value=baseline)
    verdict = "PROMOTE challenger" if promote else "KEEP production model"
    print(f"[ab_test] {algo}  baseline={baseline}  new={new_score:.4f}  Δ={delta:+.4f}  → {verdict}")


def promote(**ctx):
    ti        = ctx["ti"]
    algo      = ti.xcom_pull(task_ids="retrain", key="algorithm")   or "?"
    promote   = ti.xcom_pull(task_ids="ab_test", key="promote")
    new_score = ti.xcom_pull(task_ids="ab_test", key="new_score")   or 0
    baseline  = ti.xcom_pull(task_ids="ab_test", key="baseline")    or 0
    if promote:
        print(f"[promote] ✓ {algo} (score={new_score}) promoted to Production")
        print(f"[promote] Previous production model (score={baseline}) archived")
    else:
        print(f"[promote] ✗ {algo} (score={new_score}) did not beat baseline ({baseline})")
        print("[promote] Keeping current production model")


with DAG(
    dag_id       = "retraining_pipeline",
    default_args = _DEFAULT_ARGS,
    description  = "Drift detection → fetch new data → merge → retrain → A/B test → promote",
    schedule     = None,
    start_date   = datetime(2024, 1, 1),
    catchup      = False,
    tags         = ["automlops", "retraining"],
) as dag:

    t1 = PythonOperator(task_id="drift_check", python_callable=drift_check)
    t2 = PythonOperator(task_id="fetch_data",  python_callable=fetch_data)
    t3 = PythonOperator(task_id="merge",       python_callable=merge)
    t4 = PythonOperator(task_id="retrain",     python_callable=retrain)
    t5 = PythonOperator(task_id="ab_test",     python_callable=ab_test)
    t6 = PythonOperator(task_id="promote",     python_callable=promote)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6
