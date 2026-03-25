"""Background model trainer with MLflow tracking."""
import time
import uuid
import threading
import numpy as np
from datetime import datetime

import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_absolute_error, mean_squared_error,
    confusion_matrix, classification_report,
)

from mlops.datasets import load_dataset
from mlops.algorithms import get_algorithm, ALGORITHMS

# ── Shared job state ──────────────────────────────────────────────────────────
training_jobs: dict = {}
automl_jobs: dict = {}
_lock = threading.Lock()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_or_create_experiment(name: str) -> str:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return mlflow.create_experiment(name)
    return exp.experiment_id


def _update_job(store: dict, job_id: str, **kwargs):
    with _lock:
        store[job_id].update(kwargs)


def _classification_metrics(y_test, y_pred) -> dict:
    return {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_score":  round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
    }


def _regression_metrics(y_test, y_pred) -> dict:
    mse = float(mean_squared_error(y_test, y_pred))
    return {
        "r2_score": round(float(r2_score(y_test, y_pred)), 4),
        "mae":      round(float(mean_absolute_error(y_test, y_pred)), 4),
        "mse":      round(mse, 4),
        "rmse":     round(float(np.sqrt(mse)), 4),
    }


# ── Single training run ───────────────────────────────────────────────────────

def _do_train(job_id: str, dataset_name: str, algorithm_name: str,
              algorithm_category: str, task_type: str, custom_params: dict | None):
    """Executed in a daemon thread; updates training_jobs[job_id] in place."""
    start_time = time.time()
    try:
        _update_job(training_jobs, job_id, status="running", progress=5)
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

        # 1. Load data
        X_train, X_test, y_train, y_test, meta = load_dataset(dataset_name)
        _update_job(training_jobs, job_id, progress=20, dataset_meta=meta)

        # 2. Algorithm config
        algo_cfg = get_algorithm(task_type, algorithm_category, algorithm_name)
        params = {**algo_cfg["params"], **(custom_params or {})}

        # 3. Pre-process
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # Handle NB algorithms that can't take negative inputs
        if "Naive Bayes" in algorithm_name or "Complement" in algorithm_name:
            from sklearn.preprocessing import MinMaxScaler
            mms = MinMaxScaler()
            X_train_s = mms.fit_transform(X_train)
            X_test_s  = mms.transform(X_test)

        _update_job(training_jobs, job_id, progress=35)

        # 4. Train inside an MLflow run
        exp_id = _get_or_create_experiment(dataset_name)
        with mlflow.start_run(experiment_id=exp_id,
                              run_name=f"{algorithm_name} — {dataset_name}") as run:
            run_id = run.info.run_id
            _update_job(training_jobs, job_id, mlflow_run_id=run_id, progress=40)

            mlflow.set_tags({
                "algorithm":  algorithm_name,
                "category":   algorithm_category,
                "dataset":    dataset_name,
                "task_type":  task_type,
                "job_id":     job_id,
            })
            mlflow.log_params({"algorithm": algorithm_name,
                               "category": algorithm_category,
                               "dataset": dataset_name,
                               **{k: str(v) for k, v in params.items()}})

            _update_job(training_jobs, job_id, progress=50)

            model = algo_cfg["class"](**params)
            model.fit(X_train_s, y_train)
            _update_job(training_jobs, job_id, progress=75)

            y_pred = model.predict(X_test_s)

            if task_type == "classification":
                metrics = _classification_metrics(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred).tolist()
                extra = {"confusion_matrix": cm,
                         "report": classification_report(y_test, y_pred, output_dict=True,
                                                          zero_division=0)}
            else:
                metrics = _regression_metrics(y_test, y_pred)
                extra = {"y_test_sample": y_test[:50].tolist(),
                         "y_pred_sample": y_pred[:50].tolist()}

            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            _update_job(training_jobs, job_id, progress=90)

        duration = round(time.time() - start_time, 2)
        _update_job(training_jobs, job_id,
                    status="completed", progress=100,
                    metrics=metrics, extra=extra,
                    duration=duration,
                    completed_at=datetime.utcnow().isoformat())

    except Exception as exc:
        _update_job(training_jobs, job_id,
                    status="failed", error=str(exc), progress=0)


def start_training(dataset_name: str, algorithm_name: str,
                   algorithm_category: str, task_type: str,
                   custom_params: dict | None = None) -> str:
    """Kick off a background training job and return its job_id."""
    job_id = str(uuid.uuid4())[:8]
    with _lock:
        training_jobs[job_id] = {
            "job_id":    job_id,
            "status":    "queued",
            "progress":  0,
            "dataset":   dataset_name,
            "algorithm": algorithm_name,
            "category":  algorithm_category,
            "task_type": task_type,
            "created_at": datetime.utcnow().isoformat(),
        }
    t = threading.Thread(
        target=_do_train,
        args=(job_id, dataset_name, algorithm_name,
              algorithm_category, task_type, custom_params),
        daemon=True,
    )
    t.start()
    return job_id


# ── AutoML: exhaustive search across all algorithms ───────────────────────────

def _do_automl(job_id: str, dataset_name: str, task_type: str,
               optimize_metric: str, max_runs: int):
    """Run every algorithm for the chosen task and log the best."""
    try:
        _update_job(automl_jobs, job_id, status="running", progress=2)
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

        X_train, X_test, y_train, y_test, meta = load_dataset(dataset_name)
        _update_job(automl_jobs, job_id, dataset_meta=meta, progress=5)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        exp_id = _get_or_create_experiment(f"AutoML — {dataset_name}")

        # Collect all algorithms for this task
        all_algos = []
        for cat_name, cat in ALGORITHMS[task_type].items():
            for alg_name, alg_cfg in cat.items():
                all_algos.append((cat_name, alg_name, alg_cfg))

        if max_runs < len(all_algos):
            import random
            random.seed(42)
            all_algos = random.sample(all_algos, max_runs)

        results = []
        total = len(all_algos)

        for idx, (cat_name, alg_name, alg_cfg) in enumerate(all_algos):
            _update_job(automl_jobs, job_id,
                        progress=int(5 + 90 * idx / total),
                        current_algo=alg_name)
            try:
                with mlflow.start_run(experiment_id=exp_id,
                                      run_name=f"AutoML: {alg_name}") as run:
                    mlflow.set_tags({"algorithm": alg_name, "category": cat_name,
                                     "automl_job": job_id, "task_type": task_type})

                    # NB needs non-negative values
                    X_tr = X_train_s
                    X_te = X_test_s
                    if "Naive Bayes" in alg_name or "Complement" in alg_name:
                        from sklearn.preprocessing import MinMaxScaler
                        mms = MinMaxScaler()
                        X_tr = mms.fit_transform(X_train)
                        X_te = mms.transform(X_test)

                    model = alg_cfg["class"](**alg_cfg["params"])
                    t0 = time.time()
                    model.fit(X_tr, y_train)
                    dur = round(time.time() - t0, 2)

                    y_pred = model.predict(X_te)
                    if task_type == "classification":
                        metrics = _classification_metrics(y_test, y_pred)
                    else:
                        metrics = _regression_metrics(y_test, y_pred)

                    mlflow.log_params({"algorithm": alg_name, "category": cat_name})
                    mlflow.log_metrics(metrics)
                    mlflow.sklearn.log_model(model, "model")

                    results.append({
                        "rank":       idx + 1,
                        "algorithm":  alg_name,
                        "category":   cat_name,
                        "metrics":    metrics,
                        "duration":   dur,
                        "run_id":     run.info.run_id,
                        "color":      alg_cfg.get("color", "#8b5cf6"),
                    })
            except Exception:
                pass  # skip failed algorithms silently

        # Sort by optimise metric
        higher_is_better = optimize_metric in ("accuracy", "f1_score", "precision",
                                               "recall", "r2_score")
        results.sort(key=lambda r: r["metrics"].get(optimize_metric, 0),
                     reverse=higher_is_better)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        _update_job(automl_jobs, job_id,
                    status="completed", progress=100,
                    results=results,
                    best=results[0] if results else None,
                    completed_at=datetime.utcnow().isoformat())

    except Exception as exc:
        _update_job(automl_jobs, job_id, status="failed", error=str(exc))


def start_automl(dataset_name: str, task_type: str,
                 optimize_metric: str = "accuracy",
                 max_runs: int = 20) -> str:
    """Kick off an AutoML sweep and return the job_id."""
    job_id = str(uuid.uuid4())[:8]
    with _lock:
        automl_jobs[job_id] = {
            "job_id":     job_id,
            "status":     "queued",
            "progress":   0,
            "dataset":    dataset_name,
            "task_type":  task_type,
            "metric":     optimize_metric,
            "results":    [],
            "created_at": datetime.utcnow().isoformat(),
        }
    t = threading.Thread(
        target=_do_automl,
        args=(job_id, dataset_name, task_type, optimize_metric, max_runs),
        daemon=True,
    )
    t.start()
    return job_id
