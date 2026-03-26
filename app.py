"""AutoMLOps — ML Experiment Tracking & Pipeline Orchestration Platform."""
import os
import json
import threading
from datetime import datetime

import mlflow
import mlflow.sklearn
from flask import Flask, render_template, request, jsonify

from mlops.datasets import DATASETS, load_dataset
from mlops.algorithms import ALGORITHMS, list_algorithms, all_algorithm_names, algorithms_for_json
from mlops.trainer import (
    training_jobs, automl_jobs,
    start_training, start_automl,
)
from pipelines.dag_engine import pipeline_executions, execute_dag
from pipelines.pipeline_defs import get_pipeline, PIPELINE_BUILDERS

app = Flask(__name__)

# ── MLflow setup ───────────────────────────────────────────────────────────────
TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(TRACKING_URI)


def _mlflow_client():
    return mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)


# ── Seed demo data on first launch ────────────────────────────────────────────

def _seed_demo():
    """Pre-populate a few MLflow runs so the dashboard looks great immediately."""
    client = _mlflow_client()
    try:
        existing = client.search_runs(experiment_ids=[], max_results=1)
        if existing:
            return  # already seeded
    except Exception:
        pass

    demo_runs = [
        ("Iris Flowers",     "Ensemble / Boosting", "Random Forest",       "classification",
         {"accuracy": 0.9667, "f1_score": 0.9664, "precision": 0.9672, "recall": 0.9667}),
        ("Iris Flowers",     "Ensemble / Boosting", "XGBoost",             "classification",
         {"accuracy": 0.9600, "f1_score": 0.9598, "precision": 0.9601, "recall": 0.9600}),
        ("Iris Flowers",     "Linear Models",       "Logistic Regression", "classification",
         {"accuracy": 0.9467, "f1_score": 0.9463, "precision": 0.9472, "recall": 0.9467}),
        ("Wine Quality",     "Ensemble / Boosting", "LightGBM",            "classification",
         {"accuracy": 0.9722, "f1_score": 0.9720, "precision": 0.9725, "recall": 0.9722}),
        ("Wine Quality",     "Neural Networks",     "MLP (Medium)",        "classification",
         {"accuracy": 0.9444, "f1_score": 0.9441, "precision": 0.9449, "recall": 0.9444}),
        ("Breast Cancer",    "Support Vector Machines", "SVC (RBF Kernel)","classification",
         {"accuracy": 0.9737, "f1_score": 0.9736, "precision": 0.9741, "recall": 0.9737}),
        ("Breast Cancer",    "Ensemble / Boosting", "Gradient Boosting",   "classification",
         {"accuracy": 0.9561, "f1_score": 0.9558, "precision": 0.9565, "recall": 0.9561}),
        ("Diabetes Progression", "Ensemble / Boosting", "XGBoost Regressor","regression",
         {"r2_score": 0.4823, "mae": 44.12, "mse": 3124.5, "rmse": 55.90}),
        ("Diabetes Progression", "Linear Models",   "Ridge Regression",    "regression",
         {"r2_score": 0.4612, "mae": 45.87, "mse": 3258.3, "rmse": 57.08}),
        ("California Housing","Ensemble / Boosting","LightGBM Regressor",  "regression",
         {"r2_score": 0.8341, "mae": 0.3124, "mse": 0.2871, "rmse": 0.5358}),
    ]

    for ds, cat, alg, task, metrics in demo_runs:
        try:
            exp = client.get_experiment_by_name(ds)
            exp_id = exp.experiment_id if exp else mlflow.create_experiment(ds)
            with mlflow.start_run(experiment_id=exp_id,
                                  run_name=f"{alg} — {ds}") as run:
                mlflow.set_tags({"algorithm": alg, "category": cat,
                                 "dataset": ds, "task_type": task, "demo": "true"})
                mlflow.log_params({"algorithm": alg, "category": cat, "dataset": ds})
                mlflow.log_metrics(metrics)
        except Exception:
            pass


# Seed in background so startup isn't delayed
threading.Thread(target=_seed_demo, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def dashboard():
    client = _mlflow_client()
    try:
        runs = client.search_runs(
            experiment_ids=[],
            max_results=200,
            order_by=["start_time DESC"],
        )
    except Exception:
        runs = []

    # Stats
    total_runs   = len(runs)
    completed    = [r for r in runs if r.info.status == "FINISHED"]
    best_acc     = 0.0
    for r in completed:
        acc = r.data.metrics.get("accuracy") or r.data.metrics.get("r2_score") or 0
        if acc > best_acc:
            best_acc = acc

    # Recent runs (last 8)
    recent = []
    for r in runs[:8]:
        metrics = r.data.metrics
        primary = (metrics.get("accuracy") or metrics.get("r2_score") or 0)
        recent.append({
            "run_id":    r.info.run_id[:8],
            "algorithm": r.data.tags.get("algorithm", "—"),
            "dataset":   r.data.tags.get("dataset",   "—"),
            "category":  r.data.tags.get("category",  "—"),
            "task_type": r.data.tags.get("task_type", "classification"),
            "primary_metric": round(primary, 4),
            "status":    r.info.status,
            "duration":  round((r.info.end_time - r.info.start_time) / 1000, 1)
                         if r.info.end_time else "—",
        })

    # Algorithm distribution
    algo_counts: dict = {}
    for r in completed:
        cat = r.data.tags.get("category", "Other")
        algo_counts[cat] = algo_counts.get(cat, 0) + 1

    # Dataset distribution
    ds_counts: dict = {}
    for r in completed:
        ds = r.data.tags.get("dataset", "Other")
        ds_counts[ds] = ds_counts.get(ds, 0) + 1

    return render_template("dashboard.html",
                           total_runs=total_runs,
                           completed_runs=len(completed),
                           best_metric=round(best_acc, 4),
                           n_experiments=len(set(r.info.experiment_id for r in runs)),
                           recent_runs=recent,
                           algo_counts=json.dumps(algo_counts),
                           ds_counts=json.dumps(ds_counts),
                           datasets=DATASETS,
                           algorithms=algorithms_for_json())


@app.route("/experiments")
def experiments():
    client = _mlflow_client()
    try:
        exps = client.search_experiments()
    except Exception:
        exps = []
    exp_list = []
    for e in exps:
        runs = client.search_runs([e.experiment_id], max_results=100,
                                  order_by=["start_time DESC"])
        finished = [r for r in runs if r.info.status == "FINISHED"]
        best = 0.0
        for r in finished:
            v = r.data.metrics.get("accuracy") or r.data.metrics.get("r2_score") or 0
            if v > best:
                best = v
        exp_list.append({
            "experiment_id": e.experiment_id,
            "name":          e.name,
            "run_count":     len(runs),
            "best_metric":   round(best, 4),
            "created_at":    datetime.fromtimestamp(e.creation_time / 1000).strftime("%Y-%m-%d %H:%M")
                             if e.creation_time else "—",
        })
    return render_template("experiments.html", experiments=exp_list)


@app.route("/experiments/<experiment_id>")
def experiment_runs(experiment_id):
    client = _mlflow_client()
    exp    = client.get_experiment(experiment_id)
    runs   = client.search_runs([experiment_id], max_results=200,
                                order_by=["start_time DESC"])
    run_list = []
    for r in runs:
        m = r.data.metrics
        run_list.append({
            "run_id":    r.info.run_id,
            "run_id_short": r.info.run_id[:8],
            "algorithm": r.data.tags.get("algorithm", "—"),
            "category":  r.data.tags.get("category",  "—"),
            "dataset":   r.data.tags.get("dataset",   "—"),
            "task_type": r.data.tags.get("task_type", "classification"),
            "metrics":   {k: round(v, 4) for k, v in m.items()},
            "params":    r.data.params,
            "status":    r.info.status,
            "duration":  round((r.info.end_time - r.info.start_time) / 1000, 1)
                         if r.info.end_time else None,
            "start_time": datetime.fromtimestamp(r.info.start_time / 1000)
                          .strftime("%Y-%m-%d %H:%M:%S")
                          if r.info.start_time else "—",
        })
    return render_template("runs.html",
                           experiment={"name": exp.name, "id": experiment_id},
                           runs=run_list)


@app.route("/pipeline")
def pipeline():
    dags = {}
    for pid, builder in PIPELINE_BUILDERS.items():
        dags[pid] = builder().to_dict()
    # Strip non-serializable "loader" function before passing to template
    datasets_safe = {name: {k: v for k, v in cfg.items() if k != "loader"}
                     for name, cfg in DATASETS.items()}
    return render_template("pipeline.html", dags=json.dumps(dags),
                           datasets=datasets_safe)


@app.route("/models")
def models():
    client = _mlflow_client()
    try:
        registered = client.search_registered_models()
    except Exception:
        registered = []
    model_list = []
    for m in registered:
        versions = client.get_latest_versions(m.name)
        ver_list = []
        for v in versions:
            run = None
            metrics = {}
            try:
                run  = client.get_run(v.run_id)
                metrics = {k: round(val, 4) for k, val in run.data.metrics.items()}
            except Exception:
                pass
            ver_list.append({
                "version":    v.version,
                "stage":      v.current_stage,
                "run_id":     v.run_id[:8] if v.run_id else "—",
                "metrics":    metrics,
                "created_at": datetime.fromtimestamp(v.creation_timestamp / 1000)
                              .strftime("%Y-%m-%d %H:%M")
                              if v.creation_timestamp else "—",
            })
        model_list.append({
            "name":        m.name,
            "description": m.description or "—",
            "versions":    ver_list,
            "latest_stage": ver_list[0]["stage"] if ver_list else "None",
        })
    return render_template("models.html", models=model_list)


@app.route("/automl")
def automl():
    return render_template("automl.html",
                           datasets=DATASETS,
                           algorithms=algorithms_for_json())


# ══════════════════════════════════════════════════════════════════════════════
#  API — TRAINING
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/train", methods=["POST"])
def api_train():
    data = request.get_json(force=True)
    required = ["dataset", "algorithm", "category", "task_type"]
    if not all(k in data for k in required):
        return jsonify({"error": f"Missing fields: {required}"}), 400
    job_id = start_training(
        dataset_name=data["dataset"],
        algorithm_name=data["algorithm"],
        algorithm_category=data["category"],
        task_type=data["task_type"],
        custom_params=data.get("params"),
    )
    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/api/run/<job_id>/status")
def api_run_status(job_id):
    job = training_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/runs")
def api_runs():
    client = _mlflow_client()
    exp_filter = request.args.get("experiment")
    task_filter = request.args.get("task")
    try:
        exp_ids = []
        if exp_filter:
            exp = client.get_experiment_by_name(exp_filter)
            if exp:
                exp_ids = [exp.experiment_id]
        runs = client.search_runs(
            experiment_ids=exp_ids or [],
            max_results=200,
            order_by=["start_time DESC"],
        )
    except Exception:
        runs = []
    result = []
    for r in runs:
        if task_filter and r.data.tags.get("task_type") != task_filter:
            continue
        m = r.data.metrics
        result.append({
            "run_id":    r.info.run_id,
            "algorithm": r.data.tags.get("algorithm", "—"),
            "category":  r.data.tags.get("category",  "—"),
            "dataset":   r.data.tags.get("dataset",   "—"),
            "task_type": r.data.tags.get("task_type", "classification"),
            "metrics":   {k: round(v, 4) for k, v in m.items()},
            "status":    r.info.status,
            "start_time": r.info.start_time,
        })
    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════════════
#  API — PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/pipeline/<pipeline_id>/execute", methods=["POST"])
def api_pipeline_execute(pipeline_id):
    context = request.get_json(force=True) or {}
    try:
        dag = get_pipeline(pipeline_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Try Apache Airflow first; fall back to the built-in DAG engine if
    # Airflow is not installed or not yet ready (e.g. first startup).
    try:
        from mlops.airflow_runner import trigger_pipeline, is_available
        if is_available():
            exec_id = trigger_pipeline(pipeline_id, context=context, dag=dag)
            return jsonify({"exec_id": exec_id, "status": "queued", "engine": "airflow"})
    except Exception as af_err:
        app.logger.warning(f"Airflow trigger failed ({af_err}), falling back to built-in engine")

    exec_id = execute_dag(dag, context)
    return jsonify({"exec_id": exec_id, "status": "queued", "engine": "builtin"})


@app.route("/api/pipeline/status/<exec_id>")
def api_pipeline_status(exec_id):
    state = pipeline_executions.get(exec_id)
    if not state:
        return jsonify({"error": "Execution not found"}), 404
    return jsonify(state)


@app.route("/api/pipeline/<pipeline_id>/dag")
def api_pipeline_dag(pipeline_id):
    try:
        dag = get_pipeline(pipeline_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(dag.to_dict())


# ══════════════════════════════════════════════════════════════════════════════
#  API — MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/models/register", methods=["POST"])
def api_models_register():
    data   = request.get_json(force=True)
    run_id = data.get("run_id")
    name   = data.get("name")
    if not run_id or not name:
        return jsonify({"error": "run_id and name required"}), 400
    try:
        client = _mlflow_client()
        run    = client.get_run(run_id)
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, name)
        return jsonify({"name": result.name, "version": result.version,
                        "status": "registered"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/models/<name>/<version>/stage", methods=["POST"])
def api_model_stage(name, version):
    data  = request.get_json(force=True)
    stage = data.get("stage", "Staging")
    valid = {"Staging", "Production", "Archived", "None"}
    if stage not in valid:
        return jsonify({"error": f"stage must be one of {valid}"}), 400
    try:
        client = _mlflow_client()
        client.transition_model_version_stage(name=name, version=version,
                                              stage=stage, archive_existing_versions=False)
        return jsonify({"name": name, "version": version, "stage": stage})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ══════════════════════════════════════════════════════════════════════════════
#  API — AUTO-ML
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/automl", methods=["POST"])
def api_automl():
    data = request.get_json(force=True)
    if "dataset" not in data or "task_type" not in data:
        return jsonify({"error": "dataset and task_type required"}), 400
    job_id = start_automl(
        dataset_name=data["dataset"],
        task_type=data["task_type"],
        optimize_metric=data.get("metric", "accuracy"),
        max_runs=int(data.get("max_runs", 20)),
    )
    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/api/automl/status/<job_id>")
def api_automl_status(job_id):
    job = automl_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


# ══════════════════════════════════════════════════════════════════════════════
#  API — META
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/algorithms")
def api_algorithms():
    task = request.args.get("task", "classification")
    try:
        return jsonify(algorithms_for_json(task))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/datasets")
def api_datasets():
    result = {
        name: {k: v for k, v in cfg.items() if k != "loader"}
        for name, cfg in DATASETS.items()
    }
    return jsonify(result)


@app.route("/api/stats")
def api_stats():
    client = _mlflow_client()
    try:
        runs = client.search_runs(
            experiment_ids=[], max_results=500,
            order_by=["start_time DESC"],
        )
    except Exception:
        runs = []
    finished = [r for r in runs if r.info.status == "FINISHED"]
    best = 0.0
    for r in finished:
        v = r.data.metrics.get("accuracy") or r.data.metrics.get("r2_score") or 0
        if v > best:
            best = v

    recent = []
    for r in runs[:8]:
        m = r.data.metrics
        primary = m.get("accuracy") or m.get("r2_score") or 0
        recent.append({
            "run_id":         r.info.run_id[:8],
            "algorithm":      r.data.tags.get("algorithm", "—"),
            "category":       r.data.tags.get("category",  "—"),
            "dataset":        r.data.tags.get("dataset",   "—"),
            "primary_metric": round(primary, 4),
            "status":         r.info.status,
            "duration":       round((r.info.end_time - r.info.start_time) / 1000, 1)
                              if r.info.end_time else None,
        })

    algo_counts: dict = {}
    ds_counts:   dict = {}
    for r in finished:
        cat = r.data.tags.get("category", "Other")
        algo_counts[cat] = algo_counts.get(cat, 0) + 1
        ds = r.data.tags.get("dataset", "Other")
        ds_counts[ds] = ds_counts.get(ds, 0) + 1

    return jsonify({
        "total_runs":     len(runs),
        "completed_runs": len(finished),
        "best_metric":    round(best, 4),
        "n_experiments":  len(set(r.info.experiment_id for r in runs)),
        "recent_runs":    recent,
        "algo_counts":    algo_counts,
        "ds_counts":      ds_counts,
    })


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
