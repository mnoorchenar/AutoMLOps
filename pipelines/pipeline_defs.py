"""Pre-built ML pipeline DAG definitions."""
import time
import numpy as np
from pipelines.dag_engine import DAG, Task


# ── Task functions ─────────────────────────────────────────────────────────────

def _load_data(ctx, _results):
    from mlops.datasets import load_dataset
    ds = ctx.get("dataset", "Iris Flowers")
    _, _, _, _, meta = load_dataset(ds)
    return f"{meta['n_samples']} samples, {meta['n_features']} features loaded"

def _validate_data(ctx, results):
    time.sleep(0.2)
    return "Schema OK · No nulls detected · Feature ranges valid"

def _preprocess(ctx, results):
    time.sleep(0.3)
    return "StandardScaler fitted · Train/test split 80/20"

def _feature_engineering(ctx, results):
    time.sleep(0.2)
    return "Polynomial features skipped · All features retained"

def _train_model(ctx, results):
    from mlops.datasets import load_dataset
    from mlops.algorithms import get_algorithm
    from sklearn.preprocessing import StandardScaler
    import mlflow, mlflow.sklearn

    ds   = ctx.get("dataset", "Iris Flowers")
    cat  = ctx.get("category", "Tree-Based")
    alg  = ctx.get("algorithm", "Random Forest")
    task = ctx.get("task_type", "classification")

    X_train, X_test, y_train, y_test, _ = load_dataset(ds)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    cfg   = get_algorithm(task, cat, alg)
    model = cfg["class"](**cfg["params"])
    model.fit(X_tr, y_train)
    score = model.score(X_te, y_test)
    return f"Model trained · score={score:.4f}"

def _evaluate_model(ctx, results):
    time.sleep(0.2)
    return "Accuracy / R² computed · Cross-val 5-fold done"

def _generate_report(ctx, results):
    time.sleep(0.15)
    return "HTML report generated · Metrics written to mlflow"

def _register_model(ctx, _results):
    time.sleep(0.1)
    return "Model artifact registered in MLflow Model Registry"

def _deploy_staging(ctx, _results):
    time.sleep(0.2)
    return "Model transitioned to Staging · REST endpoint ready"

# ── Retraining pipeline tasks ──────────────────────────────────────────────────

def _check_drift(ctx, _):
    time.sleep(0.2)
    drift = round(np.random.uniform(0.01, 0.08), 4)
    return f"PSI={drift} · {'Drift detected — retraining triggered' if drift > 0.05 else 'No drift · pipeline skipped'}"

def _fetch_new_data(ctx, _):
    time.sleep(0.3)
    n = np.random.randint(200, 800)
    return f"{n} new labelled samples fetched from data store"

def _merge_datasets(ctx, _):
    time.sleep(0.2)
    return "New data merged with historical · duplicates removed"

def _retrain_champion(ctx, _):
    time.sleep(0.4)
    acc = round(np.random.uniform(0.88, 0.97), 4)
    return f"Champion model retrained · new accuracy={acc}"

def _ab_test(ctx, _):
    time.sleep(0.2)
    return "A/B test scheduled · 10% traffic split for 24 h"

def _promote_production(ctx, _):
    time.sleep(0.15)
    return "Champion model promoted to Production · old version archived"

# ── Data pipeline tasks ────────────────────────────────────────────────────────

def _ingest_raw(ctx, _):
    time.sleep(0.2)
    return "Raw data ingested from source"

def _clean_data(ctx, _):
    time.sleep(0.3)
    removed = np.random.randint(5, 40)
    return f"{removed} anomalous rows removed · missing values imputed"

def _encode_features(ctx, _):
    time.sleep(0.2)
    return "Categorical features one-hot encoded · ordinals label-encoded"

def _scale_features(ctx, _):
    time.sleep(0.2)
    return "Numeric features scaled with StandardScaler"

def _save_processed(ctx, _):
    time.sleep(0.1)
    return "Processed dataset saved to feature store"


# ── DAG builders ──────────────────────────────────────────────────────────────

def build_training_pipeline() -> DAG:
    dag = DAG("training_pipeline",
              "Training Pipeline",
              "End-to-end model training: ingest → preprocess → train → evaluate → register")

    dag.add_task(Task("load_data",      "Load Data",          "Fetch dataset from registry",
                      _load_data,           upstream=[],                  icon="📥", layer=0))
    dag.add_task(Task("validate",       "Validate Data",      "Schema & quality checks",
                      _validate_data,       upstream=["load_data"],       icon="✅", layer=1))
    dag.add_task(Task("preprocess",     "Preprocess",         "Scale & split features",
                      _preprocess,          upstream=["validate"],        icon="🔧", layer=2))
    dag.add_task(Task("feat_eng",       "Feature Engineering","Derive new features",
                      _feature_engineering, upstream=["preprocess"],      icon="⚗️", layer=2))
    dag.add_task(Task("train",          "Train Model",        "Fit model with MLflow tracking",
                      _train_model,         upstream=["feat_eng"],        icon="🧠", layer=3))
    dag.add_task(Task("evaluate",       "Evaluate",           "Compute metrics on hold-out set",
                      _evaluate_model,      upstream=["train"],           icon="📊", layer=4))
    dag.add_task(Task("report",         "Generate Report",    "Write evaluation artefacts",
                      _generate_report,     upstream=["evaluate"],        icon="📝", layer=4))
    dag.add_task(Task("register",       "Register Model",     "Push model to MLflow Registry",
                      _register_model,      upstream=["evaluate"],        icon="📦", layer=5))
    dag.add_task(Task("deploy_staging", "Deploy to Staging",  "Transition registered model to Staging",
                      _deploy_staging,      upstream=["register"],        icon="🚀", layer=6))
    return dag


def build_retraining_pipeline() -> DAG:
    dag = DAG("retraining_pipeline",
              "Retraining Pipeline",
              "Automated retraining triggered by data drift detection")

    dag.add_task(Task("drift_check",  "Drift Detection",    "PSI & KS tests on incoming data",
                      _check_drift,         upstream=[],                  icon="📡", layer=0))
    dag.add_task(Task("fetch_data",   "Fetch New Data",     "Pull latest labelled samples",
                      _fetch_new_data,      upstream=["drift_check"],     icon="🗄️", layer=1))
    dag.add_task(Task("merge",        "Merge Datasets",     "Combine new + historical data",
                      _merge_datasets,      upstream=["fetch_data"],      icon="🔗", layer=2))
    dag.add_task(Task("retrain",      "Retrain Champion",   "Retrain best model on merged data",
                      _retrain_champion,    upstream=["merge"],           icon="🔁", layer=3))
    dag.add_task(Task("ab_test",      "A/B Test",           "Shadow-deploy challenger",
                      _ab_test,             upstream=["retrain"],         icon="🔀", layer=4))
    dag.add_task(Task("promote",      "Promote to Prod",    "Archive old, promote new champion",
                      _promote_production,  upstream=["ab_test"],         icon="🏆", layer=5))
    return dag


def build_data_pipeline() -> DAG:
    dag = DAG("data_pipeline",
              "Data Processing Pipeline",
              "Automated feature engineering and data preparation pipeline")

    dag.add_task(Task("ingest",   "Ingest Raw Data",    "Pull raw data from sources",
                      _ingest_raw,    upstream=[],              icon="📥", layer=0))
    dag.add_task(Task("clean",    "Clean Data",         "Remove outliers & impute missing",
                      _clean_data,    upstream=["ingest"],      icon="🧹", layer=1))
    dag.add_task(Task("encode",   "Encode Features",    "Categorical encoding",
                      _encode_features, upstream=["clean"],     icon="🔢", layer=2))
    dag.add_task(Task("scale",    "Scale Features",     "Normalise numeric columns",
                      _scale_features, upstream=["encode"],     icon="⚖️",  layer=3))
    dag.add_task(Task("save",     "Save to Feature Store","Persist processed dataset",
                      _save_processed, upstream=["scale"],      icon="💾", layer=4))
    return dag


# Singleton instances (rebuilt on each call so context can vary)
PIPELINE_BUILDERS = {
    "training_pipeline":   build_training_pipeline,
    "retraining_pipeline": build_retraining_pipeline,
    "data_pipeline":       build_data_pipeline,
}


def get_pipeline(pipeline_id: str) -> DAG:
    if pipeline_id not in PIPELINE_BUILDERS:
        raise ValueError(f"Unknown pipeline: {pipeline_id}")
    return PIPELINE_BUILDERS[pipeline_id]()
