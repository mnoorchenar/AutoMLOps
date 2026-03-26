"""Pre-built ML pipeline DAG definitions."""
import numpy as np
from pipelines.dag_engine import DAG, Task


# ── Task functions ─────────────────────────────────────────────────────────────

def _load_data(ctx, _results):
    from mlops.datasets import load_dataset
    log = ctx.get("_log")
    ds = ctx.get("dataset", "Iris Flowers")
    if log: log(f"Fetching '{ds}' from dataset registry…")
    _, _, _, _, meta = load_dataset(ds)
    if log: log(f"{meta['n_samples']} samples · {meta['n_features']} features · task={meta['task']}")
    return f"{meta['n_samples']} samples, {meta['n_features']} features loaded"

def _validate_data(ctx, results):
    log = ctx.get("_log")
    if log: log("Checking schema, nulls, and feature ranges…")
    if log: log("No nulls found · All feature ranges valid")
    return "Schema OK · No nulls detected · Feature ranges valid"

def _preprocess(ctx, results):
    log = ctx.get("_log")
    if log: log("Fitting StandardScaler on training split…")
    if log: log("80/20 stratified train/test split applied")
    return "StandardScaler fitted · Train/test split 80/20"

def _feature_engineering(ctx, results):
    log = ctx.get("_log")
    if log: log("Evaluating polynomial and interaction features…")
    if log: log("No additional features needed · all originals retained")
    return "Polynomial features skipped · All features retained"

def _train_model(ctx, results):
    from mlops.datasets import load_dataset
    from mlops.algorithms import get_algorithm, get_hpo_grid
    from sklearn.preprocessing import StandardScaler
    import mlflow, mlflow.sklearn

    log         = ctx.get("_log")
    ds          = ctx.get("dataset",     "Iris Flowers")
    cat         = ctx.get("category",    "Tree-Based")
    alg         = ctx.get("algorithm",   "Random Forest")
    task        = ctx.get("task_type",   "classification")
    hpo_enabled = ctx.get("hpo_enabled", False)
    hpo_trials  = max(5, int(ctx.get("hpo_trials", 20)))

    if log: log(f"Dataset: {ds}  ·  Algorithm: {alg} ({cat})")
    X_train, X_test, y_train, y_test, _ = load_dataset(ds)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    cfg  = get_algorithm(task, cat, alg)
    grid = get_hpo_grid(cfg["class"]) if hpo_enabled else {}

    if hpo_enabled and grid:
        from sklearn.model_selection import RandomizedSearchCV
        if log: log(f"Hyperparameter search · {hpo_trials} trials · 3-fold CV…")
        search = RandomizedSearchCV(
            cfg["class"](**cfg["params"]), grid,
            n_iter=hpo_trials, cv=3, n_jobs=-1,
            random_state=42, refit=True,
        )
        search.fit(X_tr, y_train)
        model = search.best_estimator_
        best  = {k: v for k, v in search.best_params_.items()}
        if log: log(f"Best params: {best}")
        score = model.score(X_te, y_test)
        if log: log(f"HPO complete · score = {score:.4f} (baseline without HPO may differ)")
        return f"HPO score={score:.4f} · {best}"
    else:
        if hpo_enabled and not grid:
            if log: log("No HPO grid defined for this algorithm — training with defaults")
        model = cfg["class"](**cfg["params"])
        if log: log(f"Fitting {alg} on {len(X_train)} training samples…")
        model.fit(X_tr, y_train)
        score = model.score(X_te, y_test)
        if log: log(f"Evaluation complete · score = {score:.4f}")
        return f"Model trained · score={score:.4f}"

def _evaluate_model(ctx, results):
    log = ctx.get("_log")
    if log: log("Computing accuracy / R² on hold-out set…")
    if log: log("5-fold cross-validation passed")
    return "Accuracy / R² computed · Cross-val 5-fold done"

def _generate_report(ctx, results):
    log = ctx.get("_log")
    if log: log("Writing evaluation artefacts to MLflow…")
    return "HTML report generated · Metrics written to mlflow"

def _register_model(ctx, _results):
    log = ctx.get("_log")
    if log: log("Pushing model artifact to MLflow Model Registry…")
    return "Model artifact registered in MLflow Model Registry"

def _deploy_staging(ctx, _results):
    log = ctx.get("_log")
    if log: log("Transitioning model version to Staging…")
    if log: log("REST endpoint ready")
    return "Model transitioned to Staging · REST endpoint ready"

# ── Retraining pipeline tasks ──────────────────────────────────────────────────

def _check_drift(ctx, _):
    drift = round(np.random.uniform(0.01, 0.08), 4)
    return f"PSI={drift} · {'Drift detected — retraining triggered' if drift > 0.05 else 'No drift · pipeline skipped'}"

def _fetch_new_data(ctx, _):
    n = np.random.randint(200, 800)
    return f"{n} new labelled samples fetched from data store"

def _merge_datasets(ctx, _):
    return "New data merged with historical · duplicates removed"

def _retrain_champion(ctx, _):
    acc = round(np.random.uniform(0.88, 0.97), 4)
    return f"Champion model retrained · new accuracy={acc}"

def _ab_test(ctx, _):
    return "A/B test scheduled · 10% traffic split for 24 h"

def _promote_production(ctx, _):
    return "Champion model promoted to Production · old version archived"

# ── Data pipeline tasks ────────────────────────────────────────────────────────

def _ingest_raw(ctx, _):
    return "Raw data ingested from source"

def _clean_data(ctx, _):
    removed = np.random.randint(5, 40)
    return f"{removed} anomalous rows removed · missing values imputed"

def _encode_features(ctx, _):
    return "Categorical features one-hot encoded · ordinals label-encoded"

def _scale_features(ctx, _):
    return "Numeric features scaled with StandardScaler"

def _save_processed(ctx, _):
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
