"""
Airflow execution bridge for AutoMLOps.

Triggers a real Airflow DAG run, then watches Airflow's metadata DB for
task-state changes and mirrors them into the same ``pipeline_executions``
dict that the existing ``/api/pipeline/status/<exec_id>`` endpoint reads.

The frontend never needs to know Airflow is running — it polls the same
Flask status endpoint it always did.
"""
from __future__ import annotations
import uuid, time, threading, logging
from datetime import datetime

from pipelines.dag_engine import pipeline_executions, _lock

logger = logging.getLogger(__name__)

# Maps Airflow task states → the three states the frontend understands
_AF_STATE: dict[str | None, str] = {
    None:              "pending",
    "queued":          "pending",
    "scheduled":       "pending",
    "deferred":        "pending",
    "running":         "running",
    "success":         "success",
    "skipped":         "success",
    "failed":          "failed",
    "upstream_failed": "failed",
    "removed":         "failed",
}


def _fe_state(af: str | None) -> str:
    return _AF_STATE.get(af, "pending")


# ── watcher thread ────────────────────────────────────────────────────────────

def _watch(exec_id: str, dag_id: str, run_id: str, task_ids: list[str], task_names: dict[str, str]):
    """
    Polls the Airflow metadata DB and pushes updates into pipeline_executions.
    Exits when the DAG run reaches a terminal state (success / failed).
    """
    try:
        from airflow.models import DagRun, TaskInstance
        from airflow.utils.session import create_session
    except ImportError:
        logger.error("Airflow is not installed — watcher thread cannot run")
        with _lock:
            if exec_id in pipeline_executions:
                pipeline_executions[exec_id]["status"] = "failed"
                pipeline_executions[exec_id]["error"]  = "Airflow not installed"
        return

    seen_states: dict[str, str] = {tid: "pending" for tid in task_ids}
    _waiting_logged = False

    for _attempt in range(900):   # max ~15 min of polling
        time.sleep(1.0)
        try:
            with create_session() as session:
                dag_run = session.query(DagRun).filter(
                    DagRun.dag_id == dag_id,
                    DagRun.run_id == run_id,
                ).first()

                if dag_run is None:
                    continue          # scheduler hasn't picked it up yet

                tis = {
                    ti.task_id: ti
                    for ti in session.query(TaskInstance).filter(
                        TaskInstance.dag_id == dag_id,
                        TaskInstance.run_id == run_id,
                    ).all()
                }

            now       = datetime.utcnow().strftime("%H:%M:%S")
            done_cnt  = 0

            with _lock:
                exec_st = pipeline_executions.get(exec_id)
                if exec_st is None:
                    return

                # Show a "waiting" message while the scheduler hasn't started yet
                if not _waiting_logged and _attempt >= 3:
                    all_pending = all(
                        exec_st["task_states"].get(tid, {}).get("status") == "pending"
                        for tid in task_ids
                    )
                    if all_pending:
                        exec_st["logs"].append(f"[{now}] ⏳  Waiting for Airflow scheduler to pick up run…")
                        _waiting_logged = True

                for tid in task_ids:
                    ti    = tis.get(tid)
                    af_st = ti.state if ti else None
                    fe_st = _fe_state(af_st)
                    prev  = seen_states[tid]

                    if fe_st == prev:
                        if fe_st in ("success", "failed"):
                            done_cnt += 1
                        continue

                    seen_states[tid] = fe_st
                    name = task_names.get(tid, tid)

                    if fe_st == "running":
                        exec_st["task_states"][tid]["status"]     = "running"
                        exec_st["task_states"][tid]["started_at"] = (
                            ti.start_date.isoformat() if ti and ti.start_date else None
                        )
                        exec_st["logs"].append(f"[{now}] ▶  {name}")

                    elif fe_st == "success":
                        dur = round(ti.duration, 1) if ti and ti.duration else 0
                        exec_st["task_states"][tid]["status"]      = "success"
                        exec_st["task_states"][tid]["result"]      = f"Completed in {dur}s"
                        exec_st["task_states"][tid]["finished_at"] = (
                            ti.end_date.isoformat() if ti and ti.end_date else None
                        )
                        exec_st["logs"].append(f"[{now}] ✔  {name} — {dur}s")
                        done_cnt += 1

                    elif fe_st == "failed":
                        exec_st["task_states"][tid]["status"]      = "failed"
                        exec_st["task_states"][tid]["error"]       = "Task failed in Airflow"
                        exec_st["task_states"][tid]["finished_at"] = (
                            ti.end_date.isoformat() if ti and ti.end_date else None
                        )
                        exec_st["logs"].append(f"[{now}] ✖  {name} — failed")
                        done_cnt += 1

                total = len(task_ids) or 1
                exec_st["progress"] = int(100 * done_cnt / total)
                exec_st["status"]   = "running"

            # Check terminal state of the whole DAG run
            dag_state = str(dag_run.state) if dag_run else "running"
            if dag_state == "success":
                with _lock:
                    if exec_id in pipeline_executions:
                        pipeline_executions[exec_id]["status"]       = "completed"
                        pipeline_executions[exec_id]["progress"]     = 100
                        pipeline_executions[exec_id]["completed_at"] = datetime.utcnow().isoformat()
                        pipeline_executions[exec_id]["logs"].append(
                            f"[{now}] ✔  DAG '{dag_id}' completed successfully"
                        )
                return

            elif dag_state in ("failed", "upstream_failed"):
                with _lock:
                    if exec_id in pipeline_executions:
                        pipeline_executions[exec_id]["status"] = "failed"
                        pipeline_executions[exec_id]["error"]  = "DAG run failed in Airflow"
                        pipeline_executions[exec_id]["logs"].append(
                            f"[{now}] ✖  DAG '{dag_id}' failed"
                        )
                return

        except Exception as exc:
            logger.warning(f"[watcher] poll error: {exc}")

    # Timed out
    with _lock:
        if exec_id in pipeline_executions:
            pipeline_executions[exec_id]["status"] = "failed"
            pipeline_executions[exec_id]["error"]  = "Execution watcher timed out"


# ── public API ────────────────────────────────────────────────────────────────

def trigger_pipeline(pipeline_id: str, context: dict | None = None, dag=None) -> str:
    """
    Trigger an Airflow DAG run and return an exec_id compatible with the
    existing pipeline_executions / status endpoint contract.

    ``dag`` is the DAG object from pipeline_defs.py (used for task metadata).
    """
    from airflow.api.common.trigger_dag import trigger_dag as af_trigger

    ts      = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    run_id  = f"automlops__{ts}"
    exec_id = str(uuid.uuid4())[:8]

    dag_id = pipeline_id   # our pipeline IDs match Airflow DAG IDs exactly

    # Fire the Airflow DAG run
    af_trigger(dag_id=dag_id, run_id=run_id, conf=context or {}, replace_microseconds=False)

    # Collect task metadata from the pipeline_defs DAG object
    task_ids   = list(dag.tasks.keys())   if dag else []
    task_names = {tid: dag.tasks[tid].name for tid in task_ids} if dag else {}

    # Initialise exec state (same schema as dag_engine.execute_dag)
    task_states = {
        tid: {"status": "pending", "started_at": None,
              "finished_at": None, "result": None, "error": None}
        for tid in task_ids
    }
    now = datetime.utcnow().strftime("%H:%M:%S")
    with _lock:
        pipeline_executions[exec_id] = {
            "exec_id":     exec_id,
            "dag_id":      dag_id,
            "run_id":      run_id,
            "dag_name":    dag.name if dag else dag_id,
            "status":      "queued",
            "progress":    0,
            "task_states": task_states,
            "logs": [f"[{now}] DAG '{dag_id}' triggered in Apache Airflow  (run_id={run_id})"],
            "created_at":  datetime.utcnow().isoformat(),
        }

    # Start the watcher thread
    threading.Thread(
        target=_watch,
        args=(exec_id, dag_id, run_id, task_ids, task_names),
        daemon=True,
    ).start()

    return exec_id


def is_available() -> bool:
    """Return True if Airflow is installed and the scheduler DB is reachable."""
    try:
        from airflow.utils.session import create_session
        with create_session():
            pass
        return True
    except Exception:
        return False
