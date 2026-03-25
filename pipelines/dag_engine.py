"""Lightweight DAG execution engine — inspired by Apache Airflow concepts."""
from __future__ import annotations
import time
import uuid
import threading
from datetime import datetime
from typing import Callable

# ── Shared execution state ─────────────────────────────────────────────────────
pipeline_executions: dict = {}
_lock = threading.Lock()


class Task:
    """A single unit of work in a DAG."""

    def __init__(self, task_id: str, name: str, description: str,
                 func: Callable, upstream: list[str] | None = None,
                 icon: str = "⚙️", layer: int = 0):
        self.task_id    = task_id
        self.name       = name
        self.description = description
        self.func       = func
        self.upstream   = upstream or []   # list of task_ids this depends on
        self.icon       = icon
        self.layer      = layer            # visual column in the DAG


class DAG:
    """A directed acyclic graph of Tasks."""

    def __init__(self, dag_id: str, name: str, description: str):
        self.dag_id      = dag_id
        self.name        = name
        self.description = description
        self.tasks: dict[str, Task] = {}

    def add_task(self, task: Task):
        self.tasks[task.task_id] = task

    def topological_order(self) -> list[str]:
        """Kahn's algorithm — returns task_ids in execution order."""
        in_degree = {tid: 0 for tid in self.tasks}
        for task in self.tasks.values():
            for up in task.upstream:
                in_degree[task.task_id] += 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            # Sort for determinism
            queue.sort(key=lambda t: (self.tasks[t].layer, t))
            tid = queue.pop(0)
            order.append(tid)
            for task in self.tasks.values():
                if tid in task.upstream:
                    in_degree[task.task_id] -= 1
                    if in_degree[task.task_id] == 0:
                        queue.append(task.task_id)

        return order

    def to_dict(self) -> dict:
        """Serialise DAG structure for the frontend."""
        return {
            "dag_id":      self.dag_id,
            "name":        self.name,
            "description": self.description,
            "tasks": {
                tid: {
                    "task_id":     t.task_id,
                    "name":        t.name,
                    "description": t.description,
                    "upstream":    t.upstream,
                    "icon":        t.icon,
                    "layer":       t.layer,
                }
                for tid, t in self.tasks.items()
            },
        }


# ── Execution engine ──────────────────────────────────────────────────────────

def _run_dag(exec_id: str, dag: DAG, context: dict):
    """Execute a DAG in a background thread."""
    try:
        order = dag.topological_order()
        total = len(order)
        task_results: dict = {}

        def _upd(**kw):
            with _lock:
                pipeline_executions[exec_id].update(kw)

        def _upd_task(tid: str, **kw):
            with _lock:
                pipeline_executions[exec_id]["task_states"][tid].update(kw)

        _upd(status="running", progress=0)

        for step_idx, tid in enumerate(order):
            task = dag.tasks[tid]

            _upd_task(tid, status="running",
                      started_at=datetime.utcnow().isoformat())

            log_line = f"[{datetime.utcnow().strftime('%H:%M:%S')}] ▶  {task.name}"
            with _lock:
                pipeline_executions[exec_id]["logs"].append(log_line)

            try:
                result = task.func(context, task_results)
                task_results[tid] = result

                _upd_task(tid, status="success",
                          finished_at=datetime.utcnow().isoformat(),
                          result=str(result)[:200] if result is not None else "OK")

                ok_line = f"[{datetime.utcnow().strftime('%H:%M:%S')}] ✔  {task.name} — OK"
                with _lock:
                    pipeline_executions[exec_id]["logs"].append(ok_line)

            except Exception as exc:
                _upd_task(tid, status="failed",
                          finished_at=datetime.utcnow().isoformat(),
                          error=str(exc))
                err_line = f"[{datetime.utcnow().strftime('%H:%M:%S')}] ✖  {task.name} — {exc}"
                with _lock:
                    pipeline_executions[exec_id]["logs"].append(err_line)
                # Continue with remaining tasks (soft failure)

            progress = int(100 * (step_idx + 1) / total)
            _upd(progress=progress)

            time.sleep(0.4)   # small delay so the UI can animate

        _upd(status="completed", progress=100,
             completed_at=datetime.utcnow().isoformat())

    except Exception as exc:
        with _lock:
            pipeline_executions[exec_id]["status"] = "failed"
            pipeline_executions[exec_id]["error"]  = str(exc)


def execute_dag(dag: DAG, context: dict | None = None) -> str:
    """Start DAG execution in a background thread; return exec_id."""
    exec_id = str(uuid.uuid4())[:8]
    task_states = {
        tid: {"status": "pending", "started_at": None,
              "finished_at": None, "result": None, "error": None}
        for tid in dag.tasks
    }
    with _lock:
        pipeline_executions[exec_id] = {
            "exec_id":    exec_id,
            "dag_id":     dag.dag_id,
            "dag_name":   dag.name,
            "status":     "queued",
            "progress":   0,
            "task_states": task_states,
            "logs":       [f"[{datetime.utcnow().strftime('%H:%M:%S')}] DAG '{dag.name}' queued"],
            "created_at": datetime.utcnow().isoformat(),
        }

    t = threading.Thread(target=_run_dag, args=(exec_id, dag, context or {}), daemon=True)
    t.start()
    return exec_id
