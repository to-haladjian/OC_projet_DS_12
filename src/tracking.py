"""Experiment tracking with MLflow, locally.

The store is a **file directory** in `mlruns/`: no server, no remote service,
no cost. The UI can be browsed with `mlflow ui --backend-store-uri mlruns`.

One run per evaluation. Parameters describe *how* the evaluation was done (model,
feature set, protocol, seed) and metrics *what it yielded*, so that the run table
reads directly as a ranking.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import mlflow

from src import config

EXPERIMENT_PRACTICE = "agritech-predict"
EXPERIMENT_YIELD = "agritech-recommend"


def configure(experiment: str) -> None:
    """Point MLflow at the local database and select the experiment.

    The store is a SQLite database in the repository. MLflow put its old
    "file directory" store into maintenance mode as of version 3.16: SQLite is the
    supported path, and stays a plain local file — no server, no paid service.
    """
    config.MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{config.MLFLOW_DB}")

    # `delete_experiment` only performs a soft delete: the name stays reserved, and
    # `set_experiment` then fails with "Cannot set a deleted experiment". Cleaning its
    # runs to start from a fresh experiment is therefore enough to block any later
    # execution. We restore the experiment if it is in that state.
    client = mlflow.MlflowClient()
    existing = client.get_experiment_by_name(experiment)
    if existing is not None and existing.lifecycle_stage == "deleted":
        client.restore_experiment(existing.experiment_id)

    mlflow.set_experiment(experiment)


@contextmanager
def run(name: str, parameters: dict[str, Any], phase: str, note: str = ""):
    """Open an MLflow run, log the parameters, and let the caller log the metrics.

    `phase` places the run within the workflow (`reference`, `ablation`, `optimisation`,
    `final_evaluation`): this is what allows filtering runs in the UI.
    `note` records the interpretation — a run without annotation is hard to reread.
    """
    with mlflow.start_run(run_name=name):
        mlflow.log_params({**parameters, "seed": config.SEED, "phase": phase})
        if note:
            mlflow.set_tag("mlflow.note.content", note)
        yield mlflow


def log_evaluation(name: str, parameters: dict[str, Any], metrics: dict[str, float],
               phase: str, note: str = "") -> None:
    """Log a complete evaluation in one call."""
    with run(name, parameters, phase, note) as tracker:
        tracker.log_metrics(metrics)
