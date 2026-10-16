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

EXPERIENCE_PREDICT = "agritech-predict"
EXPERIENCE_RECOMMEND = "agritech-recommend"


def configure(experience: str) -> None:
    """Point MLflow at the local database and select the experiment.

    The store is a SQLite database in the repository. MLflow put its old
    "file directory" store into maintenance mode as of version 3.16: SQLite is the
    supported path, and stays a plain local file — no server, no paid service.
    """
    config.MLFLOW_ARTEFACTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{config.MLFLOW_BASE}")

    # `delete_experiment` only performs a soft delete: the name stays reserved, and
    # `set_experiment` then fails with "Cannot set a deleted experiment". Cleaning its
    # runs to start from a fresh experiment is therefore enough to block any later
    # execution. We restore the experiment if it is in that state.
    client = mlflow.MlflowClient()
    existante = client.get_experiment_by_name(experience)
    if existante is not None and existante.lifecycle_stage == "deleted":
        client.restore_experiment(existante.experiment_id)

    mlflow.set_experiment(experience)


@contextmanager
def run(nom: str, parametres: dict[str, Any], etape: str, note: str = ""):
    """Open an MLflow run, log the parameters, and let the caller log the metrics.

    `etape` places the run within the workflow (`reference`, `ablation`, `optimisation`,
    `evaluation_finale`): this is what allows filtering runs in the UI.
    `note` records the interpretation — a run without annotation is hard to reread.
    """
    with mlflow.start_run(run_name=nom):
        mlflow.log_params({**parametres, "seed": config.SEED, "etape": etape})
        if note:
            mlflow.set_tag("mlflow.note.content", note)
        yield mlflow


def journalise(nom: str, parametres: dict[str, Any], metriques: dict[str, float],
               etape: str, note: str = "") -> None:
    """Log a complete evaluation in one call."""
    with run(nom, parametres, etape, note) as suivi:
        suivi.log_metrics(metriques)
