"""Hyperparameter search whose every candidate is tracked like any other run.

A search that only reports its best RMSE cannot be compared with anything: neither
with the other families, nor with the reference it claims to beat. Here each candidate
is scored with the scorers shared by the whole project (:data:`evaluation.SCORES`) and
logged with the full set of metrics, under one parent run per search.
"""

from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from src import config, tracking
from src.models import evaluation

#: Prefix of the estimator step in the project's pipelines.
STEP = "model__"


def _candidate_metrics(results: dict, i: int) -> dict[str, float]:
    """Metrics of candidate `i`, named like those of any tracked run."""
    return {
        "cv_rmse_mean": float(-results["mean_test_rmse"][i]),
        "cv_rmse_std": float(results["std_test_rmse"][i]),
        "cv_mae_mean": float(-results["mean_test_mae"][i]),
        "cv_mae_std": float(results["std_test_mae"][i]),
        "cv_r2_mean": float(results["mean_test_r2"][i]),
        "cv_r2_std": float(results["std_test_r2"][i]),
        "cv_nrmse_mean": float(-results["mean_test_nrmse"][i]),
        "fit_seconds": float(results["mean_fit_time"][i]),
    }


def random_search(
    pipeline, grid: dict[str, list], X, y, cv, description: dict[str, Any],
    n_iter: int = 10, note: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Random search over `grid`, every candidate tracked.

    `description` is the run description (see :func:`tracking.describe`); the
    hyperparameters of each candidate are added to it. Returns the candidates as a
    table sorted by cross-validated RMSE, and the hyperparameters of the best one.

    `refit=False`: the search only measures. The served model is refitted elsewhere,
    once, on a choice made by a paired comparison rather than by the best mean.
    """
    search = RandomizedSearchCV(
        pipeline, grid, n_iter=n_iter, cv=cv, scoring=evaluation.SCORES, refit=False,
        random_state=config.SEED, n_jobs=1, error_score="raise",
    ).fit(X, y)
    results = search.cv_results_

    rows = []
    for i, params in enumerate(results["params"]):
        own = {key.removeprefix(STEP): value for key, value in params.items()}
        rows.append({"candidate": i, **own, **_candidate_metrics(results, i)})
    table = pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True)
    best = int(table.loc[0, "candidate"])
    best_params = {key.removeprefix(STEP): value for key, value in results["params"][best].items()}

    name = description["model_family"]
    with mlflow.start_run(run_name=f"{name} — recherche ({len(rows)} candidats)"):
        mlflow.log_params({**description, "phase": "optimisation", "seed": config.SEED,
                           "n_candidates": len(rows)})
        mlflow.set_tags({"phase": "optimisation", "search": "parent",
                         **{key: description[key] for key in tracking.DESCRIPTION}})
        mlflow.set_tag(
            "mlflow.note.content",
            note or "Recherche aléatoire ; les métriques de ce run sont celles du meilleur candidat.",
        )
        mlflow.log_metrics(_candidate_metrics(results, best))
        for i, params in enumerate(results["params"]):
            own = {key.removeprefix(STEP): value for key, value in params.items()}
            tracking.log_evaluation(
                f"{name} — candidat {i}", {**description, **own},
                _candidate_metrics(results, i), phase="optimisation", nested=True,
                note="Candidat tiré au hasard dans la grille, mêmes plis que la référence.",
            )
    return table, best_params
