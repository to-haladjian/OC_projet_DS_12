"""Experiment tracking with MLflow, on one schema so that runs can be compared.

The store is a local SQLite database (`mlflow.db`): no server, no remote service, no
cost. Browse it with `make mlflow-ui`; `make compare` prints the same comparison as a
table and writes it to `reports/model_comparison.csv`.

**Why a schema.** A run is only comparable to another if both answer the same
questions under the same names. Every run therefore carries:

* the same **description** (:data:`DESCRIPTION`): model family, dataset, feature set,
  validation protocol, phase of the work and seed — the columns to filter and group on;
* the same **cross-validation metrics** (:data:`evaluation.CV_METRICS`): RMSE, MAE, R²
  with their spread across folds, the unit-free NRMSE, and the fit time;
* for a final evaluation, the same **held-out metrics** (:data:`TEST_METRICS`).

:func:`log_evaluation` refuses a run that misses any of them: an incomplete run would
leave a hole in the comparison, and holes are what make a table unreadable.

RMSE and MAE are in t/ha and only compare within a dataset. `cv_nrmse_mean` and
`test_nrmse` divide the RMSE by the mean observed yield: they compare across datasets.
"""

from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.models import evaluation

EXPERIMENT_YIELD = "agritech-yield-engine"
EXPERIMENT_PRACTICE = "agritech-practice-effects"
EXPERIMENTS = (EXPERIMENT_YIELD, EXPERIMENT_PRACTICE)

DATASET_FAO = "fao"
DATASET_SIMULATED = "simulated"

#: Phases of the work a run can belong to.
PHASES = ("reference", "ablation", "optimisation", "final_evaluation")

#: Parameters every run must describe itself with.
DESCRIPTION = ("model_family", "dataset", "feature_set", "protocol")

#: Metrics a final evaluation adds to the cross-validation ones.
TEST_METRICS = ("test_rmse", "test_mae", "test_r2", "test_nrmse")


def configure(experiment: str) -> None:
    """Point MLflow at the local database and select the experiment."""
    config.MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{config.MLFLOW_DB}")

    # `delete_experiment` only performs a soft delete: the name stays reserved, and
    # `set_experiment` then fails with "Cannot set a deleted experiment". We restore
    # the experiment if it is in that state.
    client = mlflow.MlflowClient()
    existing = client.get_experiment_by_name(experiment)
    if existing is not None and existing.lifecycle_stage == "deleted":
        client.restore_experiment(existing.experiment_id)

    mlflow.set_experiment(experiment)


def family(model) -> str:
    """Class name of the estimator that learns, whatever wraps it.

    A pipeline is named after its last step, so that `ExtraTreesRegressor` reads the
    same whether it was tracked bare, in a pipeline, or as a search candidate.
    """
    while isinstance(model, Pipeline):
        model = model.steps[-1][1]
    return type(model).__name__


def describe(model, dataset: str, feature_set: str, protocol: str, **extra: Any) -> dict[str, Any]:
    """The description of a run, with the family read from the model itself."""
    return {
        "model_family": family(model),
        "dataset": dataset,
        "feature_set": feature_set,
        "protocol": protocol,
        **extra,
    }


def _check(description: dict[str, Any], metrics: dict[str, float], phase: str) -> None:
    """Refuse a run that would leave a hole in the comparison."""
    if phase not in PHASES:
        raise ValueError(f"phase inconnue : {phase!r}, attendu parmi {PHASES}")
    missing = [key for key in DESCRIPTION if key not in description]
    if missing:
        raise ValueError(f"description incomplete, il manque : {missing}")
    expected = list(evaluation.CV_METRICS)
    if phase == "final_evaluation":
        expected += TEST_METRICS
    missing = [key for key in expected if key not in metrics]
    if missing:
        raise ValueError(f"metriques manquantes pour un run '{phase}' : {missing}")


def log_evaluation(
    name: str, description: dict[str, Any], metrics: dict[str, float], phase: str,
    note: str = "", nested: bool = False,
) -> str:
    """Log one complete evaluation and return its run id.

    `description` holds the keys of :data:`DESCRIPTION` plus any hyperparameter;
    `metrics` holds at least :data:`evaluation.CV_METRICS`. `note` records the
    interpretation — a run without annotation is hard to reread. `nested` attaches the
    run to the one currently open (a search and its candidates).
    """
    _check(description, metrics, phase)
    with mlflow.start_run(run_name=name, nested=nested) as run:
        mlflow.log_params({**description, "phase": phase, "seed": config.SEED})
        # Tags, unlike parameters, can be filtered on in the UI search box:
        # `tags.phase = "final_evaluation"`.
        mlflow.set_tags({"phase": phase, **{key: description[key] for key in DESCRIPTION}})
        if note:
            mlflow.set_tag("mlflow.note.content", note)
        mlflow.log_metrics(metrics)
        return run.info.run_id


# --------------------------------------------------------------------------- #
# Reading the runs back, as one comparison table
# --------------------------------------------------------------------------- #

#: Columns of the comparison table, in reading order.
COMPARISON_METRICS = (
    "cv_rmse_mean", "cv_rmse_std", "cv_mae_mean", "cv_r2_mean", "cv_nrmse_mean",
    "test_rmse", "test_r2", "test_nrmse", "fit_seconds",
)


def comparison_table(experiments: tuple[str, ...] = EXPERIMENTS) -> pd.DataFrame:
    """Every tracked run, one row each, on the shared description and metrics.

    Sorted by experiment, protocol, then cross-validated RMSE: runs evaluated on the
    same folds sit together, best first. Across experiments, read the `nrmse` columns.
    """
    mlflow.set_tracking_uri(f"sqlite:///{config.MLFLOW_DB}")
    runs = mlflow.search_runs(experiment_names=list(experiments), output_format="pandas")
    if runs.empty:
        return pd.DataFrame()

    names = {
        e.experiment_id: e.name
        for e in mlflow.MlflowClient().search_experiments()
    }
    table = pd.DataFrame({"experiment": runs["experiment_id"].map(names)})
    table["run"] = runs["tags.mlflow.runName"]
    for key in (*DESCRIPTION, "phase"):
        table[key] = runs.get(f"params.{key}")
    for key in COMPARISON_METRICS:
        table[key] = runs.get(f"metrics.{key}")
    return table.sort_values(["experiment", "protocol", "cv_rmse_mean"]).reset_index(drop=True)


def main() -> None:
    table = comparison_table()
    if table.empty:
        print("aucun run : lancer `make train` ou `make notebooks` d'abord")
        return
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS / "model_comparison.csv"
    table.round(5).to_csv(path, index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 500)
    for experiment, block in table.groupby("experiment"):
        print(f"\n== {experiment} ({len(block)} runs)")
        print(block.drop(columns="experiment").round(4).to_string(index=False))
    print(f"\necrit {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
