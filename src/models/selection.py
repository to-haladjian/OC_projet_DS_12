"""Which model family should serve the yield, and what do the conditions explain?

Every candidate family is evaluated on the **same folds**, under the two protocols of
:mod:`src.models.protocol`, and tracked on the shared schema:

* **temporal** — predict a year from the ones before it, for countries the model knows.
  This is the service's situation, and the protocol the served family is chosen on;
* **leave-country-out** — predict countries the model never saw. It isolates what the
  conditions and the position explain, without any memory of the country.

The decision is made by a **paired** comparison (:func:`evaluation.compare_paired`):
a family replaces the served one only if it wins fold by fold, not on a mean.

Runnable as a script: `python -m src.models.selection` (or `make select-yield`).
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config, tracking
from src.models import evaluation, features, protocol
from src.models import yield_model as ym

#: The family currently served, and the reference of every paired comparison.
SERVED = "ExtraTreesRegressor"

#: Baseline feature set: the crop alone. A model that only knows the crop predicts its
#: mean yield; whatever a candidate gains over it is what the conditions bring.
CROP_ONLY = ["crop"]


def candidates() -> dict[str, tuple[Pipeline, str]]:
    """Candidate pipelines and the name of their feature set, keyed by display name."""
    seed = config.SEED
    return {
        "crop mean (baseline)": (ym.pipeline(LinearRegression(), CROP_ONLY), "crop_only"),
        "Ridge": (ym.pipeline(make_pipeline(StandardScaler(), Ridge())), "history"),
        "HistGradientBoostingRegressor": (
            ym.pipeline(HistGradientBoostingRegressor(
                max_iter=600, learning_rate=0.05, early_stopping=False, random_state=seed,
            )),
            "history",
        ),
        "RandomForestRegressor": (
            ym.pipeline(RandomForestRegressor(n_estimators=150, random_state=seed, n_jobs=1)),
            "history",
        ),
        SERVED: (ym.selected_model(), "history"),
    }


def folds_by_protocol(df: pd.DataFrame) -> dict[str, list]:
    """The two sets of folds, as positions in `df` (years before the test year only)."""
    if (df["year"] >= config.TEST_YEAR).any():
        raise AssertionError("l'annee de test ne doit servir a aucun choix de modele")
    return {
        "temporal": protocol.temporal_cv(df["year"], ym.VALIDATION_YEARS),
        "leave_country_out": protocol.country_cv(df["iso3"]),
    }


def evaluate(df: pd.DataFrame | None = None, verbose: bool = True) -> dict[str, dict[str, pd.DataFrame]]:
    """Evaluate and track every candidate under both protocols.

    Returns `{protocol: {candidate: per-fold scores}}`, ready for a paired comparison.
    """
    df = ym.load() if df is None else df
    df = df[df["year"] < config.TEST_YEAR].reset_index(drop=True)
    y = df[features.YIELD_TARGET]

    tracking.configure(tracking.EXPERIMENT_YIELD)
    results: dict[str, dict[str, pd.DataFrame]] = {}
    for protocol_name, folds in folds_by_protocol(df).items():
        results[protocol_name] = {}
        for name, (model, feature_set) in candidates().items():
            variables = CROP_ONLY if feature_set == "crop_only" else ym.VARIABLES
            per_fold = evaluation.validate_by_fold(model, df[variables], y, folds)
            results[protocol_name][name] = per_fold
            metrics = evaluation.summarise(per_fold)
            tracking.log_evaluation(
                f"{name} — {protocol_name}",
                tracking.describe(model, tracking.DATASET_FAO, feature_set, protocol_name,
                                  n_features=len(variables)),
                metrics, phase="reference",
                note=f"Comparaison des familles, mêmes plis. {evaluation.readable_line(metrics)}",
            )
            if verbose:
                print(f"  {protocol_name:18} {name:30} {evaluation.readable_line(metrics)}")
    return results


def main() -> None:
    results = evaluate()
    pd.set_option("display.width", 200)
    for protocol_name, per_candidate in results.items():
        labels = ym.VALIDATION_YEARS if protocol_name == "temporal" else None
        print(f"\n== {protocol_name} — comparaison appariee, reference : {SERVED}")
        print(evaluation.compare_paired(per_candidate, reference=SERVED, labels=labels)
              .round(3).to_string())


if __name__ == "__main__":
    main()
