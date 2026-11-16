"""Evaluation splits, and the checks that make them reliable.

Three protocols, because the two datasets do not have the same structure.

* **Simulated dataset** (practice effects): no time dimension, rows are interchangeable.
  A random split and `KFold` validation are suitable.
* **Historical dataset** (yield model): the yield of a country x crop pair is
  correlated at **0.985** with that of the previous year. A random split would put
  near-identical rows on both sides, and overestimate performance by a factor close
  to 5. So we must train on the past and predict the following year. A third
  protocol, leave-country-out, measures what is left for a country never seen.

Each function checks its result: it is the only way for a leak not to go unnoticed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, train_test_split

from src import config


def random_split(
    df: pd.DataFrame, variables: list[str], target: str, test_share: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train / test, once and for all.

    With the same seed, the test set is identical for every model: it can stay
    **reserved for the final evaluation** and serve no choice.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df[variables], df[target], test_size=test_share, random_state=config.SEED
    )
    if len(X_train) + len(X_test) != len(df):
        raise AssertionError("des lignes ont ete perdues au decoupage")
    if not X_train.index.intersection(X_test.index).empty:
        raise AssertionError("des lignes sont communes a l'apprentissage et au test")
    return X_train, X_test, y_train, y_test


def random_cv(n_folds: int = 5) -> KFold:
    """Cross-validation to apply to the training set only.

    Same seed hence same folds from one model to the next: scores are comparable, and
    the paired comparison of :func:`src.models.evaluation.compare_paired` makes sense.
    """
    return KFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)


def temporal_split(
    df: pd.DataFrame, variables: list[str], target: str, test_year: int | None = None,
    year_column: str = "year",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train on prior years, test on `test_year`.

    Reproduces the real situation of the service: recommending a season the model has
    never seen. Checks that no row is **later** than the test year — such a row would
    be in neither training nor test, and would vanish silently.
    """
    test_year = config.TEST_YEAR if test_year is None else test_year
    years = df[year_column]
    if not (years == test_year).any():
        raise AssertionError(f"l'annee de test {test_year} est absente du jeu")
    if (years > test_year).any():
        raise AssertionError(f"des lignes sont posterieures a {test_year}")

    app, test = df[years < test_year], df[years == test_year]
    return app[variables], test[variables], app[target], test[target]


def temporal_cv(
    years: pd.Series, validation_years: list[int], test_year: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One fold per validation year: training on **everything before it**.

    Returns positions (not index labels), directly usable by `cross_validate`. The
    folds deliberately do not cover all rows: only the last years are evaluated. This
    is why :func:`src.models.evaluation.out_of_fold_predictions` replaces
    `cross_val_predict`, which requires full coverage.
    """
    test_year = config.TEST_YEAR if test_year is None else test_year
    if any(a >= test_year for a in validation_years):
        raise AssertionError(
            f"une annee de validation atteint l'annee de test {test_year} : "
            "le test ne doit intervenir qu'une fois, a la fin"
        )

    values = years.to_numpy()
    folds = []
    for year in sorted(validation_years):
        training = np.flatnonzero(values < year)
        validation = np.flatnonzero(values == year)
        if not len(training) or not len(validation):
            raise AssertionError(f"pli {year} vide : apprentissage ou validation sans ligne")
        folds.append((training, validation))
    return folds


def country_cv(countries: pd.Series, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-country-out folds: every country is validated by a model that never saw it.

    The temporal protocol answers "how well do we predict next year for a country we
    know?". This one answers the other question a recommendation raises: "what do the
    conditions alone tell us, for a place the model has no history of?". A country
    appears in exactly one validation fold, and never on both sides of a fold.
    """
    values = countries.to_numpy()
    folds = list(GroupKFold(n_splits=n_folds).split(values, groups=values))
    for training, validation in folds:
        if set(values[training]) & set(values[validation]):
            raise AssertionError("un pays figure des deux cotes d'un pli")
    return folds


def fold_summary(folds: list[tuple[np.ndarray, np.ndarray]], validation_years: list[int]) -> pd.DataFrame:
    """Readable table of folds: validated year, training rows, evaluated rows."""
    return pd.DataFrame(
        [
            {
                "annee validee": year,
                "lignes d'apprentissage": len(app),
                "lignes evaluees": len(val),
            }
            for year, (app, val) in zip(sorted(validation_years), folds)
        ]
    )
