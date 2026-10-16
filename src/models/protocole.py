"""Evaluation splits, and the checks that make them reliable.

Two protocols, because the two datasets do not have the same structure.

* **Simulated dataset** (`/predict`): no time dimension, rows are interchangeable.
  A random split and `KFold` validation are suitable.
* **Historical dataset** (`/recommend`): the yield of a country x crop pair is
  correlated at **0.985** with that of the previous year. A random split would put
  near-identical rows on both sides, and overestimate performance by a factor close
  to 5. So we must train on the past and predict the following year.

Each function checks its result: it is the only way for a leak not to go unnoticed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from src import config


def decoupage_aleatoire(
    df: pd.DataFrame, variables: list[str], cible: str, part_test: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train / test, once and for all.

    With the same seed, the test set is identical for every model: it can stay
    **reserved for the final evaluation** and serve no choice.
    """
    X_app, X_test, y_app, y_test = train_test_split(
        df[variables], df[cible], test_size=part_test, random_state=config.SEED
    )
    if len(X_app) + len(X_test) != len(df):
        raise AssertionError("des lignes ont ete perdues au decoupage")
    if not X_app.index.intersection(X_test.index).empty:
        raise AssertionError("des lignes sont communes a l'apprentissage et au test")
    return X_app, X_test, y_app, y_test


def cv_aleatoire(n_plis: int = 5) -> KFold:
    """Cross-validation to apply to the training set only.

    Same seed hence same folds from one model to the next: scores are comparable, and
    the paired comparison of :func:`src.models.evaluation.compare_appariee` makes sense.
    """
    return KFold(n_splits=n_plis, shuffle=True, random_state=config.SEED)


def decoupage_temporel(
    df: pd.DataFrame, variables: list[str], cible: str, annee_test: int | None = None,
    colonne_annee: str = "year",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train on prior years, test on `annee_test`.

    Reproduces the real situation of the service: recommending a season the model has
    never seen. Checks that no row is **later** than the test year — such a row would
    be in neither training nor test, and would vanish silently.
    """
    annee_test = config.ANNEE_TEST if annee_test is None else annee_test
    annees = df[colonne_annee]
    if not (annees == annee_test).any():
        raise AssertionError(f"l'annee de test {annee_test} est absente du jeu")
    if (annees > annee_test).any():
        raise AssertionError(f"des lignes sont posterieures a {annee_test}")

    app, test = df[annees < annee_test], df[annees == annee_test]
    return app[variables], test[variables], app[cible], test[cible]


def cv_temporelle(
    annees: pd.Series, annees_validation: list[int], annee_test: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One fold per validation year: training on **everything before it**.

    Returns positions (not index labels), directly usable by `cross_validate`. The
    folds deliberately do not cover all rows: only the last years are evaluated. This
    is why :func:`src.models.evaluation.predictions_hors_apprentissage` replaces
    `cross_val_predict`, which requires full coverage.
    """
    annee_test = config.ANNEE_TEST if annee_test is None else annee_test
    if any(a >= annee_test for a in annees_validation):
        raise AssertionError(
            f"une annee de validation atteint l'annee de test {annee_test} : "
            "le test ne doit intervenir qu'une fois, a la fin"
        )

    valeurs = annees.to_numpy()
    plis = []
    for annee in sorted(annees_validation):
        apprentissage = np.flatnonzero(valeurs < annee)
        validation = np.flatnonzero(valeurs == annee)
        if not len(apprentissage) or not len(validation):
            raise AssertionError(f"pli {annee} vide : apprentissage ou validation sans ligne")
        plis.append((apprentissage, validation))
    return plis


def resume_plis(plis: list[tuple[np.ndarray, np.ndarray]], annees_validation: list[int]) -> pd.DataFrame:
    """Readable table of folds: validated year, training rows, evaluated rows."""
    return pd.DataFrame(
        [
            {
                "annee validee": annee,
                "lignes d'apprentissage": len(app),
                "lignes evaluees": len(val),
            }
            for annee, (app, val) in zip(sorted(annees_validation), plis)
        ]
    )
