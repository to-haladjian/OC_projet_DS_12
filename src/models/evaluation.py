"""Metrics and comparisons, shared by both services.

RMSE and R² are the main metrics; MAE complements them by giving the error directly
in t/ha.

Two precautions guide this module.

1. **Compare paired.** Two models are not judged on their standard deviations across
   folds. In temporal validation, that standard deviation mostly measures the
   difference between years — a hard year is hard for every model — and not the
   uncertainty on the gap between two models. We therefore compare **fold by fold**,
   on the same folds.
2. **Comparing error across crops requires a relative error.** Yields range from
   1.3 to 16 t/ha depending on the crop: an RMSE of 2 t/ha is excellent for potato
   and catastrophic for sorghum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_validate

#: RMSE and MAE are negative in scikit-learn: a larger score must be better.
SCORES = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def validation_par_pli(modele, X, y, cv) -> pd.DataFrame:
    """Cross-validation keeping the **per-fold detail**.

    `cross_validate` refits a copy of the model on each fold: preprocessing placed in
    the pipeline is therefore relearned without seeing the validation fold — no leakage.

    The per-fold detail is what makes the paired comparison possible; a mean alone
    would not allow it.
    """
    scores = cross_validate(modele, X, y, cv=cv, scoring=SCORES, error_score="raise")
    plis = pd.DataFrame(
        {
            # "test" here means the validation fold, not the project's test set
            "rmse": -scores["test_rmse"],
            "mae": -scores["test_mae"],
            "r2": scores["test_r2"],
            "duree_ajustement": scores["fit_time"],
        }
    )
    plis.index = pd.RangeIndex(1, len(plis) + 1, name="pli")
    return plis


def resume(plis: pd.DataFrame) -> dict[str, float]:
    """Per-fold means and standard deviations, named like the MLflow metrics."""
    metriques = {}
    for nom in ("rmse", "mae", "r2"):
        valeurs = plis[nom].to_numpy()
        metriques[f"cv_{nom}_mean"] = float(valeurs.mean())
        metriques[f"cv_{nom}_std"] = float(valeurs.std())
    return metriques


def compare_appariee(
    resultats: dict[str, pd.DataFrame], reference: str, libelles: list | None = None
) -> pd.DataFrame:
    """Compare several evaluations **fold by fold**, on the same folds.

    For each evaluation: mean RMSE, mean gap to the reference computed fold by fold,
    and **number of improved folds**. The latter figure is the most telling: a model
    that wins on average but on a single fold has won nothing.

    The standard deviation across folds is deliberately absent (see the module header).
    `libelles` — for example the validation years — adds each fold's RMSE, to see
    whether a gap holds everywhere or comes from a single year.
    """
    rmse_reference = resultats[reference]["rmse"].to_numpy()
    if libelles is not None and len(libelles) != len(rmse_reference):
        raise ValueError("un libelle par pli est attendu")

    lignes = {}
    for nom, plis in resultats.items():
        if len(plis) != len(rmse_reference):
            raise ValueError(f"{nom} : {len(plis)} plis, {len(rmse_reference)} pour la reference")
        ecarts = plis["rmse"].to_numpy() - rmse_reference
        lignes[nom] = {
            "RMSE": plis["rmse"].mean(),
            "MAE": plis["mae"].mean(),
            "R2": plis["r2"].mean(),
            "ecart a la reference": ecarts.mean(),
            "plis ameliores": f"{int((ecarts < 0).sum())}/{len(ecarts)}",
        }
        if libelles is not None:
            lignes[nom] |= dict(zip(libelles, plis["rmse"].to_numpy().round(4)))
    return pd.DataFrame.from_dict(lignes, orient="index")


def predictions_hors_apprentissage(modele, X: pd.DataFrame, y: pd.Series, plis) -> pd.Series:
    """Out-of-fold predictions, even when the folds do not cover everything.

    `cross_val_predict` requires every row to belong to a validation fold. In temporal
    validation, only the last years are evaluated: it is therefore not usable. This
    function returns a Series restricted to the rows actually evaluated, indexed like
    `X`.

    A row evaluated in two folds raises an error: one prediction per row.
    """
    plis = plis.split(X, y) if hasattr(plis, "split") else plis

    morceaux = []
    for positions_app, positions_val in plis:
        ajuste = clone(modele).fit(X.iloc[positions_app], y.iloc[positions_app])
        validation = X.iloc[positions_val]
        morceaux.append(
            pd.Series(ajuste.predict(validation), index=validation.index, name="prediction")
        )

    predictions = pd.concat(morceaux)
    if predictions.index.has_duplicates:
        raise AssertionError("une ligne est evaluee dans plusieurs plis")
    return predictions


def erreurs_par_culture(y_observe, y_predit, cultures) -> pd.DataFrame:
    """Crop-by-crop error, with an **RMSE relative** to the yield level.

    Yields range from 1.3 to 16 t/ha depending on the crop: only the RMSE divided by
    the mean yield makes errors comparable. The bias (predicted - observed) shows
    whether the model systematically over- or underestimates a crop.
    """
    table = pd.DataFrame(
        {"culture": list(cultures), "observe": list(y_observe), "predit": list(y_predit)}
    )
    if table.empty or table.isna().any().any():
        raise ValueError("entrees vides ou comportant des valeurs manquantes")
    table["erreur"] = table["predit"] - table["observe"]

    par_culture = table.groupby("culture")
    resultat = pd.DataFrame(
        {
            "n": par_culture.size(),
            "rendement moyen": par_culture["observe"].mean(),
            "RMSE": par_culture["erreur"].apply(lambda e: float((e**2).mean() ** 0.5)),
            "MAE": par_culture["erreur"].apply(lambda e: float(e.abs().mean())),
            "biais": par_culture["erreur"].mean(),
        }
    )
    resultat["RMSE relative"] = resultat["RMSE"] / resultat["rendement moyen"]
    return resultat.sort_values("rendement moyen")


def metriques(y_observe, y_predit, prefixe: str = "test") -> dict[str, float]:
    """RMSE, MAE and R² on a given set. Reserved for the final evaluation."""
    return {
        f"{prefixe}_rmse": float(root_mean_squared_error(y_observe, y_predit)),
        f"{prefixe}_mae": float(mean_absolute_error(y_observe, y_predit)),
        f"{prefixe}_r2": float(r2_score(y_observe, y_predit)),
    }


def ligne_lisible(m: dict[str, float]) -> str:
    """A one-line summary: `RMSE 0,4993 ± 0,0006 t/ha | MAE ... | R² ...`."""
    return (
        f"RMSE {m['cv_rmse_mean']:.4f} ± {m['cv_rmse_std']:.4f} t/ha"
        f" | MAE {m['cv_mae_mean']:.4f} t/ha"
        f" | R² {m['cv_r2_mean']:.4f}"
    )


# --------------------------------------------------------------------------- #
# Metrics specific to a ranking task
# --------------------------------------------------------------------------- #

def qualite_classement(
    observe: pd.DataFrame, predit: pd.Series, groupe: list[str] | None = None,
    colonne_culture: str = "crop", colonne_cible: str = "yield_t_ha",
) -> dict[str, float]:
    """Quality of the crop **ranking**, within each country-year.

    An R² measures the error on yield; it does not say whether the proposed order is
    right. Yet the order is what the user sees. Three measures, per group then averaged:

    * `top1`: is the crop ranked first the one with the best observed yield?
    * `top3`: is the best observed crop among the first three?
    * `spearman`: rank correlation between predicted and observed rankings.

    And what a wrong first place **costs**, in the unit of `colonne_cible`:

    * `regret`: observed value of the best crop minus observed value of the crop
      ranked first, averaged over groups — zero when the first place is right;
    * `regret_relatif`: the same gap divided by the best observed value. With a gross
      product target, this is the share of revenue lost by following the ranking.
    """
    groupe = groupe or ["iso3", "year"]
    table = observe[groupe + [colonne_culture, colonne_cible]].copy()
    table["predit"] = predit.to_numpy()

    top1, top3, rhos, regrets, regrets_relatifs = [], [], [], [], []
    for _, bloc in table.groupby(groupe):
        if len(bloc) < 2:
            continue
        meilleure = bloc[colonne_cible].max()
        meilleure_observee = bloc.loc[bloc[colonne_cible].idxmax(), colonne_culture]
        ordre_predit = bloc.sort_values("predit", ascending=False)
        cultures_predites = ordre_predit[colonne_culture].tolist()
        top1.append(cultures_predites[0] == meilleure_observee)
        top3.append(meilleure_observee in cultures_predites[:3])
        regret = meilleure - ordre_predit[colonne_cible].iloc[0]
        regrets.append(regret)
        regrets_relatifs.append(regret / meilleure if meilleure > 0 else 0.0)
        if len(bloc) >= 3:
            rho = bloc[colonne_cible].corr(bloc["predit"], method="spearman")
            if not np.isnan(rho):
                rhos.append(rho)

    def moyenne(valeurs):
        return float(np.mean(valeurs)) if valeurs else float("nan")

    return {
        "top1": moyenne(top1),
        "top3": moyenne(top3),
        "spearman": moyenne(rhos),
        "regret": moyenne(regrets),
        "regret_relatif": moyenne(regrets_relatifs),
        "groupes": len(top1),
    }
