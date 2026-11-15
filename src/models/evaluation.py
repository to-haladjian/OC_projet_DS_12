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


def validate_by_fold(model, X, y, cv) -> pd.DataFrame:
    """Cross-validation keeping the **per-fold detail**.

    `cross_validate` refits a copy of the model on each fold: preprocessing placed in
    the pipeline is therefore relearned without seeing the validation fold — no leakage.

    The per-fold detail is what makes the paired comparison possible; a mean alone
    would not allow it.
    """
    scores = cross_validate(model, X, y, cv=cv, scoring=SCORES, error_score="raise")
    folds = pd.DataFrame(
        {
            # "test" here means the validation fold, not the project's test set
            "rmse": -scores["test_rmse"],
            "mae": -scores["test_mae"],
            "r2": scores["test_r2"],
            "fit_seconds": scores["fit_time"],
        }
    )
    folds.index = pd.RangeIndex(1, len(folds) + 1, name="fold")
    return folds


def summarise(folds: pd.DataFrame) -> dict[str, float]:
    """Per-fold means and standard deviations, named like the MLflow metrics."""
    metrics = {}
    for name in ("rmse", "mae", "r2"):
        values = folds[name].to_numpy()
        metrics[f"cv_{name}_mean"] = float(values.mean())
        metrics[f"cv_{name}_std"] = float(values.std())
    return metrics


def compare_paired(
    results: dict[str, pd.DataFrame], reference: str, labels: list | None = None
) -> pd.DataFrame:
    """Compare several evaluations **fold by fold**, on the same folds.

    For each evaluation: mean RMSE, mean gap to the reference computed fold by fold,
    and **number of improved folds**. The latter figure is the most telling: a model
    that wins on average but on a single fold has won nothing.

    The standard deviation across folds is deliberately absent (see the module header).
    `labels` — for example the validation years — adds each fold's RMSE, to see
    whether a gap holds everywhere or comes from a single year.
    """
    rmse_reference = results[reference]["rmse"].to_numpy()
    if labels is not None and len(labels) != len(rmse_reference):
        raise ValueError("un libelle par pli est attendu")

    rows = {}
    for name, folds in results.items():
        if len(folds) != len(rmse_reference):
            raise ValueError(f"{name} : {len(folds)} plis, {len(rmse_reference)} pour la reference")
        gaps = folds["rmse"].to_numpy() - rmse_reference
        rows[name] = {
            "RMSE": folds["rmse"].mean(),
            "MAE": folds["mae"].mean(),
            "R2": folds["r2"].mean(),
            "ecart a la reference": gaps.mean(),
            "plis ameliores": f"{int((gaps < 0).sum())}/{len(gaps)}",
        }
        if labels is not None:
            rows[name] |= dict(zip(labels, folds["rmse"].to_numpy().round(4)))
    return pd.DataFrame.from_dict(rows, orient="index")


def out_of_fold_predictions(model, X: pd.DataFrame, y: pd.Series, folds) -> pd.Series:
    """Out-of-fold predictions, even when the folds do not cover everything.

    `cross_val_predict` requires every row to belong to a validation fold. In temporal
    validation, only the last years are evaluated: it is therefore not usable. This
    function returns a Series restricted to the rows actually evaluated, indexed like
    `X`.

    A row evaluated in two folds raises an error: one prediction per row.
    """
    folds = folds.split(X, y) if hasattr(folds, "split") else folds

    chunks = []
    for train_positions, positions_val in folds:
        fitted = clone(model).fit(X.iloc[train_positions], y.iloc[train_positions])
        validation = X.iloc[positions_val]
        chunks.append(
            pd.Series(fitted.predict(validation), index=validation.index, name="prediction")
        )

    predictions = pd.concat(chunks)
    if predictions.index.has_duplicates:
        raise AssertionError("une ligne est evaluee dans plusieurs plis")
    return predictions


def errors_by_crop(y_observed, y_predicted, crops) -> pd.DataFrame:
    """Crop-by-crop error, with an **RMSE relative** to the yield level.

    Yields range from 1.3 to 16 t/ha depending on the crop: only the RMSE divided by
    the mean yield makes errors comparable. The bias (predicted - observed) shows
    whether the model systematically over- or underestimates a crop.
    """
    table = pd.DataFrame(
        {"crop": list(crops), "observed": list(y_observed), "predicted": list(y_predicted)}
    )
    if table.empty or table.isna().any().any():
        raise ValueError("entrees vides ou comportant des valeurs manquantes")
    table["error"] = table["predicted"] - table["observed"]

    by_crop = table.groupby("crop")
    result = pd.DataFrame(
        {
            "n": by_crop.size(),
            "rendement moyen": by_crop["observed"].mean(),
            "RMSE": by_crop["error"].apply(lambda e: float((e**2).mean() ** 0.5)),
            "MAE": by_crop["error"].apply(lambda e: float(e.abs().mean())),
            "biais": by_crop["error"].mean(),
        }
    )
    result["RMSE relative"] = result["RMSE"] / result["rendement moyen"]
    return result.sort_values("rendement moyen")


def metrics(y_observed, y_predicted, prefix: str = "test") -> dict[str, float]:
    """RMSE, MAE and R² on a given set. Reserved for the final evaluation."""
    return {
        f"{prefix}_rmse": float(root_mean_squared_error(y_observed, y_predicted)),
        f"{prefix}_mae": float(mean_absolute_error(y_observed, y_predicted)),
        f"{prefix}_r2": float(r2_score(y_observed, y_predicted)),
    }


def readable_line(m: dict[str, float]) -> str:
    """A one-line summary: `RMSE 0,4993 ± 0,0006 t/ha | MAE ... | R² ...`."""
    return (
        f"RMSE {m['cv_rmse_mean']:.4f} ± {m['cv_rmse_std']:.4f} t/ha"
        f" | MAE {m['cv_mae_mean']:.4f} t/ha"
        f" | R² {m['cv_r2_mean']:.4f}"
    )


# --------------------------------------------------------------------------- #
# Metrics specific to a ranking task
# --------------------------------------------------------------------------- #

def ranking_quality(
    observed: pd.DataFrame, predicted: pd.Series, group: list[str] | None = None,
    crop_column: str = "crop", target_column: str = "yield_t_ha",
) -> dict[str, float]:
    """Quality of the crop **ranking**, within each country-year.

    An R² measures the error on yield; it does not say whether the proposed order is
    right. Yet the order is what the user sees. Three measures, per group then averaged:

    * `top1`: is the crop ranked first the one with the best observed yield?
    * `top3`: is the best observed crop among the first three?
    * `spearman`: rank correlation between predicted and observed rankings.

    And what a wrong first place **costs**, in the unit of `target_column`:

    * `regret`: observed value of the best crop minus observed value of the crop
      ranked first, averaged over groups — zero when the first place is right;
    * `relative_regret`: the same gap divided by the best observed value. With a gross
      product target, this is the share of revenue lost by following the ranking.
    """
    group = group or ["iso3", "year"]
    table = observed[group + [crop_column, target_column]].copy()
    table["predicted"] = predicted.to_numpy()

    top1, top3, rhos, regrets, relative_regrets = [], [], [], [], []
    for _, block in table.groupby(group):
        if len(block) < 2:
            continue
        best_value = block[target_column].max()
        best_observed = block.loc[block[target_column].idxmax(), crop_column]
        predicted_order = block.sort_values("predicted", ascending=False)
        predicted_crops = predicted_order[crop_column].tolist()
        top1.append(predicted_crops[0] == best_observed)
        top3.append(best_observed in predicted_crops[:3])
        regret = best_value - predicted_order[target_column].iloc[0]
        regrets.append(regret)
        relative_regrets.append(regret / best_value if best_value > 0 else 0.0)
        if len(block) >= 3:
            rho = block[target_column].corr(block["predicted"], method="spearman")
            if not np.isnan(rho):
                rhos.append(rho)

    def mean_of(values):
        return float(np.mean(values)) if values else float("nan")

    return {
        "top1": mean_of(top1),
        "top3": mean_of(top3),
        "spearman": mean_of(rhos),
        "regret": mean_of(regrets),
        "relative_regret": mean_of(relative_regrets),
        "groups": len(top1),
    }
