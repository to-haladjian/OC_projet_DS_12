"""`/recommend` model: ranking of crops by expected yield.

This is where the step 2 modelling work lives. Unlike `/predict`, this service relies
on a dataset where **the crop actually predicts yield**: the FAO dataset rebuilt in
step 1.

Three design choices shape this module.

1. **Temporal validation, never random.** The yield of a country x crop pair is
   correlated at 0.985 with that of the previous year. A random split would put
   near-identical rows on both sides and overestimate performance by a factor close
   to 5. We train on the past and predict the following year.
2. **No country identifier.** The model receives the country's *position*, not its
   code: two neighbouring countries then look alike, instead of being two unrelated
   levels.
3. **A ranking in raw tonnes makes no sense across crops.** Potato yields 16 t/ha
   against 2.4 t/ha for wheat, but the gap is mostly water. The ranking is therefore
   given in **dry matter**, and per agronomic group. When producer prices are
   available (`src.data.prices`), the **gross product in USD/ha** is added: it answers
   the brief's economic question, within the limits stated in that module.

Runnable as a script to retrain and save the served model.
"""

from __future__ import annotations

import json
import lzma
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config
from src.data import prices as fao_prices
from src.data.crop_mapping import CROP_TO_FAO_ITEM
from src.models import evaluation, features, history, protocol

#: Validation years: each is predicted by a model trained on everything before it.
VALIDATION_YEARS = [2008, 2009, 2010, 2011, 2012]

#: Chosen representation: the country's past conditions + geography + year.
VARIABLES = features.YIELD_FEATURE_SETS["history"]


def load() -> pd.DataFrame:
    """Read the rebuilt FAO dataset, add the derived variables, and check for leakage.

    Rows without history — the first year of each country — and those with a missing
    variable fall out of scope: a model cannot learn from an incomplete row, and there
    is nothing to impute (see step 1).
    """
    raw = pd.read_csv(config.CROP_YIELD_PREDICTION_CSV)
    prepare = history.prepare(raw)
    history.check_no_leakage(prepare)
    return prepare.dropna(subset=VARIABLES + [features.YIELD_TARGET]).reset_index(drop=True)


#: Variables with no natural order: to one-hot encode if they are in the chosen set.
#: `iso3` only appears in the ablation sets of notebook 09, which measure its contribution
#: before replacing it with the geographic position.
CATEGORICAL = ("crop", "iso3")


def pipeline(model, variables: list[str] | None = None) -> Pipeline:
    """One-hot encode the categorical variables, leave the rest as is.

    `variables` specifies the set used; by default, only `crop` is encoded — the case
    of the served model. The encoding is **inside** the pipeline, hence relearned on
    each validation fold without seeing the evaluated fold.
    """
    present = [v for v in (variables or ["crop"]) if v in CATEGORICAL]
    preparation = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), present)],
        remainder="passthrough",
    )
    return Pipeline([("preparation", preparation), ("model", model)])


#: Settings of the served model, **validated by search** (notebook 09, section 5) and not
#: borrowed from elsewhere. 36 configurations sampled over three families in temporal
#: validation:
#:
#: * the best one found gains 0.0056 t/ha, i.e. **26 times less** than the variation from
#:   one validation year to the next (standard deviation 0.147 t/ha) — indistinguishable
#:   from noise;
#: * it only wins on 3 folds out of 5, below the chosen criterion;
#: * and above all it exceeds the 50 MB limit, because `min_samples_split=2` lets the trees
#:   grow deeper: 100.7 MB at 300 trees, 50.4 MB at 150.
#:
#: `min_samples_split=3` acts as a light regularisation that **also shortens the trees**:
#: hence both a smaller size and an equivalent performance. The delivery constraint and
#: performance point the same way here.
SETTINGS = {
    "max_features": 0.9,
    "min_samples_split": 3,
}


def selected_model(n_trees: int = 150) -> Pipeline:
    """The served model: ExtraTrees with the :data:`SETTINGS` settings.

    150 trees is the largest model that fits under the 50 MB limit; RMSE plateaus from
    150 anyway (notebook 10).

    `n_jobs=1` on purpose: the machine only has 4 GB free, and a forest parallelised
    over 12 cores duplicates the data that many times.
    """
    return pipeline(
        ExtraTreesRegressor(
            n_estimators=n_trees,
            random_state=config.SEED,
            n_jobs=1,
            **SETTINGS,
        )
    )


@dataclass
class Result:
    """What a full training run produces.

    `ranking` holds one set of ranking metrics per view (see :data:`VIEWS`).
    """

    model: Pipeline
    scores: dict[str, float]
    ranking: dict[str, dict[str, float]]
    n_train: int
    n_test: int


def train(df: pd.DataFrame | None = None, n_trees: int = 150, verbose: bool = True) -> Result:
    """Train on the years before 2013, evaluate once on 2013."""
    df = load() if df is None else df
    X_train, X_test, y_train, y_test = protocol.temporal_split(
        df, VARIABLES, features.YIELD_TARGET
    )

    model = selected_model(n_trees).fit(X_train, y_train)
    predictions = model.predict(X_test)

    scores = evaluation.metrics(y_test, predictions, "test")
    test = df[df["year"] == config.TEST_YEAR]
    prices = fao_prices.load() if config.PRODUCER_PRICES_CSV.exists() else None
    ranking = quality_by_view(test, pd.Series(predictions, index=test.index), prices)

    if verbose:
        print(f"apprentissage : {len(X_train):,} lignes ({df.year.min()}-{config.TEST_YEAR - 1})")
        print(f"test          : {len(X_test):,} lignes ({config.TEST_YEAR})")
        print()
        print(f"RMSE {scores['test_rmse']:.4f} t/ha | MAE {scores['test_mae']:.4f} "
              f"| R2 {scores['test_r2']:.4f}")
        print()
        print(views_table(ranking).round(3).to_string())

    return Result(model, scores, ranking, len(X_train), len(X_test))


# --------------------------------------------------------------------------- #
# Ranking quality, measured on what the user actually sees
# --------------------------------------------------------------------------- #

#: Ranking views. The raw view is the one notebook 09 measured; the others are those
#: the service can serve. Measuring only the raw view would grade a ranking nobody
#: sees: the served order is in dry matter, or in gross product.
VIEWS = {
    "raw": "10 cultures FAO, rendement brut (t/ha)",
    "dry_matter": "10 cultures FAO, matière sèche (t/ha)",
    "dry_matter_app": "4 cultures de l'application, matière sèche (t/ha)",
    "gross_product": "10 cultures FAO, produit brut (USD/ha)",
    "gross_product_app": "4 cultures de l'application, produit brut (USD/ha)",
}


def quality_by_view(
    observed: pd.DataFrame, predicted: pd.Series, prices: pd.DataFrame | None = None,
) -> dict[str, dict[str, float]]:
    """Ranking metrics for each view of :data:`VIEWS`.

    The same conversion is applied to observed and predicted yields: dry matter share,
    or reference price. Prices are those known at the row's year (no later price).
    The gross product views are skipped when `prix` is None.

    The `regret` of a view is in that view's unit: t/ha of dry matter, or USD/ha.
    """
    target = features.YIELD_TARGET
    ms = observed["crop"].map(features.DRY_MATTER)
    product = observed["crop"].isin(PRODUCT_TO_FAO.values()).to_numpy()

    conversions = {"raw": 1.0, "dry_matter": ms}
    if prices is not None:
        conversions["gross_product"] = fao_prices.price_per_row(prices, observed)

    results = {}
    for view, factor in conversions.items():
        converted = observed.assign(**{target: observed[target] * factor})
        predicted_converted = predicted * factor
        results[view] = evaluation.ranking_quality(converted, predicted_converted)
        if f"{view}_app" in VIEWS:
            results[f"{view}_app"] = evaluation.ranking_quality(
                converted[product], predicted_converted[product]
            )
    return results


def views_table(quality: dict[str, dict[str, float]]) -> pd.DataFrame:
    """The per-view metrics as a table, one row per view."""
    table = pd.DataFrame.from_dict(quality, orient="index")
    table.index = [VIEWS[v] for v in table.index]
    return table


def mlflow_metrics(quality: dict[str, dict[str, float]]) -> dict[str, float]:
    """Flatten the per-view metrics into MLflow metric names."""
    return {
        f"classement_{view}_{name}": value
        for view, measures in quality.items()
        for name, value in measures.items()
    }


# --------------------------------------------------------------------------- #
# Served ranking
# --------------------------------------------------------------------------- #

#: Reverse mapping: product crop -> FAO label, for the 4 shared crops.
PRODUCT_TO_FAO = {k: v for k, v in CROP_TO_FAO_ITEM.items() if v}

#: Product crops missing from the FAO dataset are **left out of the ranking**. Step 1
#: gave them a donor crop to enrich the consolidated dataset, but in a ranking a donor
#: does not hold: barley would always tie with wheat, whose prediction it copies, and
#: cotton would carry the yield of sorghum, which step 1 itself rules out as a proxy.
#: Both remain available through `/predict`.
EXCLUDED_CROPS = {
    crop: "absente du jeu FAO : son classement copierait celui de sa culture donneuse"
    for crop, item in CROP_TO_FAO_ITEM.items()
    if item is None
}


def country_context(df: pd.DataFrame, iso3: str, year: int | None = None) -> dict:
    """Latest known conditions of a country, to prefill in the interface."""
    of_country = df[df["iso3"] == iso3]
    if of_country.empty:
        raise KeyError(
            f"{iso3} n'est pas un pays servi : aucune ligne exploitable (voir les pays "
            "retires faute de temperature, de pesticides ou de pluviometrie fiable)"
        )
    year = int(of_country["year"].max()) if year is None else year
    rows = of_country[of_country["year"] == year]
    if rows.empty:
        raise KeyError(f"aucune donnee pour {iso3} en {year}")
    first = rows.iloc[0]
    return {
        "iso3": iso3,
        "year": int(year),
        "rain_mm": float(first["rain_mm"]),
        "temp_hist": float(first["temp_hist"]),
        "log_pest_hist": float(first["log_pest_hist"]),
        "lat_abs": float(first["lat_abs"]),
        "geo_x": float(first["geo_x"]),
        "geo_y": float(first["geo_y"]),
        "geo_z": float(first["geo_z"]),
    }


#: Columns the ranking can be sorted by.
CRITERIA = ("dry_matter_t_ha", "gross_product_usd_ha", "raw_yield_t_ha")


def predict_crops(model: Pipeline, context: dict, crops: dict[str, str]) -> pd.DataFrame:
    """Predicted yield of each crop in a context, raw and in dry matter, with its group.

    `crops` maps the displayed name to the FAO label the model knows. No price, no
    sort: the two building blocks the service assembles (see :func:`ranking`).
    """
    rows = pd.DataFrame(
        [{**context, "crop": fao} for fao in crops.values()]
    )[["crop"] + [v for v in VARIABLES if v != "crop"]]
    predictions = model.predict(rows)

    table = pd.DataFrame(
        {
            "crop": list(crops),
            "fao_crop": list(crops.values()),
            "raw_yield_t_ha": predictions,
        }
    )
    table["dry_matter_t_ha"] = [
        features.to_dry_matter(r, fao)
        for r, fao in zip(table["raw_yield_t_ha"], table["fao_crop"])
    ]
    table["group"] = table["fao_crop"].map(features.CROP_GROUP)
    return table


def add_gross_product(table: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Add price, price source and gross product, from an already resolved price table.

    `reference` is indexed by FAO label, with `price_usd_t` and `price_source` — as
    returned by :func:`src.data.prices.reference_prices`, or built by the API from the
    user's own prices.
    """
    missing = sorted(set(table["fao_crop"]) - set(reference.index))
    if missing:
        raise KeyError(f"aucun prix pour : {missing}")
    table = table.copy()
    table["price_usd_t"] = table["fao_crop"].map(reference["price_usd_t"])
    table["price_source"] = table["fao_crop"].map(reference["price_source"])
    table["gross_product_usd_ha"] = table["raw_yield_t_ha"] * table["price_usd_t"]
    return table


def ranking(
    model: Pipeline, context: dict, observed_crops: set[str] | None = None,
    product_crops: bool = True, prices: pd.DataFrame | None = None,
    criterion: str = "dry_matter_t_ha",
) -> pd.DataFrame:
    """Rank crops by expected yield, raw, in dry matter and, if prices are given, in
    gross product.

    Four precautions, which change what the user reads:

    * **dry matter** — a ranking in raw tonnes always puts a tuber first, most of the
      gap being water;
    * **agronomic group** — a potato is not pitted directly against wheat;
    * **`observed`** — a crop never grown in the country is flagged: validation barely
      measures this case, and the model extrapolates there;
    * **`price_source`** — the gross product uses the country's own price when it
      publishes one, the world median otherwise.

    `product_crops=True` restricts to the application's crops present in the FAO
    dataset; barley and cotton are left out (see :data:`EXCLUDED_CROPS`).
    `criterion` picks the sort column, among :data:`CRITERIA`.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"critere inconnu : {criterion!r}, attendu parmi {CRITERIA}")
    if criterion == "gross_product_usd_ha" and prices is None:
        raise ValueError("le classement en produit brut demande les prix")

    if product_crops:
        requested = dict(PRODUCT_TO_FAO)
    else:
        requested = {c: c for c in features.YIELD_CROPS}

    table = predict_crops(model, context, requested)
    if prices is not None:
        reference = fao_prices.reference_prices(
            prices, context["iso3"], context["year"], table["fao_crop"]
        )
        table = add_gross_product(table, reference)
    if observed_crops is not None:
        table["observed"] = table["fao_crop"].isin(observed_crops)
    return table.sort_values(criterion, ascending=False).reset_index(drop=True)


def crops_of_country(df: pd.DataFrame, iso3: str) -> set[str]:
    """Crops actually observed in the country, to flag extrapolations."""
    return set(df.loc[df["iso3"] == iso3, "crop"])


# --------------------------------------------------------------------------- #
# Size of the served model
# --------------------------------------------------------------------------- #

def compressed_size(model: Pipeline) -> float:
    """Model size in MB, lzma-compressed."""
    return len(lzma.compress(pickle.dumps(model), preset=6)) / 1024**2


def save(result: Result, df: pd.DataFrame) -> None:
    """Save the lzma-compressed model, and its metadata.

    `lzma` rather than joblib's `zlib`: a forest compresses much better, which keeps
    the file under GitHub's 50 MB per-file limit.
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with lzma.open(config.YIELD_MODEL_PATH, "wb", preset=6) as f:
        pickle.dump(result.model, f)

    metadata = {
        "service": "recommend",
        "model": "ExtraTreesRegressor",
        "n_estimators": result.model.named_steps["model"].n_estimators,
        "settings": SETTINGS,
        "settings_validated_by": (
            "recherche de 36 configurations sur 3 familles en validation temporelle "
            "(notebook 09) : le meilleur gain, 0,0056 t/ha, est 26 fois plus petit que la "
            "variation entre annees de validation, ne tient que sur 3 plis sur 5, et "
            "depasse la limite de 50 Mo"
        ),
        "variables": VARIABLES,
        "target": features.YIELD_TARGET,
        "unit": "t/ha",
        "seed": config.SEED,
        "protocol": {
            "type": "temporal",
            "validation_years": VALIDATION_YEARS,
            "test_year": config.TEST_YEAR,
        },
        "n_train": result.n_train,
        "n_test": result.n_test,
        "scores": result.scores,
        "ranking_quality": {
            view: {"description": VIEWS[view], **measures}
            for view, measures in result.ranking.items()
        },
        "fao_crops": features.YIELD_CROPS,
        "ranked_product_crops": PRODUCT_TO_FAO,
        "excluded_crops": EXCLUDED_CROPS,
        "prices": {
            "source": config.SOURCE_PRICES_URL,
            "element": "Producer Price (USD/tonne)",
            "rule": (
                f"mediane des {fao_prices.YEAR_WINDOW} dernieres annees connues du pays, "
                "sinon mediane mondiale ; jamais un prix posterieur a l'annee du contexte"
            ),
        },
        "served_countries": sorted(df["iso3"].unique()),
        "warnings": [
            "Le classement est un ordre de grandeur a l'echelle du contexte climatique d'un "
            "pays, pas une estimation de parcelle.",
            "Le produit brut (rendement x prix producteur FAOSTAT) n'est pas une marge : les "
            "couts de production sont inconnus. Il ne designe pas la culture la plus "
            "rentable, seulement celle qui se vend le plus cher par hectare.",
            "Une culture jamais observee dans le pays est signalee : la validation ne mesure "
            "presque pas ce cas.",
        ],
    }
    path = config.MODELS_DIR / "yield_model_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    df = load()
    result = train(df)
    save(result, df)
    size_mb = config.YIELD_MODEL_PATH.stat().st_size / 1024**2
    print(f"\necrit {config.YIELD_MODEL_PATH.relative_to(config.ROOT)} ({size_mb:.1f} Mo)")
    print("ecrit models/yield_model_metadata.json")
    if size_mb > 50:
        raise AssertionError(f"modele servi de {size_mb:.1f} Mo : au-dela de la limite de 50 Mo")


if __name__ == "__main__":
    main()
