"""Serving assets: everything the API needs besides the yield model, as small JSON files.

The API must not read `data/`: processed CSVs are not versioned, and the Docker image
should only carry what it serves. This module therefore freezes, next to the models:

* `country_contexts.json` — for each served country, the latest known conditions in the
  form the yield model expects, plus its name and the crops it actually grows;
* `reference_prices.json` — reference producer price of each served crop, per country,
  with its source, plus the world medians used as fallback;
* `serving_domain.json` — what lets the API qualify an answer: the usual range of each
  condition per crop, used to **warn** about an unusual input without rejecting it; the
  relative error observed in temporal validation, used to give each estimate an
  interval; and the scikit-learn version the model was trained with.

Runnable as a script, after the yield model has been trained:
`python -m src.models.serving_assets` (or `make assets`).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import sklearn

from src import config
from src.data import prices as fao_prices
from src.models import evaluation, features, protocol
from src.models import yield_model as ym

COUNTRY_CONTEXTS_JSON = config.MODELS_DIR / "country_contexts.json"
REFERENCE_PRICES_JSON = config.MODELS_DIR / "reference_prices.json"
SERVING_DOMAIN_JSON = config.MODELS_DIR / "serving_domain.json"

#: Variables of the FAO training domain the API checks, and the request field each
#: one corresponds to.
YIELD_DOMAIN = {"avg_temp": "temperature_c", "rain_mm": "rain_mm", "pesticides_t": "pesticides_t"}

#: Share of the validation errors an interval must cover.
INTERVAL_COVERAGE = 0.95


def country_contexts(df: pd.DataFrame) -> dict[str, dict]:
    """Latest context of each country, as :func:`yield_model.country_context` builds it."""
    names = df.drop_duplicates("iso3").set_index("iso3")["area"]
    result = {}
    for iso3 in sorted(df["iso3"].unique()):
        context = ym.country_context(df, iso3)
        latest = df[(df["iso3"] == iso3) & (df["year"] == context["year"])].iloc[0]
        result[iso3] = {
            "name": str(names[iso3]),
            **context,
            # Pesticides in tonnes, for display: the model sees their log.
            "pest_hist": float(latest["pest_hist"]),
            "observed_crops": sorted(ym.crops_of_country(df, iso3)),
        }
    return result


def reference_prices(contexts: dict[str, dict], prices: pd.DataFrame) -> dict:
    """Reference price of the served crops, per country, at the context year."""
    crops = list(ym.PRODUCT_TO_FAO.values())
    year = max(c["year"] for c in contexts.values())
    window = prices[prices["year"].between(year - fao_prices.YEAR_WINDOW + 1, year)]
    return {
        "unit": "USD/t",
        "source": config.SOURCE_PRICES_URL,
        "rule": (
            f"mediane des {fao_prices.YEAR_WINDOW} dernieres annees connues du pays, "
            "sinon mediane mondiale"
        ),
        "world_median": {
            crop: float(window.loc[window["crop"] == crop, "price_usd_t"].median())
            for crop in crops
        },
        "countries": {
            iso3: fao_prices.reference_prices(prices, iso3, context["year"], crops)
            .reset_index()
            .to_dict(orient="records")
            for iso3, context in contexts.items()
        },
    }


def relative_error_bounds(df: pd.DataFrame, coverage: float = INTERVAL_COVERAGE) -> dict[str, dict]:
    """Relative error of the served settings in temporal validation, per served crop.

    For each validation year, the yield is predicted by a model trained on the years
    before it; the error is then expressed as a share of the prediction. The quantiles
    of that share give the interval the API attaches to an estimate:
    `prediction x (1 + low)` to `prediction x (1 + high)`.

    Relative rather than absolute: a wheat yield ranges from 1 to 9 t/ha across
    countries, and an error of 1 t/ha does not mean the same thing at both ends. The
    test year takes no part in it.
    """
    before = df[df["year"] < config.TEST_YEAR]
    predicted = evaluation.out_of_fold_predictions(
        ym.selected_model(), before[ym.VARIABLES], before[features.YIELD_TARGET],
        protocol.temporal_cv(before["year"], ym.VALIDATION_YEARS),
    )
    evaluated = before.loc[predicted.index]
    relative = (evaluated[features.YIELD_TARGET] - predicted) / predicted
    tail = (1 - coverage) / 2

    bounds = {}
    for crop in ym.PRODUCT_TO_FAO.values():
        errors = relative[evaluated["crop"] == crop]
        bounds[crop] = {
            "low": float(errors.quantile(tail)),
            "high": float(errors.quantile(1 - tail)),
            "n": int(len(errors)),
        }
    return bounds


def serving_domain(df: pd.DataFrame) -> dict:
    """Usual ranges of the conditions, and the interval attached to an estimate."""
    fao_domain = pd.read_csv(config.TRAINING_DOMAIN_CSV)
    fao_domain = fao_domain[fao_domain["variable"].isin(YIELD_DOMAIN)]
    return {
        "scikit_learn": sklearn.__version__,
        "bounds": {
            "source": "quantiles 5 % / 95 % par culture FAO, annee de test exclue",
            "crops": {
                crop: {
                    YIELD_DOMAIN[row.variable]: {"min": float(row.q05), "max": float(row.q95)}
                    for row in block.itertuples()
                }
                for crop, block in fao_domain.groupby("crop")
            },
        },
        "relative_error": {
            "source": (
                f"erreur relative en validation temporelle ({ym.VALIDATION_YEARS[0]}-"
                f"{ym.VALIDATION_YEARS[-1]}), quantiles couvrant {INTERVAL_COVERAGE:.0%} des cas"
            ),
            "coverage": INTERVAL_COVERAGE,
            "crops": relative_error_bounds(df),
        },
    }


def _write(path, content) -> None:
    def default_value(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"non serialisable : {type(value)}")

    path.write_text(json.dumps(content, indent=1, ensure_ascii=False, default=default_value), encoding="utf-8")
    print(f"ecrit {path.relative_to(config.ROOT)} ({path.stat().st_size / 1024:.0f} Ko)")


def main() -> None:
    df = ym.load()
    contexts = country_contexts(df)
    _write(COUNTRY_CONTEXTS_JSON, contexts)
    _write(REFERENCE_PRICES_JSON, reference_prices(contexts, fao_prices.load()))
    _write(SERVING_DOMAIN_JSON, serving_domain(df))
    print(f"{len(contexts)} pays, annee de contexte {max(c['year'] for c in contexts.values())}")


if __name__ == "__main__":
    main()
