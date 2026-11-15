"""Serving assets: everything the API needs besides the two models, as small JSON files.

The API must not read `data/`: processed CSVs are not versioned, and the Docker image
should only carry what it serves. This module therefore freezes, next to the models:

* `country_contexts.json` — for each served country, the latest known conditions in the
  form the `/recommend` model expects, plus its name and the crops it actually grows;
* `reference_prices.json` — reference producer price of each ranked crop, per country,
  with its source, plus the world medians used as fallback;
* `serving_domain.json` — training-domain bounds of both services, used to **warn**
  about an unusual input without rejecting it, and the scikit-learn version the models
  were trained with.

Runnable as a script, after both models have been trained:
`python -m src.models.serving_assets`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import sklearn

from src import config
from src.data import prices as fao_prices
from src.models import features
from src.models import yield_model as ym

COUNTRY_CONTEXTS_JSON = config.MODELS_DIR / "country_contexts.json"
REFERENCE_PRICES_JSON = config.MODELS_DIR / "reference_prices.json"
SERVING_DOMAIN_JSON = config.MODELS_DIR / "serving_domain.json"

#: Variables of the FAO training domain the API checks, and the request field each
#: one corresponds to.
YIELD_DOMAIN = {"avg_temp": "temperature_c", "rain_mm": "rain_mm", "pesticides_t": "pesticides_t"}


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
    """Reference price of the ranked crops, per country, at the context year."""
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


def serving_domain() -> dict:
    """Training-domain bounds of both services."""
    # The consolidated dataset exists in two forms: the plain CSV, regenerated locally,
    # and the versioned .csv.gz. A fresh clone only has the latter, hence the resolved
    # path rather than a hard-coded one.
    simulated = pd.read_csv(
        config.CONSOLIDATED_DATASET_CSV
        if config.CONSOLIDATED_DATASET_CSV.exists()
        else config.CONSOLIDATED_DATASET_GZ,
        usecols=features.PRACTICE_SELECTED[:2],
    )
    fao_domain = pd.read_csv(config.TRAINING_DOMAIN_CSV)
    fao_domain = fao_domain[fao_domain["variable"].isin(YIELD_DOMAIN)]
    return {
        "scikit_learn": sklearn.__version__,
        "predict": {
            "source": "min / max du jeu simule d'entrainement",
            "bounds": {
                column: {"min": float(simulated[column].min()), "max": float(simulated[column].max())}
                for column in simulated.columns
            },
        },
        "recommend": {
            "source": "quantiles 5 % / 95 % par culture FAO, annee de test exclue",
            "bounds": {
                crop: {
                    YIELD_DOMAIN[row.variable]: {"min": float(row.q05), "max": float(row.q95)}
                    for row in block.itertuples()
                }
                for crop, block in fao_domain.groupby("crop")
            },
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
    _write(SERVING_DOMAIN_JSON, serving_domain())
    print(f"{len(contexts)} pays, annee de contexte {max(c['year'] for c in contexts.values())}")


if __name__ == "__main__":
    main()
