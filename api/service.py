"""Serving logic, kept apart from HTTP so that it can be tested directly.

Everything is loaded once, at startup, from `models/` only: the yield model, the
metadata of both models and the serving assets written by `src.models.serving_assets`.
The API never reads `data/`.

**One engine, two questions.** `/predict` and `/recommend` both go through
:func:`_estimate`: the same conditions, resolved the same way, fed to the same model for
the same four crops. `/predict` returns the row of the crop asked for; `/recommend`
returns all four, sorted. They cannot disagree.

One rule runs through this module: **an unusual input is answered, with a warning; an
impossible one is rejected.** Impossible values are caught by the bounds of
`api.schemas` (422). Values outside the training domain are accepted, because leaving
the domain is not an error — it is a zone where the model has learned little, and the
user must be told so.
"""

from __future__ import annotations

import json
import logging
import lzma
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from api.schemas import Conditions, PredictInput, RecommendInput
from src import config
from src.data.crop_mapping import CROP_TO_FAO_ITEM
from src.models import yield_model as ym

logger = logging.getLogger("agritech.api")

#: Display names of the crops.
LABELS = {
    "Maize": "Maïs", "Wheat": "Blé", "Rice": "Riz",
    "Soybean": "Soja", "Barley": "Orge", "Cotton": "Coton",
}

#: Request criterion -> column of the ranking table.
CRITERION_COLUMN = {
    "yield": "raw_yield_t_ha",
    "dry_matter": "dry_matter_t_ha",
    "gross_product": "gross_product_usd_ha",
}

FAO_TO_PRODUCT = {fao: crop for crop, fao in ym.PRODUCT_TO_FAO.items()}

SOURCE_ENTERED = "saisi"

#: Practices measured on the simulated plots: coefficient name in the practice model
#: metadata -> (request-side name, display label).
PRACTICES = {
    "Fertilizer_Used": ("fertilizer", "Engrais"),
    "Irrigation_Used": ("irrigation", "Irrigation"),
}

#: Display name and unit of each condition, for the warnings.
CONDITION_LABELS = {
    "temperature_c": ("température", "°C"),
    "rain_mm": ("pluie", "mm"),
    "pesticides_t": ("pesticides", "t"),
}


class UnknownCountry(KeyError):
    """The requested country is not served."""


class UnavailableCrop(ValueError):
    """The requested crop is known to the product but absent from the FAO data."""


@dataclass
class Resources:
    """Everything the API serves, loaded once."""

    yield_model: Pipeline
    yield_meta: dict
    practice_meta: dict
    contexts: dict[str, dict]
    prices: dict
    domain: dict


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_resources(folder: Path = config.MODELS_DIR) -> Resources:
    """Load the model and the assets. Fails loudly if a file is missing."""
    with lzma.open(folder / config.YIELD_MODEL_PATH.name, "rb") as f:
        yield_model = pickle.load(f)
    resources = Resources(
        yield_model=yield_model,
        yield_meta=_read_json(folder / "yield_model_metadata.json"),
        practice_meta=_read_json(folder / "practice_model_metadata.json"),
        contexts=_read_json(folder / "country_contexts.json"),
        prices=_read_json(folder / "reference_prices.json"),
        domain=_read_json(folder / "serving_domain.json"),
    )
    trained_with = resources.domain["scikit_learn"]
    if trained_with != sklearn.__version__:
        logger.warning(
            "scikit-learn %s ici, %s a l'entrainement : le modele peut mal se charger",
            sklearn.__version__, trained_with,
        )
    return resources


# --------------------------------------------------------------------------- #
# Countries
# --------------------------------------------------------------------------- #

def stored_country_context(resources: Resources, iso3: str) -> dict:
    """Stored context of a served country."""
    code = iso3.upper()
    if code not in resources.contexts:
        raise UnknownCountry(f"{code} n'est pas un pays servi. Liste des pays : GET /countries")
    return resources.contexts[code]


def country_prices(resources: Resources, iso3: str) -> pd.DataFrame:
    """Reference prices of the served crops, indexed by FAO label."""
    return pd.DataFrame(resources.prices["countries"][iso3.upper()]).set_index("crop")


def served_countries(resources: Resources) -> list[dict]:
    return sorted(
        ({"iso3": code, "name": c["name"]} for code, c in resources.contexts.items()),
        key=lambda p: p["name"],
    )


def display_country_context(resources: Resources, iso3: str) -> dict:
    """A country's default values, in the units of the request."""
    base = stored_country_context(resources, iso3)
    prices = country_prices(resources, iso3)
    return {
        "iso3": base["iso3"],
        "name": base["name"],
        "year": base["year"],
        "temperature_c": round(base["temp_hist"], 2),
        "rain_mm": base["rain_mm"],
        "pesticides_t": round(base["pest_hist"], 1),
        "prices": [
            {"crop": FAO_TO_PRODUCT[fao], "price_usd_t": row.price_usd_t,
             "price_source": row.price_source}
            for fao, row in prices.iterrows()
        ],
        "grown_crops": [
            FAO_TO_PRODUCT[fao] for fao in base["observed_crops"] if fao in FAO_TO_PRODUCT
        ],
    }


# --------------------------------------------------------------------------- #
# The shared estimate
# --------------------------------------------------------------------------- #

@dataclass
class Estimate:
    """Predicted yield of the four served crops, and the conditions it was computed with."""

    table: pd.DataFrame
    used: dict
    base: dict


def _estimate(resources: Resources, conditions: Conditions) -> Estimate:
    """Resolve the conditions and predict the four served crops.

    The country gives the default conditions; whatever the user entered replaces them.
    Both endpoints call this function and nothing else to obtain a yield.
    """
    base = stored_country_context(resources, conditions.iso3)
    context = {v: base[v] for v in ym.VARIABLES if v != "crop"} | {"iso3": base["iso3"]}

    entered = []
    if conditions.temperature_c is not None:
        context["temp_hist"] = conditions.temperature_c
        entered.append("temperature_c")
    if conditions.rain_mm is not None:
        context["rain_mm"] = conditions.rain_mm
        entered.append("rain_mm")
    pesticides = base["pest_hist"]
    if conditions.pesticides_t is not None:
        pesticides = conditions.pesticides_t
        context["log_pest_hist"] = float(np.log1p(pesticides))
        entered.append("pesticides_t")

    table = ym.predict_crops(resources.yield_model, context, ym.PRODUCT_TO_FAO)
    table["grown"] = table["fao_crop"].isin(base["observed_crops"])
    used = {
        "iso3": base["iso3"], "country": base["name"], "year": base["year"],
        "temperature_c": round(context["temp_hist"], 2), "rain_mm": context["rain_mm"],
        "pesticides_t": round(pesticides, 1), "entered_values": entered,
    }
    return Estimate(table, used, base)


def _scale_warning(base: dict) -> str:
    return (
        f"Conditions à l'échelle du pays ({base['name']}, données {base['year']}), pas de la "
        "parcelle : l'estimation est un ordre de grandeur."
    )


def _domain_warnings(resources: Resources, used: dict, crops: dict[str, str]) -> list[str]:
    """One warning per condition, listing the crops for which the value is unusual.

    `crops` maps the product names concerned to their FAO labels.
    """
    bounds = resources.domain["bounds"]["crops"]
    messages = []
    for field_name, (name, unit) in CONDITION_LABELS.items():
        value = used[field_name]
        affected = [
            LABELS[product]
            for product, fao in crops.items()
            if not bounds[fao][field_name]["min"] <= value <= bounds[fao][field_name]["max"]
        ]
        if affected:
            messages.append(
                f"{name} = {value:g} {unit} : valeur rare dans les données pour "
                f"{', '.join(affected)} (hors des 90 % centraux) — estimation moins sûre."
            )
    return messages


def _not_grown_warning(labels: list[str]) -> str:
    return (
        f"Jamais cultivé dans ce pays : {', '.join(labels)}. "
        "Le modèle extrapole, et la validation ne mesure pas ce cas."
    )


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #

def unavailable_crops() -> list[dict]:
    """Product crops the FAO data cannot estimate, with the reason."""
    return [
        {"crop": crop, "label": LABELS[crop], "reason": reason}
        for crop, reason in ym.EXCLUDED_CROPS.items()
    ]


def practice_effects(resources: Resources) -> list[dict]:
    """Mean effect of each practice, as measured on the simulated plots."""
    coefficients = resources.practice_meta["coefficients"]
    return [
        {"practice": practice, "label": label, "effect_t_ha": round(coefficients[name], 2)}
        for name, (practice, label) in PRACTICES.items()
    ]


def predict(resources: Resources, payload: PredictInput) -> dict:
    """Expected yield of one crop, with its interval, from the shared estimate."""
    if payload.crop in ym.EXCLUDED_CROPS:
        raise UnavailableCrop(
            f"{LABELS[payload.crop]} : {ym.EXCLUDED_CROPS[payload.crop]}. "
            f"Cultures disponibles : {', '.join(LABELS[c] for c in ym.PRODUCT_TO_FAO)}."
        )
    estimate = _estimate(resources, payload)
    row = estimate.table.set_index("crop").loc[payload.crop]
    predicted = float(row.raw_yield_t_ha)

    error = resources.domain["relative_error"]["crops"][row.fao_crop]
    interval = (round(predicted * (1 + error["low"]), 3), round(predicted * (1 + error["high"]), 3))

    warnings = [
        _scale_warning(estimate.base),
        *_domain_warnings(resources, estimate.used, {payload.crop: row.fao_crop}),
    ]
    if not row.grown:
        warnings.append(_not_grown_warning([LABELS[payload.crop]]))

    return {
        "context": estimate.used,
        "crop": payload.crop,
        "label": LABELS[payload.crop],
        "yield_t_ha": round(predicted, 3),
        "interval_95_t_ha": interval,
        "dry_matter_t_ha": round(float(row.dry_matter_t_ha), 3),
        "grown_in_country": bool(row.grown),
        "practice_effects": practice_effects(resources),
        "warnings": warnings,
    }


def predict_context() -> dict:
    """Crops, bounds and units, for the interface."""
    return {
        "crops": {crop: LABELS[crop] for crop in ym.PRODUCT_TO_FAO},
        "unavailable_crops": unavailable_crops(),
        "physical_bounds": {
            "temperature_c": {"min": -30, "max": 60},
            "rain_mm": {"min": 0, "max": 5000},
            "pesticides_t": {"min": 0, "max": 5_000_000},
        },
        "units": {"temperature_c": "°C", "rain_mm": "mm", "pesticides_t": "t",
                  "yield_t_ha": "t/ha"},
    }


# --------------------------------------------------------------------------- #
# /recommend
# --------------------------------------------------------------------------- #

def recommend(resources: Resources, payload: RecommendInput) -> dict:
    """Rank the crops for a country, with the user's conditions and prices if given."""
    estimate = _estimate(resources, payload)
    base, used = estimate.base, dict(estimate.used)

    reference = country_prices(resources, base["iso3"])
    for product, price in payload.price_usd_t.items():
        reference.loc[ym.PRODUCT_TO_FAO[product]] = [price, SOURCE_ENTERED]
    if payload.price_usd_t:
        used["entered_values"] = [*used["entered_values"], "price_usd_t"]

    table = ym.add_gross_product(estimate.table, reference)
    table = table.sort_values(CRITERION_COLUMN[payload.criterion], ascending=False).reset_index(drop=True)

    warnings = [
        _scale_warning(base),
        *_domain_warnings(resources, used, ym.PRODUCT_TO_FAO),
    ]
    not_grown = [LABELS[c] for c in table.loc[~table["grown"], "crop"]]
    if not_grown:
        warnings.append(_not_grown_warning(not_grown))
    world_priced = [LABELS[c] for c in table.loc[table["price_source"] == "médiane mondiale", "crop"]]
    if world_priced:
        warnings.append(
            f"Pas de prix national pour {', '.join(world_priced)} : médiane mondiale FAOSTAT utilisée."
        )
    if payload.criterion == "gross_product":
        warnings.append(
            "Le produit brut (rendement x prix) n'est pas une marge : les coûts de production "
            "sont inconnus."
        )

    return {
        "context": used,
        "criterion": payload.criterion,
        "ranking": [
            {
                "rank": rank,
                "crop": row.crop,
                "label": LABELS[row.crop],
                "yield_t_ha": round(row.raw_yield_t_ha, 3),
                "dry_matter_t_ha": round(row.dry_matter_t_ha, 3),
                "group": row.group,
                "price_usd_t": round(float(row.price_usd_t), 1),
                "price_source": row.price_source,
                "gross_product_usd_ha": round(float(row.gross_product_usd_ha), 1),
                "grown_in_country": bool(row.grown),
            }
            for rank, row in enumerate(table.itertuples(), start=1)
        ],
        "unranked_crops": unavailable_crops(),
        "warnings": warnings,
    }


def _check() -> None:
    """Every crop of the application has a display name."""
    missing = set(CROP_TO_FAO_ITEM) - set(LABELS)
    if missing:
        raise ValueError(f"libelles manquants : {sorted(missing)}")


_check()
