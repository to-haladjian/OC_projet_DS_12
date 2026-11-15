"""Serving logic, kept apart from HTTP so that it can be tested directly.

Everything is loaded once, at startup, from `models/` only: the two models, their
metadata and the serving assets written by `src.models.serving_assets`. The API never
reads `data/`.

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

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from api.schemas import PredictInput, RecommendInput
from src import config
from src.data.crop_mapping import CROP_TO_FAO_ITEM
from src.models import features
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

#: Two-sided 95 % quantile of the Gaussian noise of `/predict`.
Z_95 = 1.96

SOURCE_ENTERED = "saisi"

#: Shown with every `/predict` answer: the simulated training data carry no crop effect.
CROP_WARNING = (
    "La culture choisie ne change pas l'estimation : dans les données d'entraînement, le "
    "rendement dépend des conditions et des pratiques de la parcelle, pas de la culture."
)


class UnknownCountry(KeyError):
    """The requested country is not served."""


@dataclass
class Resources:
    """Everything the API serves, loaded once."""

    predict: Pipeline
    recommend: Pipeline
    practice_meta: dict
    yield_meta: dict
    contexts: dict[str, dict]
    prices: dict
    domain: dict


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_resources(folder: Path = config.MODELS_DIR) -> Resources:
    """Load models and assets. Fails loudly if a file is missing."""
    with lzma.open(folder / config.YIELD_MODEL_PATH.name, "rb") as f:
        recommend = pickle.load(f)
    resources = Resources(
        predict=joblib.load(folder / config.PRACTICE_MODEL_PATH.name),
        recommend=recommend,
        practice_meta=_read_json(folder / "practice_model_metadata.json"),
        yield_meta=_read_json(folder / "yield_model_metadata.json"),
        contexts=_read_json(folder / "country_contexts.json"),
        prices=_read_json(folder / "reference_prices.json"),
        domain=_read_json(folder / "serving_domain.json"),
    )
    trained_with = resources.domain["scikit_learn"]
    if trained_with != sklearn.__version__:
        logger.warning(
            "scikit-learn %s ici, %s a l'entrainement : les modeles peuvent mal se charger",
            sklearn.__version__, trained_with,
        )
    return resources


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #

def _out_of_domain(name: str, value: float, bounds: dict[str, float], unit: str) -> str | None:
    if bounds["min"] <= value <= bounds["max"]:
        return None
    return (
        f"{name} = {value:g} {unit} est hors du domaine appris "
        f"({bounds['min']:.0f} à {bounds['max']:.0f} {unit}) : estimation extrapolée."
    )


def predict_yield(resources: Resources, payload: PredictInput) -> dict:
    """Yield estimate for one plot, with a 95 % interval and warnings."""
    row = pd.DataFrame([{
        "Rainfall_mm": payload.rain_mm,
        "Temperature_Celsius": payload.temperature_c,
        "Fertilizer_Used": float(payload.fertilizer),
        "Irrigation_Used": float(payload.irrigation),
    }])[features.PRACTICE_SELECTED]
    yield_value = float(resources.predict.predict(row)[0])

    bounds = resources.domain["predict"]["bounds"]
    warnings = [CROP_WARNING]
    for message in (
        _out_of_domain("pluie", payload.rain_mm, bounds["Rainfall_mm"], "mm"),
        _out_of_domain("température", payload.temperature_c, bounds["Temperature_Celsius"], "°C"),
    ):
        if message:
            warnings.append(message)
    if yield_value < 0:
        warnings.append(
            "L'estimation brute est négative, ramenée à 0 : ces conditions sont hors de ce "
            "que le modèle sait décrire."
        )
        yield_value = 0.0

    half_width = Z_95 * resources.practice_meta["noise_sigma"]
    return {
        "crop": payload.crop,
        "label": LABELS[payload.crop],
        "yield_t_ha": round(yield_value, 3),
        "interval_95_t_ha": (round(max(yield_value - half_width, 0.0), 3),
                               round(yield_value + half_width, 3)),
        "warnings": warnings,
    }


def predict_context(resources: Resources) -> dict:
    """Crops, bounds and units, for the interface."""
    domain = resources.domain["predict"]["bounds"]
    return {
        "crops": LABELS,
        "physical_bounds": {"rain_mm": {"min": 0, "max": 5000},
                             "temperature_c": {"min": -30, "max": 60}},
        "learned_domain": {"rain_mm": domain["Rainfall_mm"],
                                 "temperature_c": domain["Temperature_Celsius"]},
        "units": {"rain_mm": "mm", "temperature_c": "°C", "yield_t_ha": "t/ha"},
    }


# --------------------------------------------------------------------------- #
# /recommend
# --------------------------------------------------------------------------- #

def stored_country_context(resources: Resources, iso3: str) -> dict:
    """Stored context of a served country."""
    code = iso3.upper()
    if code not in resources.contexts:
        raise UnknownCountry(
            f"{code} n'est pas un pays servi. Liste des pays : GET /recommend/countries"
        )
    return resources.contexts[code]


def country_prices(resources: Resources, iso3: str) -> pd.DataFrame:
    """Reference prices of the ranked crops, indexed by FAO label."""
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


def _domain_warnings(resources: Resources, used: dict) -> list[str]:
    """One warning per variable, listing the crops for which the value is unusual."""
    bounds = resources.domain["recommend"]["bounds"]
    labels = {"temperature_c": ("température", "°C"), "rain_mm": ("pluie", "mm"),
                "pesticides_t": ("pesticides", "t")}
    messages = []
    for field_name, (name, unit) in labels.items():
        value = used[field_name]
        affected = [
            LABELS[product]
            for product, fao in ym.PRODUCT_TO_FAO.items()
            if not bounds[fao][field_name]["min"] <= value <= bounds[fao][field_name]["max"]
        ]
        if affected:
            messages.append(
                f"{name} = {value:g} {unit} : valeur rare dans les données pour "
                f"{', '.join(affected)} (hors des 90 % centraux) — classement moins sûr."
            )
    return messages


def recommend(resources: Resources, payload: RecommendInput) -> dict:
    """Rank the crops for a country, with the user's conditions and prices if given."""
    base = stored_country_context(resources, payload.iso3)
    context = {v: base[v] for v in ym.VARIABLES if v != "crop"} | {"iso3": base["iso3"]}

    entered_fields = []
    if payload.temperature_c is not None:
        context["temp_hist"] = payload.temperature_c
        entered_fields.append("temperature_c")
    if payload.rain_mm is not None:
        context["rain_mm"] = payload.rain_mm
        entered_fields.append("rain_mm")
    pesticides = base["pest_hist"]
    if payload.pesticides_t is not None:
        pesticides = payload.pesticides_t
        context["log_pest_hist"] = float(np.log1p(pesticides))
        entered_fields.append("pesticides_t")

    reference = country_prices(resources, base["iso3"])
    for product, prices in payload.price_usd_t.items():
        reference.loc[ym.PRODUCT_TO_FAO[product]] = [prices, SOURCE_ENTERED]
    if payload.price_usd_t:
        entered_fields.append("price_usd_t")

    table = ym.predict_crops(resources.recommend, context, ym.PRODUCT_TO_FAO)
    table = ym.add_gross_product(table, reference)
    table["grown"] = table["fao_crop"].isin(base["observed_crops"])
    table = table.sort_values(CRITERION_COLUMN[payload.criterion], ascending=False).reset_index(drop=True)

    used = {
        "iso3": base["iso3"], "country": base["name"], "year": base["year"],
        "temperature_c": round(context["temp_hist"], 2), "rain_mm": context["rain_mm"],
        "pesticides_t": round(pesticides, 1), "entered_values": entered_fields,
    }

    warnings = [
        f"Conditions à l'échelle du pays ({base['name']}, données {base['year']}), pas de la "
        "parcelle : le classement est un ordre de grandeur.",
        *_domain_warnings(resources, used),
    ]
    not_grown = [LABELS[c] for c in table.loc[~table["grown"], "crop"]]
    if not_grown:
        warnings.append(
            f"Jamais cultivé dans ce pays : {', '.join(not_grown)}. "
            "Le modèle extrapole, et la validation ne mesure pas ce cas."
        )
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
        "unranked_crops": [
            {"crop": crop, "label": LABELS[crop], "reason": reason}
            for crop, reason in ym.EXCLUDED_CROPS.items()
        ],
        "warnings": warnings,
    }


def _check() -> None:
    """Every crop of the application has a display name."""
    missing = set(CROP_TO_FAO_ITEM) - set(LABELS)
    if missing:
        raise ValueError(f"libelles manquants : {sorted(missing)}")


_check()
