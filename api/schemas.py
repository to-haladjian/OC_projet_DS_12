"""Request and response models of the API.

Field names are in English, like the rest of the code; every label, description and
message a user reads is in French.

Bounds declared here are **physical** limits: a value outside them is impossible and
is rejected with a 422. Values that are possible but outside what the model has
learned are accepted, and flagged in `warnings` (see `api.service`).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

#: The 6 crops the product knows, as named in the simulated dataset.
Crop = Literal["Maize", "Wheat", "Rice", "Soybean", "Barley", "Cotton"]

#: The crops the yield model can estimate and rank: those present in the FAO dataset.
RankedCrop = Literal["Maize", "Wheat", "Rice", "Soybean"]

#: Sort criteria of `/recommend`. `yield` is the default the brief asks for.
Criterion = Literal["yield", "dry_matter", "gross_product"]

Price = Annotated[float, Field(gt=0, le=100_000)]

# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #


class Health(BaseModel):
    status: Literal["ok"]
    api_version: str
    models: dict[str, str]
    scikit_learn: str
    scikit_learn_training: str


# --------------------------------------------------------------------------- #
# Conditions: shared by /predict and /recommend
# --------------------------------------------------------------------------- #


class Conditions(BaseModel):
    """A country, and optionally the conditions that differ from its own.

    Both endpoints take exactly these fields and resolve them the same way: this is
    what guarantees that `/predict` and `/recommend` return the same yield for the same
    crop in the same conditions.
    """

    iso3: str = Field(min_length=3, max_length=3, description="Code ISO 3166-1 alpha-3 du pays.")
    temperature_c: float | None = Field(
        default=None, ge=-30, le=60,
        description="Température moyenne annuelle des dernières années, en °C. "
                    "Par défaut : la valeur du pays.",
    )
    rain_mm: float | None = Field(
        default=None, ge=0, le=5000,
        description="Pluie annuelle normale, en mm. Par défaut : la valeur du pays.",
    )
    pesticides_t: float | None = Field(
        default=None, ge=0, le=5_000_000,
        description="Pesticides utilisés dans le pays, en tonnes par an (moyenne des "
                    "dernières années). Par défaut : la valeur du pays.",
    )


class UsedContext(BaseModel):
    """The conditions the estimate was actually computed with."""

    iso3: str
    country: str
    year: int = Field(description="Année des conditions du pays : la dernière des données.")
    temperature_c: float
    rain_mm: float
    pesticides_t: float
    entered_values: list[str] = Field(description="Champs remplacés par la saisie de l'utilisateur.")


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #


class PredictInput(Conditions):
    """One crop, in a country, under conditions the user may adjust."""

    crop: Crop = Field(
        description="Culture à estimer. L'orge et le coton, absents des données FAO, sont refusés."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"iso3": "FRA", "crop": "Wheat"},
                {"iso3": "FRA", "crop": "Wheat", "temperature_c": 13.5, "rain_mm": 700},
            ]
        }
    }


class PracticeEffect(BaseModel):
    practice: Literal["fertilizer", "irrigation"]
    label: str
    effect_t_ha: float = Field(description="Gain moyen de rendement, mesuré sur les parcelles simulées.")


class PredictOutput(BaseModel):
    context: UsedContext
    crop: RankedCrop
    label: str
    yield_t_ha: float = Field(description="Rendement estimé, en tonnes par hectare.")
    interval_95_t_ha: tuple[float, float] = Field(
        description="Intervalle couvrant 95 % des erreurs observées en validation."
    )
    dry_matter_t_ha: float
    grown_in_country: bool
    practice_effects: list[PracticeEffect] = Field(
        description="Effets de l'engrais et de l'irrigation, issus des parcelles simulées : "
                    "un ordre de grandeur indicatif, non additionné à l'estimation."
    )
    warnings: list[str]


class UnrankedCrop(BaseModel):
    crop: Crop
    label: str
    reason: str


class PredictContext(BaseModel):
    """What the interface needs to build the `/predict` form."""

    crops: dict[RankedCrop, str]
    unavailable_crops: list[UnrankedCrop]
    physical_bounds: dict[str, dict[str, float]]
    units: dict[str, str]


# --------------------------------------------------------------------------- #
# /recommend
# --------------------------------------------------------------------------- #


class RecommendInput(Conditions):
    """A country, and optionally the conditions and prices that differ from its own."""

    price_usd_t: dict[RankedCrop, Price] = Field(
        default_factory=dict,
        description="Prix de vente saisis, en USD par tonne. Les cultures absentes "
                    "gardent le prix de référence FAOSTAT.",
    )
    criterion: Criterion = Field(default="yield", description="Critère de tri du classement.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"iso3": "FRA"},
                {"iso3": "FRA", "temperature_c": 13.5, "price_usd_t": {"Wheat": 260},
                 "criterion": "gross_product"},
            ]
        }
    }


class RankingRow(BaseModel):
    rank: int
    crop: RankedCrop
    label: str
    yield_t_ha: float
    dry_matter_t_ha: float
    group: str
    price_usd_t: float
    price_source: Literal["pays", "médiane mondiale", "saisi"]
    gross_product_usd_ha: float = Field(description="Rendement x prix. Ce n'est pas une marge.")
    grown_in_country: bool


class RecommendOutput(BaseModel):
    context: UsedContext
    criterion: Criterion
    ranking: list[RankingRow]
    unranked_crops: list[UnrankedCrop]
    warnings: list[str]


# --------------------------------------------------------------------------- #
# Countries
# --------------------------------------------------------------------------- #


class Country(BaseModel):
    iso3: str
    name: str


class ReferencePrice(BaseModel):
    crop: RankedCrop
    price_usd_t: float
    price_source: Literal["pays", "médiane mondiale"]


class CountryContext(BaseModel):
    """Default values of a country, to prefill both forms."""

    iso3: str
    name: str
    year: int
    temperature_c: float
    rain_mm: float
    pesticides_t: float
    prices: list[ReferencePrice]
    grown_crops: list[RankedCrop]
