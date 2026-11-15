"""Request and response models of the API.

Field names are in English, like the rest of the code; every label, description and
message a user reads is in French.
Bounds declared here are **physical** limits: a value outside them is impossible and
is rejected with a 422. Values that are possible but outside what the models have
learned are accepted, and flagged in `warnings` (see `api.service`).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

#: The 6 crops of the application, as named in the simulated dataset.
Crop = Literal["Maize", "Wheat", "Rice", "Soybean", "Barley", "Cotton"]

#: The crops `/recommend` ranks: those present in the FAO dataset.
RankedCrop = Literal["Maize", "Wheat", "Rice", "Soybean"]

#: Sort criteria of `/recommend`. `rendement` is the default required by the brief.
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
# /predict
# --------------------------------------------------------------------------- #


class PredictInput(BaseModel):
    """A plot: its crop, its conditions and its practices."""

    crop: Crop = Field(description="Culture semée (sans effet mesurable sur l'estimation).")
    rain_mm: float = Field(ge=0, le=5000, description="Pluie de la saison culturale, en mm.")
    temperature_c: float = Field(ge=-30, le=60, description="Température moyenne de la saison, en °C.")
    fertilizer: bool = Field(description="Engrais utilisé sur la parcelle.")
    irrigation: bool = Field(description="Parcelle irriguée.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"crop": "Wheat", "rain_mm": 600, "temperature_c": 22,
                 "fertilizer": True, "irrigation": False}
            ]
        }
    }


class PredictOutput(BaseModel):
    crop: Crop
    label: str
    yield_t_ha: float = Field(description="Rendement estimé, en tonnes par hectare.")
    interval_95_t_ha: tuple[float, float] = Field(
        description="95 % des parcelles comparables tombent dans cet intervalle."
    )
    warnings: list[str]


class PredictContext(BaseModel):
    """What the interface needs to build the `/predict` form."""

    crops: dict[Crop, str]
    physical_bounds: dict[str, dict[str, float]]
    learned_domain: dict[str, dict[str, float]]
    units: dict[str, str]


# --------------------------------------------------------------------------- #
# /recommend
# --------------------------------------------------------------------------- #


class RecommendInput(BaseModel):
    """A country, and optionally the conditions and prices that differ from its own."""

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


class UsedContext(BaseModel):
    iso3: str
    country: str
    year: int = Field(description="Année des conditions du pays : la dernière des données.")
    temperature_c: float
    rain_mm: float
    pesticides_t: float
    entered_values: list[str] = Field(description="Champs remplacés par la saisie de l'utilisateur.")


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


class UnrankedCrop(BaseModel):
    crop: Crop
    label: str
    reason: str


class RecommendOutput(BaseModel):
    context: UsedContext
    criterion: Criterion
    ranking: list[RankingRow]
    unranked_crops: list[UnrankedCrop]
    warnings: list[str]


class Country(BaseModel):
    iso3: str
    name: str


class ReferencePrice(BaseModel):
    crop: RankedCrop
    price_usd_t: float
    price_source: Literal["pays", "médiane mondiale"]


class CountryContext(BaseModel):
    """Default values of a country, to prefill the `/recommend` form."""

    iso3: str
    name: str
    year: int
    temperature_c: float
    rain_mm: float
    pesticides_t: float
    prices: list[ReferencePrice]
    grown_crops: list[RankedCrop]
