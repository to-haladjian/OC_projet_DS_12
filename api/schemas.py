"""Request and response models of the API.

Field names are in French, like the rest of the project and the interface they feed.
Bounds declared here are **physical** limits: a value outside them is impossible and
is rejected with a 422. Values that are possible but outside what the models have
learned are accepted, and flagged in `avertissements` (see `api.service`).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

#: The 6 crops of the application, as named in the simulated dataset.
Culture = Literal["Maize", "Wheat", "Rice", "Soybean", "Barley", "Cotton"]

#: The crops `/recommend` ranks: those present in the FAO dataset.
CultureClassee = Literal["Maize", "Wheat", "Rice", "Soybean"]

#: Sort criteria of `/recommend`. `rendement` is the default required by the brief.
Critere = Literal["rendement", "matiere_seche", "produit_brut"]

Prix = Annotated[float, Field(gt=0, le=100_000)]

# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #


class Sante(BaseModel):
    statut: Literal["ok"]
    version_api: str
    modeles: dict[str, str]
    scikit_learn: str
    scikit_learn_entrainement: str


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #


class EntreePredict(BaseModel):
    """A plot: its crop, its conditions and its practices."""

    culture: Culture = Field(description="Culture semée (sans effet mesurable sur l'estimation).")
    pluie_mm: float = Field(ge=0, le=5000, description="Pluie de la saison culturale, en mm.")
    temperature_c: float = Field(ge=-30, le=60, description="Température moyenne de la saison, en °C.")
    engrais: bool = Field(description="Engrais utilisé sur la parcelle.")
    irrigation: bool = Field(description="Parcelle irriguée.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"culture": "Wheat", "pluie_mm": 600, "temperature_c": 22,
                 "engrais": True, "irrigation": False}
            ]
        }
    }


class SortiePredict(BaseModel):
    culture: Culture
    libelle: str
    rendement_t_ha: float = Field(description="Rendement estimé, en tonnes par hectare.")
    intervalle_95_t_ha: tuple[float, float] = Field(
        description="95 % des parcelles comparables tombent dans cet intervalle."
    )
    avertissements: list[str]


class ContextePredict(BaseModel):
    """What the interface needs to build the `/predict` form."""

    cultures: dict[Culture, str]
    bornes_physiques: dict[str, dict[str, float]]
    domaine_entrainement: dict[str, dict[str, float]]
    unites: dict[str, str]


# --------------------------------------------------------------------------- #
# /recommend
# --------------------------------------------------------------------------- #


class EntreeRecommend(BaseModel):
    """A country, and optionally the conditions and prices that differ from its own."""

    iso3: str = Field(min_length=3, max_length=3, description="Code ISO 3166-1 alpha-3 du pays.")
    temperature_c: float | None = Field(
        default=None, ge=-30, le=60,
        description="Température moyenne annuelle des dernières années, en °C. "
                    "Par défaut : la valeur du pays.",
    )
    pluie_mm: float | None = Field(
        default=None, ge=0, le=5000,
        description="Pluie annuelle normale, en mm. Par défaut : la valeur du pays.",
    )
    pesticides_t: float | None = Field(
        default=None, ge=0, le=5_000_000,
        description="Pesticides utilisés dans le pays, en tonnes par an (moyenne des "
                    "dernières années). Par défaut : la valeur du pays.",
    )
    prix_usd_t: dict[CultureClassee, Prix] = Field(
        default_factory=dict,
        description="Prix de vente saisis, en USD par tonne. Les cultures absentes "
                    "gardent le prix de référence FAOSTAT.",
    )
    critere: Critere = Field(default="rendement", description="Critère de tri du classement.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"iso3": "FRA"},
                {"iso3": "FRA", "temperature_c": 13.5, "prix_usd_t": {"Wheat": 260},
                 "critere": "produit_brut"},
            ]
        }
    }


class ContexteUtilise(BaseModel):
    iso3: str
    pays: str
    annee: int = Field(description="Année des conditions du pays : la dernière des données.")
    temperature_c: float
    pluie_mm: float
    pesticides_t: float
    valeurs_saisies: list[str] = Field(description="Champs remplacés par la saisie de l'utilisateur.")


class LigneClassement(BaseModel):
    rang: int
    culture: CultureClassee
    libelle: str
    rendement_t_ha: float
    matiere_seche_t_ha: float
    groupe: str
    prix_usd_t: float
    prix_source: Literal["pays", "médiane mondiale", "saisi"]
    produit_brut_usd_ha: float = Field(description="Rendement x prix. Ce n'est pas une marge.")
    cultivee_dans_le_pays: bool


class CultureNonClassee(BaseModel):
    culture: Culture
    libelle: str
    motif: str


class SortieRecommend(BaseModel):
    contexte: ContexteUtilise
    critere: Critere
    classement: list[LigneClassement]
    cultures_non_classees: list[CultureNonClassee]
    avertissements: list[str]


class Pays(BaseModel):
    iso3: str
    nom: str


class PrixReference(BaseModel):
    culture: CultureClassee
    prix_usd_t: float
    prix_source: Literal["pays", "médiane mondiale"]


class ContextePays(BaseModel):
    """Default values of a country, to prefill the `/recommend` form."""

    iso3: str
    nom: str
    annee: int
    temperature_c: float
    pluie_mm: float
    pesticides_t: float
    prix: list[PrixReference]
    cultures_pratiquees: list[CultureClassee]
