"""Serving logic, kept apart from HTTP so that it can be tested directly.

Everything is loaded once, at startup, from `models/` only: the two models, their
metadata and the serving assets written by `src.models.actifs_service`. The API never
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

from api.schemas import EntreePredict, EntreeRecommend
from src import config
from src.data.crop_mapping import CROP_TO_FAO_ITEM
from src.models import features
from src.models import recommend_model as rm

journal = logging.getLogger("agritech.api")

#: Display names of the crops.
LIBELLES = {
    "Maize": "Maïs", "Wheat": "Blé", "Rice": "Riz",
    "Soybean": "Soja", "Barley": "Orge", "Cotton": "Coton",
}

#: Request criterion -> column of the ranking table.
COLONNE_CRITERE = {
    "rendement": "rendement_brut_t_ha",
    "matiere_seche": "matiere_seche_t_ha",
    "produit_brut": "produit_brut_usd_ha",
}

FAO_VERS_PRODUIT = {fao: culture for culture, fao in rm.PRODUIT_VERS_FAO.items()}

#: Two-sided 95 % quantile of the Gaussian noise of `/predict`.
Z_95 = 1.96

SOURCE_SAISIE = "saisi"

#: Shown with every `/predict` answer: the simulated training data carry no crop effect.
AVERTISSEMENT_CULTURE = (
    "La culture choisie ne change pas l'estimation : dans les données d'entraînement, le "
    "rendement dépend des conditions et des pratiques de la parcelle, pas de la culture."
)


class PaysInconnu(KeyError):
    """The requested country is not served."""


@dataclass
class Ressources:
    """Everything the API serves, loaded once."""

    predict: Pipeline
    recommend: Pipeline
    meta_predict: dict
    meta_recommend: dict
    contextes: dict[str, dict]
    prix: dict
    domaine: dict


def _lit_json(chemin: Path) -> dict:
    return json.loads(chemin.read_text(encoding="utf-8"))


def charge_ressources(dossier: Path = config.MODELES) -> Ressources:
    """Load models and assets. Fails loudly if a file is missing."""
    with lzma.open(dossier / config.MODELE_RECOMMEND.name, "rb") as f:
        recommend = pickle.load(f)
    ressources = Ressources(
        predict=joblib.load(dossier / config.MODELE_PREDICT.name),
        recommend=recommend,
        meta_predict=_lit_json(dossier / "predict_model_metadata.json"),
        meta_recommend=_lit_json(dossier / "recommend_model_metadata.json"),
        contextes=_lit_json(dossier / "contextes_pays.json"),
        prix=_lit_json(dossier / "prix_reference.json"),
        domaine=_lit_json(dossier / "domaine_service.json"),
    )
    entrainement = ressources.domaine["scikit_learn"]
    if entrainement != sklearn.__version__:
        journal.warning(
            "scikit-learn %s ici, %s a l'entrainement : les modeles peuvent mal se charger",
            sklearn.__version__, entrainement,
        )
    return ressources


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #

def _hors_domaine(nom: str, valeur: float, bornes: dict[str, float], unite: str) -> str | None:
    if bornes["min"] <= valeur <= bornes["max"]:
        return None
    return (
        f"{nom} = {valeur:g} {unite} est hors du domaine appris "
        f"({bornes['min']:.0f} à {bornes['max']:.0f} {unite}) : estimation extrapolée."
    )


def predit(ressources: Ressources, entree: EntreePredict) -> dict:
    """Yield estimate for one plot, with a 95 % interval and warnings."""
    ligne = pd.DataFrame([{
        "Rainfall_mm": entree.pluie_mm,
        "Temperature_Celsius": entree.temperature_c,
        "Fertilizer_Used": float(entree.engrais),
        "Irrigation_Used": float(entree.irrigation),
    }])[features.PREDICT_RETENUES]
    rendement = float(ressources.predict.predict(ligne)[0])

    bornes = ressources.domaine["predict"]["bornes"]
    avertissements = [AVERTISSEMENT_CULTURE]
    for message in (
        _hors_domaine("pluie", entree.pluie_mm, bornes["Rainfall_mm"], "mm"),
        _hors_domaine("température", entree.temperature_c, bornes["Temperature_Celsius"], "°C"),
    ):
        if message:
            avertissements.append(message)
    if rendement < 0:
        avertissements.append(
            "L'estimation brute est négative, ramenée à 0 : ces conditions sont hors de ce "
            "que le modèle sait décrire."
        )
        rendement = 0.0

    demi_largeur = Z_95 * ressources.meta_predict["sigma_bruit"]
    return {
        "culture": entree.culture,
        "libelle": LIBELLES[entree.culture],
        "rendement_t_ha": round(rendement, 3),
        "intervalle_95_t_ha": (round(max(rendement - demi_largeur, 0.0), 3),
                               round(rendement + demi_largeur, 3)),
        "avertissements": avertissements,
    }


def contexte_predict(ressources: Ressources) -> dict:
    """Crops, bounds and units, for the interface."""
    domaine = ressources.domaine["predict"]["bornes"]
    return {
        "cultures": LIBELLES,
        "bornes_physiques": {"pluie_mm": {"min": 0, "max": 5000},
                             "temperature_c": {"min": -30, "max": 60}},
        "domaine_entrainement": {"pluie_mm": domaine["Rainfall_mm"],
                                 "temperature_c": domaine["Temperature_Celsius"]},
        "unites": {"pluie_mm": "mm", "temperature_c": "°C", "rendement_t_ha": "t/ha"},
    }


# --------------------------------------------------------------------------- #
# /recommend
# --------------------------------------------------------------------------- #

def contexte_du_pays(ressources: Ressources, iso3: str) -> dict:
    """Stored context of a served country."""
    code = iso3.upper()
    if code not in ressources.contextes:
        raise PaysInconnu(
            f"{code} n'est pas un pays servi. Liste des pays : GET /recommend/pays"
        )
    return ressources.contextes[code]


def prix_du_pays(ressources: Ressources, iso3: str) -> pd.DataFrame:
    """Reference prices of the ranked crops, indexed by FAO label."""
    return pd.DataFrame(ressources.prix["pays"][iso3.upper()]).set_index("crop")


def pays_servis(ressources: Ressources) -> list[dict]:
    return sorted(
        ({"iso3": code, "nom": c["nom"]} for code, c in ressources.contextes.items()),
        key=lambda p: p["nom"],
    )


def contexte_pays_affichable(ressources: Ressources, iso3: str) -> dict:
    """A country's default values, in the units of the request."""
    base = contexte_du_pays(ressources, iso3)
    prix = prix_du_pays(ressources, iso3)
    return {
        "iso3": base["iso3"],
        "nom": base["nom"],
        "annee": base["year"],
        "temperature_c": round(base["temp_hist"], 2),
        "pluie_mm": base["rain_mm"],
        "pesticides_t": round(base["pest_hist"], 1),
        "prix": [
            {"culture": FAO_VERS_PRODUIT[fao], "prix_usd_t": ligne.prix_usd_t,
             "prix_source": ligne.prix_source}
            for fao, ligne in prix.iterrows()
        ],
        "cultures_pratiquees": [
            FAO_VERS_PRODUIT[fao] for fao in base["cultures_observees"] if fao in FAO_VERS_PRODUIT
        ],
    }


def _avertissements_domaine(ressources: Ressources, utilise: dict) -> list[str]:
    """One warning per variable, listing the crops for which the value is unusual."""
    bornes = ressources.domaine["recommend"]["bornes"]
    libelles = {"temperature_c": ("température", "°C"), "pluie_mm": ("pluie", "mm"),
                "pesticides_t": ("pesticides", "t")}
    messages = []
    for champ, (nom, unite) in libelles.items():
        valeur = utilise[champ]
        concernees = [
            LIBELLES[produit]
            for produit, fao in rm.PRODUIT_VERS_FAO.items()
            if not bornes[fao][champ]["min"] <= valeur <= bornes[fao][champ]["max"]
        ]
        if concernees:
            messages.append(
                f"{nom} = {valeur:g} {unite} : valeur rare dans les données pour "
                f"{', '.join(concernees)} (hors des 90 % centraux) — classement moins sûr."
            )
    return messages


def recommande(ressources: Ressources, entree: EntreeRecommend) -> dict:
    """Rank the crops for a country, with the user's conditions and prices if given."""
    base = contexte_du_pays(ressources, entree.iso3)
    contexte = {v: base[v] for v in rm.VARIABLES if v != "crop"} | {"iso3": base["iso3"]}

    saisies = []
    if entree.temperature_c is not None:
        contexte["temp_hist"] = entree.temperature_c
        saisies.append("temperature_c")
    if entree.pluie_mm is not None:
        contexte["rain_mm"] = entree.pluie_mm
        saisies.append("pluie_mm")
    pesticides = base["pest_hist"]
    if entree.pesticides_t is not None:
        pesticides = entree.pesticides_t
        contexte["log_pest_hist"] = float(np.log1p(pesticides))
        saisies.append("pesticides_t")

    reference = prix_du_pays(ressources, base["iso3"])
    for produit, prix in entree.prix_usd_t.items():
        reference.loc[rm.PRODUIT_VERS_FAO[produit]] = [prix, SOURCE_SAISIE]
    if entree.prix_usd_t:
        saisies.append("prix_usd_t")

    table = rm.predit_cultures(ressources.recommend, contexte, rm.PRODUIT_VERS_FAO)
    table = rm.ajoute_produit_brut(table, reference)
    table["cultivee"] = table["culture_fao"].isin(base["cultures_observees"])
    table = table.sort_values(COLONNE_CRITERE[entree.critere], ascending=False).reset_index(drop=True)

    utilise = {
        "iso3": base["iso3"], "pays": base["nom"], "annee": base["year"],
        "temperature_c": round(contexte["temp_hist"], 2), "pluie_mm": contexte["rain_mm"],
        "pesticides_t": round(pesticides, 1), "valeurs_saisies": saisies,
    }

    avertissements = [
        f"Conditions à l'échelle du pays ({base['nom']}, données {base['year']}), pas de la "
        "parcelle : le classement est un ordre de grandeur.",
        *_avertissements_domaine(ressources, utilise),
    ]
    non_cultivees = [LIBELLES[c] for c in table.loc[~table["cultivee"], "culture"]]
    if non_cultivees:
        avertissements.append(
            f"Jamais cultivé dans ce pays : {', '.join(non_cultivees)}. "
            "Le modèle extrapole, et la validation ne mesure pas ce cas."
        )
    mondiaux = [LIBELLES[c] for c in table.loc[table["prix_source"] == "médiane mondiale", "culture"]]
    if mondiaux:
        avertissements.append(
            f"Pas de prix national pour {', '.join(mondiaux)} : médiane mondiale FAOSTAT utilisée."
        )
    if entree.critere == "produit_brut":
        avertissements.append(
            "Le produit brut (rendement x prix) n'est pas une marge : les coûts de production "
            "sont inconnus."
        )

    return {
        "contexte": utilise,
        "critere": entree.critere,
        "classement": [
            {
                "rang": rang,
                "culture": ligne.culture,
                "libelle": LIBELLES[ligne.culture],
                "rendement_t_ha": round(ligne.rendement_brut_t_ha, 3),
                "matiere_seche_t_ha": round(ligne.matiere_seche_t_ha, 3),
                "groupe": ligne.groupe,
                "prix_usd_t": round(float(ligne.prix_usd_t), 1),
                "prix_source": ligne.prix_source,
                "produit_brut_usd_ha": round(float(ligne.produit_brut_usd_ha), 1),
                "cultivee_dans_le_pays": bool(ligne.cultivee),
            }
            for rang, ligne in enumerate(table.itertuples(), start=1)
        ],
        "cultures_non_classees": [
            {"culture": culture, "libelle": LIBELLES[culture], "motif": motif}
            for culture, motif in rm.CULTURES_EXCLUES.items()
        ],
        "avertissements": avertissements,
    }


def _controle() -> None:
    """Every crop of the application has a display name."""
    manquantes = set(CROP_TO_FAO_ITEM) - set(LIBELLES)
    if manquantes:
        raise ValueError(f"libelles manquants : {sorted(manquantes)}")


_controle()
