"""Serving assets: everything the API needs besides the two models, as small JSON files.

The API must not read `data/`: processed CSVs are not versioned, and the Docker image
should only carry what it serves. This module therefore freezes, next to the models:

* `contextes_pays.json` — for each served country, the latest known conditions in the
  form the `/recommend` model expects, plus its name and the crops it actually grows;
* `prix_reference.json` — reference producer price of each ranked crop, per country,
  with its source, plus the world medians used as fallback;
* `domaine_service.json` — training-domain bounds of both services, used to **warn**
  about an unusual input without rejecting it, and the scikit-learn version the models
  were trained with.

Runnable as a script, after both models have been trained:
`python -m src.models.actifs_service`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import sklearn

from src import config
from src.data import prix as prix_fao
from src.models import features
from src.models import recommend_model as rm

CONTEXTES_PAYS_JSON = config.MODELES / "contextes_pays.json"
PRIX_REFERENCE_JSON = config.MODELES / "prix_reference.json"
DOMAINE_SERVICE_JSON = config.MODELES / "domaine_service.json"

#: Variables of the FAO training domain the API checks, and the request field each
#: one corresponds to.
DOMAINE_RECOMMEND = {"avg_temp": "temperature_c", "rain_mm": "pluie_mm", "pesticides_t": "pesticides_t"}


def contextes_pays(df: pd.DataFrame) -> dict[str, dict]:
    """Latest context of each country, as :func:`recommend_model.contexte_pays` builds it."""
    noms = df.drop_duplicates("iso3").set_index("iso3")["area"]
    resultat = {}
    for iso3 in sorted(df["iso3"].unique()):
        contexte = rm.contexte_pays(df, iso3)
        dernier = df[(df["iso3"] == iso3) & (df["year"] == contexte["year"])].iloc[0]
        resultat[iso3] = {
            "nom": str(noms[iso3]),
            **contexte,
            # Pesticides in tonnes, for display: the model sees their log.
            "pest_hist": float(dernier["pest_hist"]),
            "cultures_observees": sorted(rm.cultures_du_pays(df, iso3)),
        }
    return resultat


def prix_reference(contextes: dict[str, dict], prix: pd.DataFrame) -> dict:
    """Reference price of the ranked crops, per country, at the context year."""
    cultures = list(rm.PRODUIT_VERS_FAO.values())
    annee = max(c["year"] for c in contextes.values())
    fenetre = prix[prix["year"].between(annee - prix_fao.FENETRE_ANNEES + 1, annee)]
    return {
        "unite": "USD/t",
        "source": config.SOURCE_PRIX_URL,
        "regle": (
            f"mediane des {prix_fao.FENETRE_ANNEES} dernieres annees connues du pays, "
            "sinon mediane mondiale"
        ),
        "mediane_mondiale": {
            culture: float(fenetre.loc[fenetre["crop"] == culture, "prix_usd_t"].median())
            for culture in cultures
        },
        "pays": {
            iso3: prix_fao.prix_reference(prix, iso3, contexte["year"], cultures)
            .reset_index()
            .to_dict(orient="records")
            for iso3, contexte in contextes.items()
        },
    }


def domaine_service() -> dict:
    """Training-domain bounds of both services."""
    simule = pd.read_csv(config.DATASET_CONSOLIDE_CSV, usecols=features.PREDICT_RETENUES[:2])
    domaine_fao = pd.read_csv(config.DOMAINE_APPRENTISSAGE_CSV)
    domaine_fao = domaine_fao[domaine_fao["variable"].isin(DOMAINE_RECOMMEND)]
    return {
        "scikit_learn": sklearn.__version__,
        "predict": {
            "source": "min / max du jeu simule d'entrainement",
            "bornes": {
                colonne: {"min": float(simule[colonne].min()), "max": float(simule[colonne].max())}
                for colonne in simule.columns
            },
        },
        "recommend": {
            "source": "quantiles 5 % / 95 % par culture FAO, annee de test exclue",
            "bornes": {
                culture: {
                    DOMAINE_RECOMMEND[ligne.variable]: {"min": float(ligne.q05), "max": float(ligne.q95)}
                    for ligne in bloc.itertuples()
                }
                for culture, bloc in domaine_fao.groupby("crop")
            },
        },
    }


def _ecrit(chemin, contenu) -> None:
    def defaut(valeur):
        if isinstance(valeur, np.generic):
            return valeur.item()
        raise TypeError(f"non serialisable : {type(valeur)}")

    chemin.write_text(json.dumps(contenu, indent=1, ensure_ascii=False, default=defaut), encoding="utf-8")
    print(f"ecrit {chemin.relative_to(config.RACINE)} ({chemin.stat().st_size / 1024:.0f} Ko)")


def main() -> None:
    df = rm.charge()
    contextes = contextes_pays(df)
    _ecrit(CONTEXTES_PAYS_JSON, contextes)
    _ecrit(PRIX_REFERENCE_JSON, prix_reference(contextes, prix_fao.charge()))
    _ecrit(DOMAINE_SERVICE_JSON, domaine_service())
    print(f"{len(contextes)} pays, annee de contexte {max(c['year'] for c in contextes.values())}")


if __name__ == "__main__":
    main()
