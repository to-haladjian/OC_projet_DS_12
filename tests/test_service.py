"""Serving functions, tested without HTTP."""

import pandas as pd
import pytest

from api import service
from api.schemas import EntreeRecommend
from src.models import recommend_model as rm


def test_les_ressources_couvrent_les_pays_et_les_prix(ressources):
    assert len(ressources.contextes) == 117
    assert set(ressources.prix["pays"]) == set(ressources.contextes)
    for iso3 in ressources.contextes:
        prix = service.prix_du_pays(ressources, iso3)
        assert set(prix.index) == set(rm.PRODUIT_VERS_FAO.values())
        assert (prix["prix_usd_t"] > 0).all()


def test_contexte_stocke_conforme_au_modele(ressources):
    """Each stored context carries every variable the model expects."""
    attendues = set(rm.VARIABLES) - {"crop"}
    for contexte in ressources.contextes.values():
        assert attendues <= set(contexte)


def test_pays_inconnu(ressources):
    with pytest.raises(service.PaysInconnu):
        service.contexte_du_pays(ressources, "ZZZ")


def test_recommande_equivaut_au_classement_des_notebooks(ressources):
    """The API ranking is the step 2 ranking: same model, same context, same order."""
    contexte = {k: v for k, v in ressources.contextes["FRA"].items()
                if k in rm.VARIABLES or k == "iso3"}
    reference = rm.classement(ressources.recommend, contexte, critere="rendement_brut_t_ha")
    api = service.recommande(ressources, EntreeRecommend(iso3="FRA"))["classement"]
    assert [l["culture"] for l in api] == list(reference["culture"])
    assert [l["rendement_t_ha"] for l in api] == pytest.approx(
        list(reference["rendement_brut_t_ha"]), abs=1e-3)


def test_ajoute_produit_brut_refuse_un_prix_manquant():
    table = pd.DataFrame({"culture": ["Wheat"], "culture_fao": ["Wheat"],
                          "rendement_brut_t_ha": [3.0]})
    with pytest.raises(KeyError):
        rm.ajoute_produit_brut(table, pd.DataFrame(columns=["prix_usd_t", "prix_source"]))


def test_libelles_complets():
    assert set(service.LIBELLES) == {"Maize", "Wheat", "Rice", "Soybean", "Barley", "Cotton"}
