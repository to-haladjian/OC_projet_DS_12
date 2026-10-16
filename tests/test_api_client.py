"""The interface's HTTP client, with `requests` replaced by a fake: no API needed."""

import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import api_client  # noqa: E402

DEFAUTS_FRA = {
    "temperature_c": 11.32, "pluie_mm": 867.0, "pesticides_t": 62163.2,
    "prix": [{"culture": "Maize", "prix_usd_t": 247.0, "prix_source": "pays"},
             {"culture": "Wheat", "prix_usd_t": 246.4, "prix_source": "pays"}],
}


class FausseReponse:
    def __init__(self, statut, corps):
        self.status_code, self._corps = statut, corps
        self.ok = statut < 400

    def json(self):
        return self._corps


@pytest.fixture
def appels(monkeypatch):
    """Record the calls, and answer with whatever the test queued."""
    enregistres, reponses = [], []

    def faux_request(methode, url, json=None, timeout=None):
        enregistres.append({"methode": methode, "url": url, "json": json, "timeout": timeout})
        reponse = reponses.pop(0)
        if isinstance(reponse, Exception):
            raise reponse
        return reponse

    monkeypatch.setattr(api_client.requests, "request", faux_request)
    return enregistres, reponses


# --------------------------------------------------------------------------- #
# Only send what changed
# --------------------------------------------------------------------------- #

def test_rien_de_modifie():
    saisie = {"temperature_c": 11.32, "pluie_mm": 867.0, "pesticides_t": 62163.2,
              "prix_usd_t": {"Maize": 247.0, "Wheat": 246.4}}
    assert api_client.champs_modifies(saisie, DEFAUTS_FRA) == {}


def test_seules_les_modifications_partent():
    saisie = {"temperature_c": 13.0, "pluie_mm": 867.0, "pesticides_t": 62163.2,
              "prix_usd_t": {"Maize": 247.0, "Wheat": 300.0}}
    assert api_client.champs_modifies(saisie, DEFAUTS_FRA) == {
        "temperature_c": 13.0, "prix_usd_t": {"Wheat": 300.0}}


# --------------------------------------------------------------------------- #
# Calls and errors
# --------------------------------------------------------------------------- #

def test_url_lue_dans_l_environnement(monkeypatch, appels):
    enregistres, reponses = appels
    monkeypatch.setenv("API_URL", "http://api:8000/")
    reponses.append(FausseReponse(200, {"statut": "ok"}))
    api_client.sante()
    assert enregistres[0]["url"] == "http://api:8000/health"


def test_recommander_envoie_pays_critere_et_modifications(appels):
    enregistres, reponses = appels
    reponses.append(FausseReponse(200, {"classement": []}))
    api_client.recommander("FRA", "produit_brut", {"temperature_c": 13.0})
    assert enregistres[0]["json"] == {"iso3": "FRA", "critere": "produit_brut", "temperature_c": 13.0}
    assert enregistres[0]["timeout"] == api_client.DELAI_CALCUL


def test_api_injoignable(appels):
    _, reponses = appels
    reponses.append(requests.ConnectionError("refus"))
    with pytest.raises(api_client.ErreurAPI, match="ne répond pas"):
        api_client.pays()


def test_delai_depasse(appels):
    _, reponses = appels
    reponses.append(requests.Timeout("lent"))
    with pytest.raises(api_client.ErreurAPI, match="à temps"):
        api_client.predire("Wheat", 600, 22, True, False)


def test_404_redonne_le_message_de_l_api(appels):
    _, reponses = appels
    reponses.append(FausseReponse(404, {"detail": "ZZZ n'est pas un pays servi."}))
    with pytest.raises(api_client.ErreurAPI) as erreur:
        api_client.contexte_pays("ZZZ")
    assert erreur.value.statut == 404
    assert "ZZZ" in erreur.value.message


def test_422_lisible(appels):
    _, reponses = appels
    reponses.append(FausseReponse(422, {"detail": [
        {"loc": ["body", "pluie_mm"], "msg": "Input should be greater than or equal to 0"}]}))
    with pytest.raises(api_client.ErreurAPI, match="pluie_mm : Input should be"):
        api_client.predire("Wheat", -1, 22, True, False)


def test_503_pendant_le_demarrage(appels):
    _, reponses = appels
    reponses.append(FausseReponse(503, {"detail": "modèles non chargés"}))
    with pytest.raises(api_client.ErreurAPI, match="réessayez"):
        api_client.sante()
