"""HTTP contract of the API: statuses, shapes, and the values the models must return."""

import pytest


def test_health(client):
    reponse = client.get("/health")
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["statut"] == "ok"
    assert corps["scikit_learn"] == corps["scikit_learn_entrainement"]


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #

def test_predict_retrouve_la_formule(client, parcelle, coefficients):
    """The estimate is exactly the linear formula of the served model."""
    coef, constante = coefficients
    attendu = (constante + coef["Rainfall_mm"] * 600 + coef["Temperature_Celsius"] * 22
               + coef["Fertilizer_Used"] * 1 + coef["Irrigation_Used"] * 0)
    corps = client.post("/predict", json=parcelle).json()
    assert corps["rendement_t_ha"] == pytest.approx(attendu, abs=1e-3)
    bas, haut = corps["intervalle_95_t_ha"]
    assert bas < corps["rendement_t_ha"] < haut


def test_predict_ne_depend_pas_de_la_culture(client, parcelle):
    rendements = {
        client.post("/predict", json=parcelle | {"culture": c}).json()["rendement_t_ha"]
        for c in ("Wheat", "Rice", "Cotton")
    }
    assert len(rendements) == 1


@pytest.mark.parametrize("champ, valeur", [
    ("pluie_mm", -1), ("temperature_c", 80), ("culture", "Teff"), ("engrais", "peut-etre"),
])
def test_predict_refuse_l_impossible(client, parcelle, champ, valeur):
    assert client.post("/predict", json=parcelle | {champ: valeur}).status_code == 422


def test_predict_signale_le_hors_domaine(client, parcelle):
    """2 000 mm is possible but beyond the simulated data (max 1 000): answered, flagged."""
    reponse = client.post("/predict", json=parcelle | {"pluie_mm": 2000})
    assert reponse.status_code == 200
    assert any("hors du domaine" in a for a in reponse.json()["avertissements"])


def test_predict_contexte(client):
    corps = client.get("/predict/contexte").json()
    assert len(corps["cultures"]) == 6
    assert corps["domaine_entrainement"]["pluie_mm"]["max"] == pytest.approx(1000, abs=1)


# --------------------------------------------------------------------------- #
# /recommend
# --------------------------------------------------------------------------- #

def test_recommend_trie_par_rendement_par_defaut(client):
    corps = client.post("/recommend", json={"iso3": "FRA"}).json()
    classement = corps["classement"]
    assert [l["culture"] for l in classement][:1] == ["Maize"]
    assert len(classement) == 4
    rendements = [l["rendement_t_ha"] for l in classement]
    assert rendements == sorted(rendements, reverse=True)
    assert [l["rang"] for l in classement] == [1, 2, 3, 4]
    assert {c["culture"] for c in corps["cultures_non_classees"]} == {"Barley", "Cotton"}


@pytest.mark.parametrize("critere, champ", [
    ("matiere_seche", "matiere_seche_t_ha"), ("produit_brut", "produit_brut_usd_ha"),
])
def test_recommend_trie_selon_le_critere(client, critere, champ):
    classement = client.post("/recommend", json={"iso3": "BRA", "critere": critere}).json()["classement"]
    valeurs = [l[champ] for l in classement]
    assert valeurs == sorted(valeurs, reverse=True)


def test_recommend_surcharge_le_contexte(client):
    defaut = client.post("/recommend", json={"iso3": "FRA"}).json()
    modifie = client.post("/recommend", json={"iso3": "FRA", "temperature_c": 25.0}).json()
    assert modifie["contexte"]["temperature_c"] == 25.0
    assert modifie["contexte"]["valeurs_saisies"] == ["temperature_c"]
    assert defaut["classement"] != modifie["classement"]


def test_recommend_prix_saisi(client):
    corps = client.post("/recommend", json={"iso3": "FRA", "prix_usd_t": {"Wheat": 1000}}).json()
    ble = next(l for l in corps["classement"] if l["culture"] == "Wheat")
    assert ble["prix_source"] == "saisi"
    assert ble["produit_brut_usd_ha"] == pytest.approx(ble["rendement_t_ha"] * 1000, rel=1e-3)


def test_recommend_accepte_la_casse(client):
    assert client.post("/recommend", json={"iso3": "fra"}).status_code == 200


@pytest.mark.parametrize("corps, statut", [
    ({"iso3": "ZZZ"}, 404),
    ({"iso3": "FRANCE"}, 422),
    ({"iso3": "FRA", "critere": "bénéfice"}, 422),
    ({"iso3": "FRA", "prix_usd_t": {"Barley": 200}}, 422),
    ({"iso3": "FRA", "prix_usd_t": {"Wheat": -5}}, 422),
])
def test_recommend_erreurs(client, corps, statut):
    assert client.post("/recommend", json=corps).status_code == statut


def test_pays(client):
    pays = client.get("/recommend/pays").json()
    assert len(pays) == 117
    assert {"iso3": "FRA", "nom": "France"} in pays
    assert client.get("/recommend/pays/ZZZ").status_code == 404


def test_contexte_pays_redonne_le_meme_classement(client):
    """Sending back the prefilled values must give the default ranking."""
    defaut = client.get("/recommend/pays/FRA").json()
    explicite = client.post("/recommend", json={
        "iso3": "FRA", "temperature_c": defaut["temperature_c"],
        "pluie_mm": defaut["pluie_mm"], "pesticides_t": defaut["pesticides_t"],
    }).json()
    implicite = client.post("/recommend", json={"iso3": "FRA"}).json()
    for a, b in zip(explicite["classement"], implicite["classement"]):
        assert a["culture"] == b["culture"]
        assert a["rendement_t_ha"] == pytest.approx(b["rendement_t_ha"], abs=0.05)
