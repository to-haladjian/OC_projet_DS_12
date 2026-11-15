"""HTTP contract of the API: statuses, shapes, and the values the model must return."""

import pytest

CROPS = ("Maize", "Wheat", "Rice", "Soybean")


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["scikit_learn"] == body["scikit_learn_training"]
    assert set(body["models"]) == {"yield", "practice_effects"}


# --------------------------------------------------------------------------- #
# One engine: /predict and /recommend cannot disagree
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("conditions", [
    {"iso3": "FRA"},
    {"iso3": "IND"},
    {"iso3": "BRA", "temperature_c": 18.0, "rain_mm": 900},
    {"iso3": "FRA", "pesticides_t": 1000},
])
def test_predict_and_recommend_agree(client, conditions):
    """The same crop in the same conditions gets the same yield from both endpoints."""
    ranking = client.post("/recommend", json=conditions).json()["ranking"]
    ranked = {row["crop"]: row for row in ranking}
    for crop in CROPS:
        predicted = client.post("/predict", json=conditions | {"crop": crop}).json()
        assert predicted["yield_t_ha"] == ranked[crop]["yield_t_ha"]
        assert predicted["dry_matter_t_ha"] == ranked[crop]["dry_matter_t_ha"]
        assert predicted["grown_in_country"] == ranked[crop]["grown_in_country"]


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #

def test_predict_shape(client, wheat_in_france):
    body = client.post("/predict", json=wheat_in_france).json()
    assert body["crop"] == "Wheat" and body["label"] == "Blé"
    assert body["context"]["country"] == "France"
    assert body["context"]["entered_values"] == []
    low, high = body["interval_95_t_ha"]
    assert 0 < low < body["yield_t_ha"] < high


def test_predict_depends_on_the_crop(client):
    yields = {
        client.post("/predict", json={"iso3": "FRA", "crop": c}).json()["yield_t_ha"]
        for c in CROPS
    }
    assert len(yields) == len(CROPS)


def test_predict_overrides_the_context(client, wheat_in_france):
    default_value = client.post("/predict", json=wheat_in_france).json()
    changed = client.post("/predict", json=wheat_in_france | {"temperature_c": 25.0}).json()
    assert changed["context"]["temperature_c"] == 25.0
    assert changed["context"]["entered_values"] == ["temperature_c"]
    assert changed["yield_t_ha"] != default_value["yield_t_ha"]


def test_predict_reports_practice_effects_apart(client, wheat_in_france):
    """Fertilizer and irrigation come from the simulated plots, never added to the yield."""
    effects = client.post("/predict", json=wheat_in_france).json()["practice_effects"]
    by_practice = {e["practice"]: e["effect_t_ha"] for e in effects}
    assert by_practice == {
        "fertilizer": pytest.approx(1.50, abs=0.02),
        "irrigation": pytest.approx(1.20, abs=0.02),
    }


@pytest.mark.parametrize("crop", ["Barley", "Cotton"])
def test_predict_refuses_crops_absent_from_the_data(client, crop):
    """Known to the product, absent from the FAO data: refused with the reason."""
    response = client.post("/predict", json={"iso3": "FRA", "crop": crop})
    assert response.status_code == 422
    assert "absente des données FAO" in response.json()["detail"]


@pytest.mark.parametrize("field_name, value", [
    ("rain_mm", -1), ("temperature_c", 80), ("crop", "Teff"), ("iso3", "FRANCE"),
])
def test_predict_rejects_the_impossible(client, wheat_in_france, field_name, value):
    assert client.post("/predict", json=wheat_in_france | {field_name: value}).status_code == 422


def test_predict_unknown_country(client):
    assert client.post("/predict", json={"iso3": "ZZZ", "crop": "Wheat"}).status_code == 404


def test_predict_flags_out_of_domain(client, wheat_in_france):
    """40 °C is possible but rare for wheat: answered, flagged."""
    response = client.post("/predict", json=wheat_in_france | {"temperature_c": 40})
    assert response.status_code == 200
    assert any("valeur rare" in a for a in response.json()["warnings"])


def test_predict_context(client):
    body = client.get("/predict/context").json()
    assert set(body["crops"]) == set(CROPS)
    assert {c["crop"] for c in body["unavailable_crops"]} == {"Barley", "Cotton"}
    assert body["physical_bounds"]["rain_mm"]["max"] == 5000


# --------------------------------------------------------------------------- #
# /recommend
# --------------------------------------------------------------------------- #

def test_recommend_sorts_by_yield_by_default(client):
    body = client.post("/recommend", json={"iso3": "FRA"}).json()
    ranking = body["ranking"]
    assert [row["crop"] for row in ranking][:1] == ["Maize"]
    assert len(ranking) == 4
    yields = [row["yield_t_ha"] for row in ranking]
    assert yields == sorted(yields, reverse=True)
    assert [row["rank"] for row in ranking] == [1, 2, 3, 4]
    assert {c["crop"] for c in body["unranked_crops"]} == {"Barley", "Cotton"}


@pytest.mark.parametrize("criterion, field_name", [
    ("dry_matter", "dry_matter_t_ha"), ("gross_product", "gross_product_usd_ha"),
])
def test_recommend_sorts_by_criterion(client, criterion, field_name):
    ranking = client.post("/recommend", json={"iso3": "BRA", "criterion": criterion}).json()["ranking"]
    values = [row[field_name] for row in ranking]
    assert values == sorted(values, reverse=True)


def test_recommend_overrides_the_context(client):
    default_value = client.post("/recommend", json={"iso3": "FRA"}).json()
    changed = client.post("/recommend", json={"iso3": "FRA", "temperature_c": 25.0}).json()
    assert changed["context"]["temperature_c"] == 25.0
    assert changed["context"]["entered_values"] == ["temperature_c"]
    assert default_value["ranking"] != changed["ranking"]


def test_recommend_entered_price(client):
    body = client.post("/recommend", json={"iso3": "FRA", "price_usd_t": {"Wheat": 1000}}).json()
    wheat = next(row for row in body["ranking"] if row["crop"] == "Wheat")
    assert wheat["price_source"] == "saisi"
    assert wheat["gross_product_usd_ha"] == pytest.approx(wheat["yield_t_ha"] * 1000, rel=1e-3)
    assert body["context"]["entered_values"] == ["price_usd_t"]


def test_recommend_accepts_any_case(client):
    assert client.post("/recommend", json={"iso3": "fra"}).status_code == 200


@pytest.mark.parametrize("body, status", [
    ({"iso3": "ZZZ"}, 404),
    ({"iso3": "FRANCE"}, 422),
    ({"iso3": "FRA", "criterion": "bénéfice"}, 422),
    ({"iso3": "FRA", "price_usd_t": {"Barley": 200}}, 422),
    ({"iso3": "FRA", "price_usd_t": {"Wheat": -5}}, 422),
])
def test_recommend_errors(client, body, status):
    assert client.post("/recommend", json=body).status_code == status


# --------------------------------------------------------------------------- #
# Countries
# --------------------------------------------------------------------------- #

def test_countries(client):
    countries = client.get("/countries").json()
    assert len(countries) == 117
    assert {"iso3": "FRA", "name": "France"} in countries
    assert client.get("/countries/ZZZ").status_code == 404


def test_country_context_gives_the_same_ranking(client):
    """Sending back the prefilled values must give the default ranking."""
    default_value = client.get("/countries/FRA").json()
    explicit = client.post("/recommend", json={
        "iso3": "FRA", "temperature_c": default_value["temperature_c"],
        "rain_mm": default_value["rain_mm"], "pesticides_t": default_value["pesticides_t"],
    }).json()
    implicit = client.post("/recommend", json={"iso3": "FRA"}).json()
    for a, b in zip(explicit["ranking"], implicit["ranking"]):
        assert a["crop"] == b["crop"]
        assert a["yield_t_ha"] == pytest.approx(b["yield_t_ha"], abs=0.05)
