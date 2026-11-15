"""HTTP contract of the API: statuses, shapes, and the values the models must return."""

import pytest


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["scikit_learn"] == body["scikit_learn_training"]


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #

def test_predict_recovers_the_formula(client, parcel, coefficients):
    """The estimate is exactly the linear formula of the served model."""
    coef, intercept = coefficients
    expected = (intercept + coef["Rainfall_mm"] * 600 + coef["Temperature_Celsius"] * 22
               + coef["Fertilizer_Used"] * 1 + coef["Irrigation_Used"] * 0)
    body = client.post("/predict", json=parcel).json()
    assert body["yield_t_ha"] == pytest.approx(expected, abs=1e-3)
    low, high = body["interval_95_t_ha"]
    assert low < body["yield_t_ha"] < high


def test_predict_does_not_depend_on_the_crop(client, parcel):
    yields = {
        client.post("/predict", json=parcel | {"crop": c}).json()["yield_t_ha"]
        for c in ("Wheat", "Rice", "Cotton")
    }
    assert len(yields) == 1


@pytest.mark.parametrize("field_name, value", [
    ("rain_mm", -1), ("temperature_c", 80), ("crop", "Teff"), ("fertilizer", "peut-etre"),
])
def test_predict_rejects_the_impossible(client, parcel, field_name, value):
    assert client.post("/predict", json=parcel | {field_name: value}).status_code == 422


def test_predict_flags_out_of_domain(client, parcel):
    """2 000 mm is possible but beyond the simulated data (max 1 000): answered, flagged."""
    response = client.post("/predict", json=parcel | {"rain_mm": 2000})
    assert response.status_code == 200
    assert any("hors du domaine" in a for a in response.json()["warnings"])


def test_predict_context(client):
    body = client.get("/predict/context").json()
    assert len(body["crops"]) == 6
    assert body["learned_domain"]["rain_mm"]["max"] == pytest.approx(1000, abs=1)


# --------------------------------------------------------------------------- #
# /recommend
# --------------------------------------------------------------------------- #

def test_recommend_sorts_by_yield_by_default(client):
    body = client.post("/recommend", json={"iso3": "FRA"}).json()
    ranking = body["ranking"]
    assert [l["crop"] for l in ranking][:1] == ["Maize"]
    assert len(ranking) == 4
    yields = [l["yield_t_ha"] for l in ranking]
    assert yields == sorted(yields, reverse=True)
    assert [l["rank"] for l in ranking] == [1, 2, 3, 4]
    assert {c["crop"] for c in body["unranked_crops"]} == {"Barley", "Cotton"}


@pytest.mark.parametrize("criterion, field_name", [
    ("dry_matter", "dry_matter_t_ha"), ("gross_product", "gross_product_usd_ha"),
])
def test_recommend_sorts_by_criterion(client, criterion, field_name):
    ranking = client.post("/recommend", json={"iso3": "BRA", "criterion": criterion}).json()["ranking"]
    values = [l[field_name] for l in ranking]
    assert values == sorted(values, reverse=True)


def test_recommend_overrides_the_context(client):
    default_value = client.post("/recommend", json={"iso3": "FRA"}).json()
    changed = client.post("/recommend", json={"iso3": "FRA", "temperature_c": 25.0}).json()
    assert changed["context"]["temperature_c"] == 25.0
    assert changed["context"]["entered_values"] == ["temperature_c"]
    assert default_value["ranking"] != changed["ranking"]


def test_recommend_entered_price(client):
    body = client.post("/recommend", json={"iso3": "FRA", "price_usd_t": {"Wheat": 1000}}).json()
    wheat = next(l for l in body["ranking"] if l["crop"] == "Wheat")
    assert wheat["price_source"] == "saisi"
    assert wheat["gross_product_usd_ha"] == pytest.approx(wheat["yield_t_ha"] * 1000, rel=1e-3)


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


def test_countries(client):
    countries = client.get("/recommend/countries").json()
    assert len(countries) == 117
    assert {"iso3": "FRA", "name": "France"} in countries
    assert client.get("/recommend/countries/ZZZ").status_code == 404


def test_country_context_gives_the_same_ranking(client):
    """Sending back the prefilled values must give the default ranking."""
    default_value = client.get("/recommend/countries/FRA").json()
    explicit = client.post("/recommend", json={
        "iso3": "FRA", "temperature_c": default_value["temperature_c"],
        "rain_mm": default_value["rain_mm"], "pesticides_t": default_value["pesticides_t"],
    }).json()
    implicit = client.post("/recommend", json={"iso3": "FRA"}).json()
    for a, b in zip(explicit["ranking"], implicit["ranking"]):
        assert a["crop"] == b["crop"]
        assert a["yield_t_ha"] == pytest.approx(b["yield_t_ha"], abs=0.05)
