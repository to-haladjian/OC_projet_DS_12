"""Serving functions, tested without HTTP."""

import pandas as pd
import pytest

from api import service
from api.schemas import PredictInput, RecommendInput
from src.models import yield_model as ym


def test_resources_cover_countries_and_prices(resources):
    assert len(resources.contexts) == 117
    assert set(resources.prices["countries"]) == set(resources.contexts)
    for iso3 in resources.contexts:
        prices = service.country_prices(resources, iso3)
        assert set(prices.index) == set(ym.PRODUCT_TO_FAO.values())
        assert (prices["price_usd_t"] > 0).all()


def test_stored_context_matches_the_model(resources):
    """Each stored context carries every variable the model expects."""
    expected_keys = set(ym.VARIABLES) - {"crop"}
    for context in resources.contexts.values():
        assert expected_keys <= set(context)


def test_domain_covers_every_served_crop(resources):
    """Every served crop has its usual ranges and its validation error bounds."""
    served = set(ym.PRODUCT_TO_FAO.values())
    assert served <= set(resources.domain["bounds"]["crops"])
    errors = resources.domain["relative_error"]["crops"]
    assert set(errors) == served
    for bounds in errors.values():
        assert -1 < bounds["low"] < 0 < bounds["high"]


def test_unknown_country(resources):
    with pytest.raises(service.UnknownCountry):
        service.stored_country_context(resources, "ZZZ")


def test_unavailable_crop(resources):
    with pytest.raises(service.UnavailableCrop, match="Orge"):
        service.predict(resources, PredictInput(iso3="FRA", crop="Barley"))


def test_recommend_matches_the_training_code(resources):
    """The API ranking is the one the training code computes: same model, same order."""
    context = {k: v for k, v in resources.contexts["FRA"].items()
               if k in ym.VARIABLES or k == "iso3"}
    reference = ym.ranking(resources.yield_model, context, criterion="raw_yield_t_ha")
    api = service.recommend(resources, RecommendInput(iso3="FRA"))["ranking"]
    assert [row["crop"] for row in api] == list(reference["crop"])
    assert [row["yield_t_ha"] for row in api] == pytest.approx(
        list(reference["raw_yield_t_ha"]), abs=1e-3)


def test_interval_uses_the_validation_error(resources):
    """The interval is the prediction scaled by the crop's relative error bounds."""
    body = service.predict(resources, PredictInput(iso3="FRA", crop="Wheat"))
    bounds = resources.domain["relative_error"]["crops"]["Wheat"]
    low, high = body["interval_95_t_ha"]
    assert low == pytest.approx(body["yield_t_ha"] * (1 + bounds["low"]), abs=2e-3)
    assert high == pytest.approx(body["yield_t_ha"] * (1 + bounds["high"]), abs=2e-3)


def test_add_gross_product_rejects_a_missing_price():
    table = pd.DataFrame({"crop": ["Wheat"], "fao_crop": ["Wheat"],
                          "raw_yield_t_ha": [3.0]})
    with pytest.raises(KeyError):
        ym.add_gross_product(table, pd.DataFrame(columns=["price_usd_t", "price_source"]))


def test_labels_are_complete():
    assert set(service.LABELS) == {"Maize", "Wheat", "Rice", "Soybean", "Barley", "Cotton"}
