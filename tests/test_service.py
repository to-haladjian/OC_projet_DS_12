"""Serving functions, tested without HTTP."""

import pandas as pd
import pytest

from api import service
from api.schemas import RecommendInput
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


def test_unknown_country(resources):
    with pytest.raises(service.UnknownCountry):
        service.stored_country_context(resources, "ZZZ")


def test_recommend_matches_the_notebook_ranking(resources):
    """The API ranking is the step 2 ranking: same model, same context, same order."""
    context = {k: v for k, v in resources.contexts["FRA"].items()
                if k in ym.VARIABLES or k == "iso3"}
    reference = ym.ranking(resources.recommend, context, criterion="raw_yield_t_ha")
    api = service.recommend(resources, RecommendInput(iso3="FRA"))["ranking"]
    assert [l["crop"] for l in api] == list(reference["crop"])
    assert [l["yield_t_ha"] for l in api] == pytest.approx(
        list(reference["raw_yield_t_ha"]), abs=1e-3)


def test_add_gross_product_rejects_a_missing_price():
    table = pd.DataFrame({"crop": ["Wheat"], "fao_crop": ["Wheat"],
                          "raw_yield_t_ha": [3.0]})
    with pytest.raises(KeyError):
        ym.add_gross_product(table, pd.DataFrame(columns=["price_usd_t", "price_source"]))


def test_labels_are_complete():
    assert set(service.LABELS) == {"Maize", "Wheat", "Rice", "Soybean", "Barley", "Cotton"}
