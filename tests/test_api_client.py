"""The interface's HTTP client, with `requests` replaced by a fake: no API needed."""

import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import api_client  # noqa: E402

FRA_DEFAULTS = {
    "temperature_c": 11.32, "rain_mm": 867.0, "pesticides_t": 62163.2,
    "prices": [{"crop": "Maize", "price_usd_t": 247.0, "price_source": "pays"},
             {"crop": "Wheat", "price_usd_t": 246.4, "price_source": "pays"}],
}


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.ok = status < 400

    def json(self):
        return self._body


@pytest.fixture
def calls(monkeypatch):
    """Record the calls, and answer with whatever the test queued."""
    recorded, responses = [], []

    def fake_request(method, url, json=None, timeout=None):
        recorded.append({"method": method, "url": url, "json": json, "timeout": timeout})
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(api_client.requests, "request", fake_request)
    return recorded, responses


# --------------------------------------------------------------------------- #
# Only send what changed
# --------------------------------------------------------------------------- #

def test_nothing_changed():
    entered = {"temperature_c": 11.32, "rain_mm": 867.0, "pesticides_t": 62163.2,
              "price_usd_t": {"Maize": 247.0, "Wheat": 246.4}}
    assert api_client.changed_fields(entered, FRA_DEFAULTS) == {}


def test_only_changes_are_sent():
    entered = {"temperature_c": 13.0, "rain_mm": 867.0, "pesticides_t": 62163.2,
              "price_usd_t": {"Maize": 247.0, "Wheat": 300.0}}
    assert api_client.changed_fields(entered, FRA_DEFAULTS) == {
        "temperature_c": 13.0, "price_usd_t": {"Wheat": 300.0}}


def test_changes_without_prices():
    """The prediction form has no prices: the conditions alone are compared."""
    entered = {"temperature_c": 11.32, "rain_mm": 900.0, "pesticides_t": 62163.2}
    assert api_client.changed_fields(entered, FRA_DEFAULTS) == {"rain_mm": 900.0}


# --------------------------------------------------------------------------- #
# Calls and errors
# --------------------------------------------------------------------------- #

def test_url_read_from_the_environment(monkeypatch, calls):
    recorded, responses = calls
    monkeypatch.setenv("API_URL", "http://api:8000/")
    responses.append(FakeResponse(200, {"status": "ok"}))
    api_client.health()
    assert recorded[0]["url"] == "http://api:8000/health"


def test_recommend_sends_country_criterion_and_changes(calls):
    recorded, responses = calls
    responses.append(FakeResponse(200, {"ranking": []}))
    api_client.recommend("FRA", "gross_product", {"temperature_c": 13.0})
    assert recorded[0]["json"] == {"iso3": "FRA", "criterion": "gross_product", "temperature_c": 13.0}
    assert recorded[0]["timeout"] == api_client.COMPUTE_TIMEOUT


def test_predict_sends_country_crop_and_changes(calls):
    recorded, responses = calls
    responses.append(FakeResponse(200, {"yield_t_ha": 7.0}))
    api_client.predict("FRA", "Wheat", {"rain_mm": 700.0})
    assert recorded[0]["url"].endswith("/predict")
    assert recorded[0]["json"] == {"iso3": "FRA", "crop": "Wheat", "rain_mm": 700.0}


def test_api_unreachable(calls):
    _, responses = calls
    responses.append(requests.ConnectionError("refus"))
    with pytest.raises(api_client.APIError, match="ne répond pas"):
        api_client.countries()


def test_timeout(calls):
    _, responses = calls
    responses.append(requests.Timeout("lent"))
    with pytest.raises(api_client.APIError, match="à temps"):
        api_client.predict("FRA", "Wheat")


def test_404_returns_the_api_message(calls):
    _, responses = calls
    responses.append(FakeResponse(404, {"detail": "ZZZ n'est pas un pays servi."}))
    with pytest.raises(api_client.APIError) as error:
        api_client.country_context("ZZZ")
    assert error.value.status == 404
    assert "ZZZ" in error.value.message


def test_422_is_readable(calls):
    _, responses = calls
    responses.append(FakeResponse(422, {"detail": [
        {"loc": ["body", "rain_mm"], "msg": "Input should be greater than or equal to 0"}]}))
    with pytest.raises(api_client.APIError, match="rain_mm : Input should be"):
        api_client.predict("FRA", "Wheat", {"rain_mm": -1})


def test_422_with_a_sentence_is_shown_as_is(calls):
    """A crop absent from the data is refused by the API in plain French: keep it."""
    _, responses = calls
    responses.append(FakeResponse(422, {"detail": "Orge : absente des données FAO."}))
    with pytest.raises(api_client.APIError, match="^Orge : absente des données FAO.$"):
        api_client.predict("FRA", "Barley")


def test_503_during_startup(calls):
    _, responses = calls
    responses.append(FakeResponse(503, {"detail": "modèles non chargés"}))
    with pytest.raises(api_client.APIError, match="réessayez"):
        api_client.health()
