"""HTTP client of the Agritech Answers API — the interface's only contact with the models.

No Streamlit here, and no ML: this module sends what the user typed and returns what
the API answered. It is therefore testable on its own.

The API address comes from the `API_URL` environment variable (set by Docker Compose,
or by Streamlit secrets once deployed), `http://localhost:8000` by default. Where no
API can be hosted next to the interface, `API_EMBEDDED=1` starts one in this very
process and points the client at it — see `embedded_api`.
"""

from __future__ import annotations

import os

import requests

DEFAULT_URL = "http://localhost:8000"

#: Reads answer instantly; computations load the forest on the API side.
READ_TIMEOUT, COMPUTE_TIMEOUT = 5, 15

#: Below this gap, a prefilled value is considered unchanged (slider rounding).
TOLERANCE = 1e-6


class APIError(Exception):
    """An API call failed; `message` is meant to be shown to the user as is."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.message = message
        self.status = status


def embedded_mode() -> bool:
    """True when the API is to run inside this process rather than being reached.

    `API_URL` wins: an explicit address always means a real API somewhere else.
    """
    if os.environ.get("API_URL"):
        return False
    return os.environ.get("API_EMBEDDED", "").strip().lower() in {"1", "true", "oui"}


def api_base_url() -> str:
    if embedded_mode():
        import embedded_api

        return embedded_api.start()
    return os.environ.get("API_URL", DEFAULT_URL).rstrip("/")


def _detail_422(body: dict) -> str:
    """Pydantic validation errors, one readable line each."""
    rows = []
    for error in body.get("detail", []):
        field_name = ".".join(str(p) for p in error.get("loc", []) if p != "body")
        rows.append(f"{field_name} : {error.get('msg', 'valeur invalide')}")
    return "Saisie refusée par l'API — " + " ; ".join(rows) if rows else "Saisie refusée par l'API."


def _call(method: str, path: str, timeout: float, body: dict | None = None):
    url = f"{api_base_url()}{path}"
    try:
        response = requests.request(method, url, json=body, timeout=timeout)
    except requests.Timeout as error:
        raise APIError(f"L'API n'a pas répondu à temps ({timeout} s) : {url}") from error
    except requests.ConnectionError as error:
        # In embedded mode, telling the user to start a server would be nonsense: the
        # server is this process, so its thread has died.
        hint = ("Le service interne s'est arrêté : rechargez la page."
                  if embedded_mode()
                  else "Est-elle démarrée ? (`uvicorn api.main:app` ou `docker compose up`)")
        raise APIError(f"L'API ne répond pas à {api_base_url()}. {hint}") from error

    if response.status_code == 422:
        raise APIError(_detail_422(response.json()), 422)
    if response.status_code == 404:
        raise APIError(response.json().get("detail", "Ressource introuvable."), 404)
    if response.status_code == 503:
        raise APIError("L'API démarre et charge ses modèles : réessayez dans quelques secondes.", 503)
    if not response.ok:
        raise APIError(f"Erreur de l'API ({response.status_code}).", response.status_code)
    return response.json()


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

def health() -> dict:
    return _call("GET", "/health", READ_TIMEOUT)


def predict_context() -> dict:
    return _call("GET", "/predict/context", READ_TIMEOUT)


def predict(crop: str, rain_mm: float, temperature_c: float,
            fertilizer: bool, irrigation: bool) -> dict:
    return _call("POST", "/predict", COMPUTE_TIMEOUT, {
        "crop": crop, "rain_mm": rain_mm, "temperature_c": temperature_c,
        "fertilizer": fertilizer, "irrigation": irrigation,
    })


def countries() -> list[dict]:
    return _call("GET", "/recommend/countries", READ_TIMEOUT)


def country_context(iso3: str) -> dict:
    return _call("GET", f"/recommend/countries/{iso3}", READ_TIMEOUT)


def recommend(iso3: str, criterion: str, modifications: dict | None = None) -> dict:
    """Ranking for a country; `modifications` only holds what the user changed."""
    return _call("POST", "/recommend", COMPUTE_TIMEOUT,
                  {"iso3": iso3, "criterion": criterion, **(modifications or {})})


# --------------------------------------------------------------------------- #
# Only send what the user changed
# --------------------------------------------------------------------------- #

def _differs(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) > TOLERANCE * max(1.0, abs(float(b)))


def changed_fields(entered: dict, defaults: dict) -> dict:
    """The fields of a `/recommend` form that differ from the country's defaults.

    `entered` holds `temperature_c`, `rain_mm`, `pesticides_t` and `price_usd_t`
    (`{culture: price}`); `defaults` is the answer of `GET /recommend/countries/{iso3}`.
    Sending only what changed keeps `entered_values` in the API answer truthful.
    """
    modifications = {
        field_name: entered[field_name]
        for field_name in ("temperature_c", "rain_mm", "pesticides_t")
        if _differs(entered[field_name], defaults[field_name])
    }
    default_prices = {p["crop"]: p["price_usd_t"] for p in defaults["prices"]}
    prices = {
        crop: value
        for crop, value in entered.get("price_usd_t", {}).items()
        if _differs(value, default_prices[crop])
    }
    if prices:
        modifications["price_usd_t"] = prices
    return modifications
