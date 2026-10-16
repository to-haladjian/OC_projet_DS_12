"""HTTP client of the Agritech Answers API — the interface's only contact with the models.

No Streamlit here, and no ML: this module sends what the user typed and returns what
the API answered. It is therefore testable on its own.

The API address comes from the `API_URL` environment variable (set by Docker Compose,
or by Streamlit secrets once deployed), `http://localhost:8000` by default.
"""

from __future__ import annotations

import os

import requests

URL_PAR_DEFAUT = "http://localhost:8000"

#: Reads answer instantly; computations load the forest on the API side.
DELAI_LECTURE, DELAI_CALCUL = 5, 15

#: Below this gap, a prefilled value is considered unchanged (slider rounding).
TOLERANCE = 1e-6


class ErreurAPI(Exception):
    """An API call failed; `message` is meant to be shown to the user as is."""

    def __init__(self, message: str, statut: int | None = None):
        super().__init__(message)
        self.message = message
        self.statut = statut


def url_api() -> str:
    return os.environ.get("API_URL", URL_PAR_DEFAUT).rstrip("/")


def _detail_422(corps: dict) -> str:
    """Pydantic validation errors, one readable line each."""
    lignes = []
    for erreur in corps.get("detail", []):
        champ = ".".join(str(p) for p in erreur.get("loc", []) if p != "body")
        lignes.append(f"{champ} : {erreur.get('msg', 'valeur invalide')}")
    return "Saisie refusée par l'API — " + " ; ".join(lignes) if lignes else "Saisie refusée par l'API."


def _appel(methode: str, chemin: str, delai: float, corps: dict | None = None):
    url = f"{url_api()}{chemin}"
    try:
        reponse = requests.request(methode, url, json=corps, timeout=delai)
    except requests.Timeout as erreur:
        raise ErreurAPI(f"L'API n'a pas répondu à temps ({delai} s) : {url}") from erreur
    except requests.ConnectionError as erreur:
        raise ErreurAPI(
            f"L'API ne répond pas à {url_api()}. Est-elle démarrée ? "
            "(`uvicorn api.main:app` ou `docker compose up`)"
        ) from erreur

    if reponse.status_code == 422:
        raise ErreurAPI(_detail_422(reponse.json()), 422)
    if reponse.status_code == 404:
        raise ErreurAPI(reponse.json().get("detail", "Ressource introuvable."), 404)
    if reponse.status_code == 503:
        raise ErreurAPI("L'API démarre et charge ses modèles : réessayez dans quelques secondes.", 503)
    if not reponse.ok:
        raise ErreurAPI(f"Erreur de l'API ({reponse.status_code}).", reponse.status_code)
    return reponse.json()


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

def sante() -> dict:
    return _appel("GET", "/health", DELAI_LECTURE)


def contexte_predict() -> dict:
    return _appel("GET", "/predict/contexte", DELAI_LECTURE)


def predire(culture: str, pluie_mm: float, temperature_c: float,
            engrais: bool, irrigation: bool) -> dict:
    return _appel("POST", "/predict", DELAI_CALCUL, {
        "culture": culture, "pluie_mm": pluie_mm, "temperature_c": temperature_c,
        "engrais": engrais, "irrigation": irrigation,
    })


def pays() -> list[dict]:
    return _appel("GET", "/recommend/pays", DELAI_LECTURE)


def contexte_pays(iso3: str) -> dict:
    return _appel("GET", f"/recommend/pays/{iso3}", DELAI_LECTURE)


def recommander(iso3: str, critere: str, modifications: dict | None = None) -> dict:
    """Ranking for a country; `modifications` only holds what the user changed."""
    return _appel("POST", "/recommend", DELAI_CALCUL,
                  {"iso3": iso3, "critere": critere, **(modifications or {})})


# --------------------------------------------------------------------------- #
# Only send what the user changed
# --------------------------------------------------------------------------- #

def _differe(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) > TOLERANCE * max(1.0, abs(float(b)))


def champs_modifies(saisie: dict, defauts: dict) -> dict:
    """The fields of a `/recommend` form that differ from the country's defaults.

    `saisie` holds `temperature_c`, `pluie_mm`, `pesticides_t` and `prix_usd_t`
    (`{culture: price}`); `defauts` is the answer of `GET /recommend/pays/{iso3}`.
    Sending only what changed keeps `valeurs_saisies` in the API answer truthful.
    """
    modifications = {
        champ: saisie[champ]
        for champ in ("temperature_c", "pluie_mm", "pesticides_t")
        if _differe(saisie[champ], defauts[champ])
    }
    prix_defaut = {p["culture"]: p["prix_usd_t"] for p in defauts["prix"]}
    prix = {
        culture: valeur
        for culture, valeur in saisie.get("prix_usd_t", {}).items()
        if _differe(valeur, prix_defaut[culture])
    }
    if prix:
        modifications["prix_usd_t"] = prix
    return modifications
