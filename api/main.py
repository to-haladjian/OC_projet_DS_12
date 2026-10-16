"""Agritech Answers API — step 3.

Serves the two models of step 2:

* `POST /predict`: one crop and the conditions of a plot -> expected yield;
* `POST /recommend`: a country and, optionally, its conditions and prices -> the crops,
  ranked by predicted yield (or dry matter, or gross product).

Plus the read-only endpoints the Streamlit interface needs to build its forms, since it
contains no ML logic. Interactive documentation: `/docs`.

Run locally: `uvicorn api.main:app --reload`.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import sklearn
from fastapi import Depends, FastAPI, HTTPException, Request

from api import __version__, service
from api.schemas import (
    ContextePays, ContextePredict, EntreePredict, EntreeRecommend, Pays, Sante,
    SortiePredict, SortieRecommend,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
journal = logging.getLogger("agritech.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and assets once, before serving the first request."""
    debut = time.perf_counter()
    app.state.ressources = service.charge_ressources()
    journal.info("modeles charges en %.1f s", time.perf_counter() - debut)
    yield
    app.state.ressources = None


app = FastAPI(
    title="Agritech Answers — API",
    version=__version__,
    description=(
        "Estimation du rendement d'une parcelle (`/predict`) et classement des cultures "
        "pour un pays (`/recommend`). Les valeurs inhabituelles sont acceptées et "
        "signalées dans `avertissements` ; les valeurs impossibles sont refusées (422)."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def journalise(request: Request, call_next):
    """One log line per request: method, path, status, latency. Stdout only."""
    debut = time.perf_counter()
    reponse = await call_next(request)
    journal.info(
        "%s %s -> %d (%.0f ms)", request.method, request.url.path,
        reponse.status_code, (time.perf_counter() - debut) * 1000,
    )
    return reponse


def ressources(request: Request) -> service.Ressources:
    """The loaded resources, or a 503 if the service is not ready."""
    chargees = getattr(request.app.state, "ressources", None)
    if chargees is None:
        raise HTTPException(status_code=503, detail="modèles non chargés")
    return chargees


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #

@app.get("/", include_in_schema=False)
def accueil() -> dict:
    return {"service": app.title, "version": __version__, "documentation": "/docs"}


@app.get("/health", response_model=Sante, tags=["service"])
def sante(r: service.Ressources = Depends(ressources)) -> dict:
    """Service state and versions. Used by the Docker healthcheck and the CI."""
    return {
        "statut": "ok",
        "version_api": __version__,
        "modeles": {
            "predict": r.meta_predict["modele"],
            "recommend": f"{r.meta_recommend['modele']} ({r.meta_recommend['n_estimators']} arbres)",
        },
        "scikit_learn": sklearn.__version__,
        "scikit_learn_entrainement": r.domaine["scikit_learn"],
    }


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #

@app.get("/predict/contexte", response_model=ContextePredict, tags=["predict"])
def predict_contexte(r: service.Ressources = Depends(ressources)) -> dict:
    """Crops, bounds and units, to build the `/predict` form."""
    return service.contexte_predict(r)


@app.post("/predict", response_model=SortiePredict, tags=["predict"])
def predict(entree: EntreePredict, r: service.Ressources = Depends(ressources)) -> dict:
    """Expected yield of one plot, with a 95 % interval."""
    return service.predit(r, entree)


# --------------------------------------------------------------------------- #
# /recommend
# --------------------------------------------------------------------------- #

@app.get("/recommend/pays", response_model=list[Pays], tags=["recommend"])
def recommend_pays(r: service.Ressources = Depends(ressources)) -> list[dict]:
    """The served countries."""
    return service.pays_servis(r)


@app.get("/recommend/pays/{iso3}", response_model=ContextePays, tags=["recommend"])
def recommend_contexte_pays(iso3: str, r: service.Ressources = Depends(ressources)) -> dict:
    """A country's default conditions and reference prices, to prefill the form."""
    try:
        return service.contexte_pays_affichable(r, iso3)
    except service.PaysInconnu as erreur:
        raise HTTPException(status_code=404, detail=erreur.args[0]) from erreur


@app.post("/recommend", response_model=SortieRecommend, tags=["recommend"])
def recommend(entree: EntreeRecommend, r: service.Ressources = Depends(ressources)) -> dict:
    """All rankable crops, sorted by predicted yield (default) or by `critere`."""
    try:
        return service.recommande(r, entree)
    except service.PaysInconnu as erreur:
        raise HTTPException(status_code=404, detail=erreur.args[0]) from erreur
