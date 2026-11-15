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
    CountryContext, PredictContext, PredictInput, RecommendInput, Country, Health,
    PredictOutput, RecommendOutput,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("agritech.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and assets once, before serving the first request."""
    started_at = time.perf_counter()
    app.state.resources = service.load_resources()
    logger.info("modeles charges en %.1f s", time.perf_counter() - started_at)
    yield
    app.state.resources = None


app = FastAPI(
    title="Agritech Answers — API",
    version=__version__,
    description=(
        "Estimation du rendement d'une parcelle (`/predict`) et classement des cultures "
        "pour un pays (`/recommend`). Les valeurs inhabituelles sont acceptées et "
        "signalées dans `warnings` ; les valeurs impossibles sont refusées (422)."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def log_request(request: Request, call_next):
    """One log line per request: method, path, status, latency. Stdout only."""
    started_at = time.perf_counter()
    response = await call_next(request)
    logger.info(
        "%s %s -> %d (%.0f ms)", request.method, request.url.path,
        response.status_code, (time.perf_counter() - started_at) * 1000,
    )
    return response


def resources(request: Request) -> service.Resources:
    """The loaded resources, or a 503 if the service is not ready."""
    loaded = getattr(request.app.state, "resources", None)
    if loaded is None:
        raise HTTPException(status_code=503, detail="modèles non chargés")
    return loaded


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #

@app.get("/", include_in_schema=False)
def home() -> dict:
    return {"service": app.title, "version": __version__, "documentation": "/docs"}


@app.get("/health", response_model=Health, tags=["service"])
def health(r: service.Resources = Depends(resources)) -> dict:
    """Service state and versions. Used by the Docker healthcheck and the CI."""
    return {
        "status": "ok",
        "api_version": __version__,
        "models": {
            "predict": r.practice_meta["model"],
            "recommend": f"{r.yield_meta['model']} ({r.yield_meta['n_estimators']} arbres)",
        },
        "scikit_learn": sklearn.__version__,
        "scikit_learn_training": r.domain["scikit_learn"],
    }


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #

@app.get("/predict/context", response_model=PredictContext, tags=["predict"])
def get_predict_context(r: service.Resources = Depends(resources)) -> dict:
    """Crops, bounds and units, to build the `/predict` form."""
    return service.predict_context(r)


@app.post("/predict", response_model=PredictOutput, tags=["predict"])
def predict(payload: PredictInput, r: service.Resources = Depends(resources)) -> dict:
    """Expected yield of one plot, with a 95 % interval."""
    return service.predict_yield(r, payload)


# --------------------------------------------------------------------------- #
# /recommend
# --------------------------------------------------------------------------- #

@app.get("/recommend/countries", response_model=list[Country], tags=["recommend"])
def get_countries(r: service.Resources = Depends(resources)) -> list[dict]:
    """The served countries."""
    return service.served_countries(r)


@app.get("/recommend/countries/{iso3}", response_model=CountryContext, tags=["recommend"])
def get_country_context(iso3: str, r: service.Resources = Depends(resources)) -> dict:
    """A country's default conditions and reference prices, to prefill the form."""
    try:
        return service.display_country_context(r, iso3)
    except service.UnknownCountry as error:
        raise HTTPException(status_code=404, detail=error.args[0]) from error


@app.post("/recommend", response_model=RecommendOutput, tags=["recommend"])
def recommend(payload: RecommendInput, r: service.Resources = Depends(resources)) -> dict:
    """All rankable crops, sorted by predicted yield (default) or by `criterion`."""
    try:
        return service.recommend(r, payload)
    except service.UnknownCountry as error:
        raise HTTPException(status_code=404, detail=error.args[0]) from error
