"""Agritech Answers API.

One yield model, two questions asked of it:

* `POST /predict`: a country, one crop and, optionally, the conditions that differ from
  the country's own -> expected yield, with an interval;
* `POST /recommend`: the same conditions and, optionally, prices -> the crops, ranked
  by predicted yield (or dry matter, or gross product).

Both go through the same estimate (see `api.service`), so the same crop in the same
conditions gets the same yield from either endpoint.

Plus the read-only endpoints the Streamlit interface needs to build its forms, since it
contains no ML logic. Interactive documentation: `/docs`.

Run locally: `uvicorn api.main:app --reload` (or `make api`).
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import sklearn
from fastapi import Depends, FastAPI, HTTPException, Request

from api import __version__, service
from api.schemas import (
    Country,
    CountryContext,
    Health,
    PredictContext,
    PredictInput,
    PredictOutput,
    RecommendInput,
    RecommendOutput,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("agritech.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and the assets once, before serving the first request."""
    started_at = time.perf_counter()
    app.state.resources = service.load_resources()
    logger.info("modele charge en %.1f s", time.perf_counter() - started_at)
    # Nothing to release on shutdown, and the state is deliberately left in place: the
    # application object can be served by more than one server in a process (the test
    # suite does it), and one of them stopping must not blind the others.
    yield


app = FastAPI(
    title="Agritech Answers — API",
    version=__version__,
    description=(
        "Estimation du rendement d'une culture (`/predict`) et classement des cultures "
        "(`/recommend`) pour un pays et ses conditions, par un même modèle. Les valeurs "
        "inhabituelles sont acceptées et signalées dans `warnings` ; les valeurs "
        "impossibles sont refusées (422)."
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
        raise HTTPException(status_code=503, detail="modèle non chargé")
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
            "yield": f"{r.yield_meta['model']} ({r.yield_meta['n_estimators']} arbres)",
            "practice_effects": r.practice_meta["model"],
        },
        "scikit_learn": sklearn.__version__,
        "scikit_learn_training": r.domain["scikit_learn"],
    }


# --------------------------------------------------------------------------- #
# Countries
# --------------------------------------------------------------------------- #

@app.get("/countries", response_model=list[Country], tags=["countries"])
def get_countries(r: service.Resources = Depends(resources)) -> list[dict]:
    """The served countries."""
    return service.served_countries(r)


@app.get("/countries/{iso3}", response_model=CountryContext, tags=["countries"])
def get_country_context(iso3: str, r: service.Resources = Depends(resources)) -> dict:
    """A country's default conditions and reference prices, to prefill both forms."""
    try:
        return service.display_country_context(r, iso3)
    except service.UnknownCountry as error:
        raise HTTPException(status_code=404, detail=error.args[0]) from error


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #

@app.get("/predict/context", response_model=PredictContext, tags=["predict"])
def get_predict_context() -> dict:
    """Crops, bounds and units, to build the `/predict` form."""
    return service.predict_context()


@app.post("/predict", response_model=PredictOutput, tags=["predict"])
def predict(payload: PredictInput, r: service.Resources = Depends(resources)) -> dict:
    """Expected yield of one crop, with the interval observed in validation."""
    try:
        return service.predict(r, payload)
    except service.UnknownCountry as error:
        raise HTTPException(status_code=404, detail=error.args[0]) from error
    except service.UnavailableCrop as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


# --------------------------------------------------------------------------- #
# /recommend
# --------------------------------------------------------------------------- #

@app.post("/recommend", response_model=RecommendOutput, tags=["recommend"])
def recommend(payload: RecommendInput, r: service.Resources = Depends(resources)) -> dict:
    """All rankable crops, sorted by predicted yield (default) or by `criterion`."""
    try:
        return service.recommend(r, payload)
    except service.UnknownCountry as error:
        raise HTTPException(status_code=404, detail=error.args[0]) from error
