"""Shared fixtures: models are loaded once for the whole test session (~3 s)."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def ressources(client):
    return client.app.state.ressources


@pytest.fixture
def parcelle():
    return {"culture": "Wheat", "pluie_mm": 600, "temperature_c": 22,
            "engrais": True, "irrigation": False}


@pytest.fixture(scope="session")
def coefficients(ressources):
    return ressources.meta_predict["coefficients"], ressources.meta_predict["constante"]

