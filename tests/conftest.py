"""Shared fixtures: models are loaded once for the whole test session (~3 s)."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def resources(client):
    return client.app.state.resources


@pytest.fixture
def parcel():
    return {"crop": "Wheat", "rain_mm": 600, "temperature_c": 22,
            "fertilizer": True, "irrigation": False}


@pytest.fixture(scope="session")
def coefficients(resources):
    return resources.practice_meta["coefficients"], resources.practice_meta["intercept"]

