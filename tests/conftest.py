"""Shared fixtures: the model is loaded once for the whole test session (~3 s)."""

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
def wheat_in_france():
    return {"iso3": "FRA", "crop": "Wheat"}
