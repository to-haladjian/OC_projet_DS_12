"""Smoke test of the Streamlit interface against the real API.

The API is started on a free port in a background thread; the interface is run
headless with Streamlit's `AppTest` and pointed at it through `API_URL`.
"""

import socket
import threading
import time
from pathlib import Path

import pytest
import requests
import uvicorn
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")


@pytest.fixture(scope="module")
def api_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config("api.main:app", host="127.0.0.1", port=port,
                                            log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if requests.get(f"{url}/health", timeout=1).ok:
                break
        except requests.ConnectionError:
            pass
        time.sleep(0.2)
    else:
        pytest.fail("l'API de test n'a pas démarré")
    yield url
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def app(api_url, monkeypatch):
    monkeypatch.setenv("API_URL", api_url)
    return AppTest.from_file(APP, default_timeout=30)


def test_prediction(app):
    app.run()
    assert not app.exception
    app.button[0].click().run()          # the form's submit button, default values
    assert not app.exception
    yield_value = app.metric[0].value          # French format: "4,94 t/ha"
    assert yield_value.endswith("t/ha")
    assert float(yield_value.split()[0].replace("\u202f", "").replace(",", ".")) > 0


def test_recommendation_france(app):
    app.run()
    app.sidebar.radio[0].set_value("Recommandation").run()
    assert not app.exception
    assert app.selectbox[0].value == "FRA"
    app.button[0].click().run()
    assert not app.exception
    display_table = app.dataframe[0].value
    assert len(display_table) == 4
    assert display_table.iloc[0]["label"] == "Maïs"


def test_api_unreachable(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert not app.exception
    assert any("ne répond pas" in e.value for e in app.error)
