"""The embedded API: the interface serving itself when no host can run two services.

Two levels of checks. The routing — which environment variable wins — is tested in
process and costs nothing. Whether a uvicorn thread really comes up and answers can
only be proved by doing it, and that happens in a subprocess: `api.main:app` is a single
application object, so starting a server on the one the session's TestClient already
owns would reload the models underneath it and leave the other tests on a dead state.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import api_client  # noqa: E402

#: Run by the subprocess below: the real client, a real server, real HTTP calls.
PROGRAM = """
import json, sys
sys.path.insert(0, "app")
import api_client
print(json.dumps({
    "url": api_client.url_api(),
    "sante": api_client.sante(),
    "rendement": api_client.predire("Wheat", 600, 22, True, False)["rendement_t_ha"],
}))
"""

#: The whole interface, driven headless, exactly as Streamlit Cloud will run it: no
#: API_URL anywhere, both screens served by the API started inside the process.
PROGRAM_INTERFACE = """
import json, sys
from pathlib import Path
sys.path.insert(0, "app")
from streamlit.testing.v1 import AppTest

# AppTest resolves a relative path against its *caller*, which here is a -c program.
app = AppTest.from_file(str(Path.cwd() / "app" / "app.py"), default_timeout=180)
app.run()
assert not app.exception, app.exception
app.button[0].click().run()
assert not app.exception, app.exception
rendement = app.metric[0].value
erreurs = [e.value for e in app.error]

app.sidebar.radio[0].set_value("Recommandation").run()
app.button[0].click().run()
assert not app.exception, app.exception
tableau = app.dataframe[0].value
erreurs += [e.value for e in app.error]

print(json.dumps({"rendement": rendement, "erreurs": erreurs,
                  "cultures": len(tableau), "premiere": tableau.iloc[0]["libelle"]}))
"""


def _lance(programme):
    """Run a program against the repository, in embedded mode, and read its JSON."""
    environnement = {**os.environ, "API_EMBEDDED": "1"}
    environnement.pop("API_URL", None)

    acheve = subprocess.run(
        [sys.executable, "-c", programme],
        cwd=ROOT, env=environnement, capture_output=True, text=True, timeout=300,
    )
    assert acheve.returncode == 0, acheve.stderr
    return json.loads(acheve.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# Which address the client resolves to
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("valeur, attendu", [
    ("1", True), ("true", True), ("TRUE", True), ("oui", True), (" 1 ", True),
    ("0", False), ("", False), ("non", False),
])
def test_embedded_mode_reads_the_flag(monkeypatch, valeur, attendu):
    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.setenv("API_EMBEDDED", valeur)
    assert api_client.embedded_mode() is attendu


def test_embedded_mode_is_off_by_default(monkeypatch):
    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.delenv("API_EMBEDDED", raising=False)
    assert api_client.embedded_mode() is False
    assert api_client.url_api() == api_client.URL_PAR_DEFAUT


def test_api_url_wins_over_embedded_mode(monkeypatch):
    """An explicit address always means a real API elsewhere — Compose sets one."""
    monkeypatch.setenv("API_URL", "http://api:8000")
    monkeypatch.setenv("API_EMBEDDED", "1")
    assert api_client.embedded_mode() is False
    assert api_client.url_api() == "http://api:8000"


def test_url_api_delegates_to_the_embedded_module(monkeypatch):
    """In embedded mode the client asks the module for an address and uses it as is."""
    import embedded_api

    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.setenv("API_EMBEDDED", "1")
    monkeypatch.setattr(embedded_api, "start", lambda: "http://127.0.0.1:54321")
    assert api_client.url_api() == "http://127.0.0.1:54321"


def test_start_is_idempotent(monkeypatch):
    """Streamlit replays the script on every click; the server must start once only."""
    import embedded_api

    ouvertures = []
    monkeypatch.setattr(embedded_api, "_address", "http://127.0.0.1:54321")
    monkeypatch.setattr(embedded_api, "_free_port", lambda: ouvertures.append(1) or 0)
    assert embedded_api.start() == "http://127.0.0.1:54321"
    assert ouvertures == []


# --------------------------------------------------------------------------- #
# The server really starts and really answers
# --------------------------------------------------------------------------- #

def test_the_embedded_server_actually_answers():
    resultat = _lance(PROGRAM)
    assert resultat["url"].startswith("http://127.0.0.1:")
    # Not port 8000: the point is that nothing had to be started beforehand.
    assert not resultat["url"].endswith(":8000")
    assert resultat["sante"]["statut"] == "ok"
    assert resultat["sante"]["scikit_learn"] == resultat["sante"]["scikit_learn_entrainement"]
    assert 0 < resultat["rendement"] < 100


def test_the_interface_works_end_to_end_without_any_api_url():
    """The deployment that matters: both screens, served from inside the process."""
    resultat = _lance(PROGRAM_INTERFACE)
    assert resultat["erreurs"] == []
    assert resultat["rendement"].endswith("t/ha")
    assert resultat["cultures"] == 4
    assert resultat["premiere"] == "Maïs"
