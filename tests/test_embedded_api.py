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
    "url": api_client.api_base_url(),
    "health": api_client.health(),
    "yield": api_client.predict("FRA", "Wheat")["yield_t_ha"],
}))
"""

#: The whole interface, driven headless, exactly as Streamlit Cloud will run it: the
#: root entrypoint, no API_URL anywhere, both screens served by the embedded API. Note
#: that nothing here puts `app/` on the path — `streamlit_app.py` has to do that itself,
#: as it must on the host.
PROGRAM_INTERFACE = """
import json
from pathlib import Path
from streamlit.testing.v1 import AppTest

# AppTest resolves a relative path against its *caller*, which here is a -c program.
app = AppTest.from_file(str(Path.cwd() / "streamlit_app.py"), default_timeout=180)
app.run()
assert not app.exception, app.exception
app.button[0].click().run()
assert not app.exception, app.exception
yield_text = app.metric[0].value
errors = [e.value for e in app.error]

app.sidebar.radio[0].set_value("Recommandation").run()
app.button[0].click().run()
assert not app.exception, app.exception
table = app.dataframe[0].value
errors += [e.value for e in app.error]

print(json.dumps({"yield": yield_text, "errors": errors,
                  "crops": len(table), "first": table.iloc[0]["label"]}))
"""


def _launch(program):
    """Run a program against the repository, in embedded mode, and read its JSON."""
    environment = {**os.environ, "API_EMBEDDED": "1"}
    environment.pop("API_URL", None)

    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# Which address the client resolves to
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value, expected", [
    ("1", True), ("true", True), ("TRUE", True), ("oui", True), (" 1 ", True),
    ("0", False), ("", False), ("non", False),
])
def test_embedded_mode_reads_the_flag(monkeypatch, value, expected):
    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.setenv("API_EMBEDDED", value)
    assert api_client.embedded_mode() is expected


def test_embedded_mode_is_off_by_default(monkeypatch):
    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.delenv("API_EMBEDDED", raising=False)
    assert api_client.embedded_mode() is False
    assert api_client.api_base_url() == api_client.DEFAULT_URL


def test_api_url_wins_over_embedded_mode(monkeypatch):
    """An explicit address always means a real API elsewhere — Compose sets one."""
    monkeypatch.setenv("API_URL", "http://api:8000")
    monkeypatch.setenv("API_EMBEDDED", "1")
    assert api_client.embedded_mode() is False
    assert api_client.api_base_url() == "http://api:8000"


def test_url_api_delegates_to_the_embedded_module(monkeypatch):
    """In embedded mode the client asks the module for an address and uses it as is."""
    import embedded_api

    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.setenv("API_EMBEDDED", "1")
    monkeypatch.setattr(embedded_api, "start", lambda: "http://127.0.0.1:54321")
    assert api_client.api_base_url() == "http://127.0.0.1:54321"


def test_start_is_idempotent(monkeypatch):
    """Streamlit replays the script on every click; the server must start once only."""
    import embedded_api

    openings = []
    monkeypatch.setattr(embedded_api, "_address", "http://127.0.0.1:54321")
    monkeypatch.setattr(embedded_api, "_free_port", lambda: openings.append(1) or 0)
    assert embedded_api.start() == "http://127.0.0.1:54321"
    assert openings == []


# --------------------------------------------------------------------------- #
# The server really starts and really answers
# --------------------------------------------------------------------------- #

def test_the_embedded_server_actually_answers():
    result = _launch(PROGRAM)
    assert result["url"].startswith("http://127.0.0.1:")
    # Not port 8000: the point is that nothing had to be started beforehand.
    assert not result["url"].endswith(":8000")
    assert result["health"]["status"] == "ok"
    assert result["health"]["scikit_learn"] == result["health"]["scikit_learn_training"]
    assert 0 < result["yield"] < 100


def test_the_interface_works_end_to_end_without_any_api_url():
    """The deployment that matters: both screens, served from inside the process."""
    result = _launch(PROGRAM_INTERFACE)
    assert result["errors"] == []
    assert result["yield"].endswith("t/ha")
    assert result["crops"] == 4
    assert result["first"] == "Maïs"


def test_the_root_requirements_declare_both_halves():
    """Community Cloud reads this one file, so it has to pull in the API's half too.

    A guard, not a formality: dropping `api/requirements.txt` from it would install the
    interface alone and the embedded API would fail on `import uvicorn`, on the host and
    nowhere else.
    """
    rows = {line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()}
    assert "-r app/requirements.txt" in rows
    assert "-r api/requirements.txt" in rows


def test_nothing_at_the_root_outranks_the_requirements_file():
    """Community Cloud's order is uv.lock, Pipfile, environment.yml, then requirements.

    `uv lock` writes a lock file at the root, and committing it would silently hand the
    deploy a resolution made from pyproject.toml — which does not list streamlit among
    its default dependencies. The failure would appear only on the host.
    """
    higher_priority = [f for f in ("uv.lock", "Pipfile", "environment.yml", "environment.yaml")
                    if (ROOT / f).exists()]
    assert higher_priority == [], (
        f"{higher_priority} passe(nt) avant requirements.txt sur Streamlit Cloud ; "
        "ne pas versionner ce fichier, ou déclarer les dépendances dedans."
    )
