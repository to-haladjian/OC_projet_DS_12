"""Runs the API inside the interface's own process, for single-service hosting.

Streamlit Community Cloud runs one Python process from the repository and exposes one
port: it cannot host the API alongside the interface. And no free host is left for a
container of this size — Hugging Face now requires a paid plan to *create* a Docker
Space, even though CPU Basic hardware itself still costs nothing.

The answer is not to reimplement the models in the interface. `api.main:app` is started
here on a loopback port, and `api_client` queries it over HTTP exactly as it always
does: same application object, same Pydantic validation, same status codes, same
warnings — only the hop is local. The interface still holds no ML logic.

Enabled by setting `API_EMBEDDED=1` and leaving `API_URL` unset. Docker Compose and
`app/Dockerfile` do neither: they have a real API next door, and the interface image
deliberately contains no model.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HOST = "127.0.0.1"

#: Starting the API means loading the 39 MB forest: a few seconds locally, more on a
#: cold free host. Generous, because exceeding it is a hard failure for the interface.
STARTUP_TIMEOUT = 120.0

#: `app/` is not a package — Streamlit puts the script's folder on the path, not the
#: repository root, so importing `api.main` has to be made possible explicitly.
ROOT = Path(__file__).resolve().parents[1]

_lock = threading.Lock()
_address: str | None = None


def _free_port() -> int:
    """A port the kernel just confirmed free.

    There is a window between closing this socket and uvicorn binding it, but both ends
    live inside one container on loopback, where nothing else competes for ports.
    """
    with socket.socket() as sonde:
        sonde.bind((HOST, 0))
        return sonde.getsockname()[1]


def _wait_until_ready(address: str, timeout: float) -> None:
    """Block until `/health` answers, or give up with a readable error.

    Uvicorn only opens its socket once the lifespan has loaded the models, so the
    connection is refused rather than slow while the forest is being read.
    """
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{address}/health", timeout=5) as answer:
                if answer.status == 200:
                    return
                last = f"statut {answer.status}"
        except (urllib.error.URLError, OSError) as erreur:
            last = str(erreur)
        time.sleep(0.5)
    raise RuntimeError(f"L'API interne n'a pas démarré en {timeout:.0f} s ({last}).")


def start() -> str:
    """Start the API once and return its base address; later calls reuse it.

    Streamlit replays the whole script on every interaction, so this has to be
    idempotent — hence the module-level address, guarded by a lock because several
    sessions may share one process.
    """
    global _address
    with _lock:
        if _address is not None:
            return _address

        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        import uvicorn

        from api.main import app as application

        port = _free_port()
        server = uvicorn.Server(uvicorn.Config(
            application, host=HOST, port=port, log_level="warning",
        ))
        # A daemon thread: it must not keep the interface alive on shutdown. Uvicorn
        # skips installing signal handlers outside the main thread, so this is safe.
        threading.Thread(target=server.run, name="embedded-api", daemon=True).start()

        address = f"http://{HOST}:{port}"
        _wait_until_ready(address, STARTUP_TIMEOUT)
        _address = address
        return address
