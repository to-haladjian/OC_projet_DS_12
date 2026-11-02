"""Entrypoint for Streamlit Community Cloud. The interface itself lives in `app/app.py`.

This file exists for one reason: Community Cloud picks the dependency file it finds in
the *entrypoint's own directory* before the one at the repository root. With `app/app.py`
as the entrypoint it would read `app/requirements.txt` — streamlit and requests only —
and the API started inside the process (see `app/embedded_api.py`) would die on
`import uvicorn`. Pointing the host at a root entrypoint makes the root
`requirements.txt`, which declares both halves, the file it reads.

`app/requirements.txt` therefore stays what step 4 says it is: the interface's own
dependencies, with no model and no scikit-learn, which is what `app/Dockerfile` installs.

Everything else runs the interface unchanged: `app/` goes on the path, because `app.py`
imports `api_client` as a sibling, and Streamlit only puts the entrypoint's directory
there.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INTERFACE = ROOT / "app" / "app.py"

if str(ROOT / "app") not in sys.path:
    sys.path.insert(0, str(ROOT / "app"))

# Streamlit re-executes the entrypoint on every interaction, and so does this: the
# interface is a script, not a module with a main(), exactly as `streamlit run
# app/app.py` would execute it.
runpy.run_path(str(INTERFACE), run_name="__main__")
