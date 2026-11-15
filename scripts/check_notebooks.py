"""Check that the notebooks are valid **and actually executed**.

This script exists because of a mistake made during step 2: a notebook whose
execution had failed on a syntax error was declared "error-free", because the
check only counted outputs of type `error`. But a notebook that never ran has
no output at all — hence no error. The absence of errors proves nothing unless
we first check that the notebook was actually executed.

Three checks, in this order:

1. **Syntax**: every code cell compiles. Catches generation errors before
   launching a twenty-minute execution for nothing.
2. **Execution**: every code cell carries an `execution_count`. This is the check
   that was missing.
3. **No errors**: no output of type `error`.

Usage: `python scripts/check_notebooks.py [pattern]`
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def check(path: Path) -> list[str]:
    """Return the list of a notebook's anomalies. Empty list = all good."""
    nb = nbformat.read(str(path), as_version=4)
    cells = [c for c in nb.cells if c.cell_type == "code"]
    anomalies = []

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        try:
            compile(cell.source, "<cellule>", "exec")
        except SyntaxError as error:
            anomalies.append(
                f"cellule {i} : syntaxe invalide ligne {error.lineno} — {error.msg}"
            )

    not_executed = [c for c in cells if not c.get("execution_count")]
    if not_executed:
        anomalies.append(
            f"{len(not_executed)}/{len(cells)} cellules de code n'ont pas ete executees"
        )

    for i, cell in enumerate(nb.cells):
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                anomalies.append(
                    f"cellule {i} : {output.ename} — {str(output.evalue)[:120]}"
                )

    return anomalies


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "*.ipynb"
    paths = sorted(NOTEBOOKS.glob(pattern))
    if not paths:
        print(f"aucun notebook pour le motif {pattern!r}")
        return 1

    total = 0
    for path in paths:
        anomalies = check(path)
        total += len(anomalies)
        flag = "OK  " if not anomalies else "ECHEC"
        print(f"[{flag}] {path.name}")
        for anomaly in anomalies:
            print(f"         {anomaly}")

    print()
    print(f"{len(paths)} notebooks verifies, {total} anomalies")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
