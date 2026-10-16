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

Usage: `python scripts/verifie_notebooks.py [pattern]`
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat

RACINE = Path(__file__).resolve().parents[1]
NOTEBOOKS = RACINE / "notebooks"


def verifie(chemin: Path) -> list[str]:
    """Return the list of a notebook's anomalies. Empty list = all good."""
    nb = nbformat.read(str(chemin), as_version=4)
    cellules = [c for c in nb.cells if c.cell_type == "code"]
    anomalies = []

    for i, cellule in enumerate(nb.cells):
        if cellule.cell_type != "code":
            continue
        try:
            compile(cellule.source, "<cellule>", "exec")
        except SyntaxError as erreur:
            anomalies.append(
                f"cellule {i} : syntaxe invalide ligne {erreur.lineno} — {erreur.msg}"
            )

    non_executees = [c for c in cellules if not c.get("execution_count")]
    if non_executees:
        anomalies.append(
            f"{len(non_executees)}/{len(cellules)} cellules de code n'ont pas ete executees"
        )

    for i, cellule in enumerate(nb.cells):
        for sortie in cellule.get("outputs", []):
            if sortie.output_type == "error":
                anomalies.append(
                    f"cellule {i} : {sortie.ename} — {str(sortie.evalue)[:120]}"
                )

    return anomalies


def main() -> int:
    motif = sys.argv[1] if len(sys.argv) > 1 else "*.ipynb"
    chemins = sorted(NOTEBOOKS.glob(motif))
    if not chemins:
        print(f"aucun notebook pour le motif {motif!r}")
        return 1

    total = 0
    for chemin in chemins:
        anomalies = verifie(chemin)
        total += len(anomalies)
        marque = "OK  " if not anomalies else "ECHEC"
        print(f"[{marque}] {chemin.name}")
        for anomalie in anomalies:
            print(f"         {anomalie}")

    print()
    print(f"{len(chemins)} notebooks verifies, {total} anomalies")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
