"""Small log of the decisions made during a data preparation.

Each cleaning function returns its result *and* a report, so that the notebooks
can display what was done rather than rewriting it by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Rapport:
    """Sequence of steps (label, detail) describing a preparation."""

    etapes: list[tuple[str, str]] = field(default_factory=list)

    def ajoute(self, libelle: str, detail: str) -> None:
        self.etapes.append((libelle, detail))

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.etapes, columns=["etape", "detail"])

    def __str__(self) -> str:
        return self.table().to_string(index=False)
