"""Small log of the decisions made during a data preparation.

Each cleaning function returns its result *and* a report, so that the notebooks
can display what was done rather than rewriting it by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PrepLog:
    """Sequence of steps (label, detail) describing a preparation."""

    entries: list[tuple[str, str]] = field(default_factory=list)

    def add(self, label: str, detail: str) -> None:
        self.entries.append((label, detail))

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries, columns=["action", "detail"])

    def __str__(self) -> str:
        return self.table().to_string(index=False)
