"""Cross-cutting helpers for the notebooks: figure style and quick profiling.

Gathers here what would otherwise be copied at the top of every notebook.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config

#: Sample size for figures. Plotting a million points is slow and adds nothing
#: visually; statistics are still computed on the full dataset.
ECHANTILLON_FIGURES = 100_000


def style() -> None:
    """Apply a common style to all figures."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.dpi"] = 110
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 40)


def enregistre(nom: str) -> None:
    """Save the current figure to `reports/figures`, for the business report."""
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(config.FIGURES / f"{nom}.png", bbox_inches="tight")


def profil(df: pd.DataFrame, nom: str = "") -> pd.DataFrame:
    """Profiling table: type, missing values, distinct values, example."""
    resultat = pd.DataFrame(
        {
            "type": df.dtypes.astype(str),
            "manquantes": df.isna().sum(),
            "manquantes_%": (100 * df.isna().mean()).round(2),
            "valeurs_uniques": df.nunique(),
            "exemple": [
                df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns
            ],
        }
    )
    resultat.index.name = f"colonne {nom}".strip()
    return resultat


def ecart_par_groupe(
    df: pd.DataFrame, groupe: str, cible: str, bornes: str | None = None, n_tranches: int = 8
) -> pd.DataFrame:
    """Mean of `cible` per level of `groupe`, with the maximum gap.

    If `bornes` is given, means are computed **under comparable conditions**,
    per bin of that continuous variable. This is what distinguishes an absence of
    mean effect from an absence of interaction.
    """
    if bornes is None:
        table = df.groupby(groupe, observed=True)[cible].mean().to_frame().T
    else:
        tranches = pd.cut(df[bornes], bins=n_tranches)
        table = df.pivot_table(
            index=tranches, columns=groupe, values=cible, aggfunc="mean", observed=True
        )
    table["ecart max"] = table.max(axis=1) - table.min(axis=1)
    return table.round(4)


def effet_levier(df: pd.DataFrame, levier: str, groupe: str, cible: str) -> pd.DataFrame:
    """Effect of a boolean lever on `cible`, level by level of `groupe`.

    Also returns how much that effect varies across levels: this is the interaction
    measure, expressed in the target's unit rather than as a p-value — with a
    million rows, everything is significant, only magnitude matters.
    """
    table = df.pivot_table(index=groupe, columns=levier, values=cible, observed=True)
    table["effet"] = table[True] - table[False]
    table.attrs["variation"] = table["effet"].max() - table["effet"].min()
    return table.round(4)
