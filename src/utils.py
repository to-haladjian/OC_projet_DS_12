"""Cross-cutting helpers for the notebooks: figure style and quick profiling.

Gathers here what would otherwise be copied at the top of every notebook.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config

#: Sample size for figures. Plotting a million points is slow and adds nothing
#: visually; statistics are still computed on the full dataset.
FIGURE_SAMPLE = 100_000


def style() -> None:
    """Apply a common style to all figures."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.dpi"] = 110
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 40)


def save_figure(name: str) -> None:
    """Save the current figure to `reports/figures`, for the business report."""
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(config.FIGURES / f"{name}.png", bbox_inches="tight")


def profile(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """Profiling table: type, missing values, distinct values, example."""
    result = pd.DataFrame(
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
    result.index.name = f"colonne {name}".strip()
    return result


def gap_by_group(
    df: pd.DataFrame, group: str, target: str, bounds: str | None = None, n_bins: int = 8
) -> pd.DataFrame:
    """Mean of `target` per level of `group`, with the maximum gap.

    If `bounds` is given, means are computed **under comparable conditions**,
    per bin of that continuous variable. This is what distinguishes an absence of
    mean effect from an absence of interaction.
    """
    if bounds is None:
        table = df.groupby(group, observed=True)[target].mean().to_frame().T
    else:
        bins = pd.cut(df[bounds], bins=n_bins)
        table = df.pivot_table(
            index=bins, columns=group, values=target, aggfunc="mean", observed=True
        )
    table["ecart max"] = table.max(axis=1) - table.min(axis=1)
    return table.round(4)


def lever_effect(df: pd.DataFrame, lever: str, group: str, target: str) -> pd.DataFrame:
    """Effect of a boolean lever on `target`, level by level of `group`.

    Also returns how much that effect varies across levels: this is the interaction
    measure, expressed in the target's unit rather than as a p-value — with a
    million rows, everything is significant, only magnitude matters.
    """
    table = df.pivot_table(index=group, columns=lever, values=target, observed=True)
    table["effet"] = table[True] - table[False]
    table.attrs["variation"] = table["effet"].max() - table["effet"].min()
    return table.round(4)
