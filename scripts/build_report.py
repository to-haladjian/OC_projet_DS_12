"""Build the business report: its two generated figures, then the PDF.

    python scripts/build_report.py        (or `make report`)

The report is written in Markdown (`reports/rapport_metier.md`) so that it stays easy to
edit and to review; this script renders it to `reports/rapport_metier.pdf` with tools
that need no system package: `markdown` for the HTML, `xhtml2pdf` for the PDF, and the
DejaVu font shipped with matplotlib so that every French character is embedded.

Two figures are regenerated from the project's own outputs before rendering, so the
report cannot drift from the measurements:

* the comparison of model families, read from `reports/model_comparison.csv`
  (written by `make compare`);
* the pipeline diagram.
"""

from __future__ import annotations

import sys
from pathlib import Path

import markdown
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from xhtml2pdf import pisa  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SOURCE = REPORTS / "rapport_metier.md"
TARGET = REPORTS / "rapport_metier.pdf"
COMPARISON = REPORTS / "model_comparison.csv"

# Reference palette: categorical slots 1 and 2, text and grid inks.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

FAMILY_LABELS = {
    "ExtraTreesRegressor": "Arbres extrêmement aléatoires (servi)",
    "RandomForestRegressor": "Forêt aléatoire",
    "HistGradientBoostingRegressor": "Gradient boosting",
    "Ridge": "Régression linéaire (Ridge)",
    "LinearRegression": "Culture seule (référence basse)",
}
PROTOCOL_LABELS = {
    "temporal": "Pays connu, année suivante",
    "leave_country_out": "Pays jamais vu",
}


def family_comparison_figure() -> Path:
    """Error of each candidate family under the two protocols, same folds."""
    runs = pd.read_csv(COMPARISON)
    runs = runs[(runs["experiment"] == "agritech-yield-engine") & (runs["phase"] == "reference")
                & runs["run"].str.endswith(tuple(PROTOCOL_LABELS))]
    # The same candidate may have been tracked more than once: keep one row each.
    runs = runs.drop_duplicates(["model_family", "feature_set", "protocol"])
    table = runs.pivot(index="model_family", columns="protocol", values="cv_rmse_mean")
    table = table.loc[[f for f in FAMILY_LABELS if f in table.index]].sort_values("temporal")

    fig, ax = plt.subplots(figsize=(8.2, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    height = 0.36
    positions = range(len(table))
    for offset, (protocol, colour) in zip((-height / 2, height / 2),
                                          (("temporal", BLUE), ("leave_country_out", ORANGE))):
        values = table[protocol]
        bars = ax.barh([p + offset for p in positions], values, height=height - 0.04,
                       color=colour, label=PROTOCOL_LABELS[protocol])
        for bar, value in zip(bars, values):
            ax.text(value + 0.07, bar.get_y() + bar.get_height() / 2,
                    f"{value:.2f}".replace(".", ","), va="center", fontsize=8.5, color=INK)
    ax.set_yticks(list(positions), [FAMILY_LABELS[f] for f in table.index], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Erreur typique (RMSE, t/ha) — plus court est meilleur", fontsize=9, color=MUTED)
    ax.set_xlim(0, table.max().max() * 1.12)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5, length=0)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    # Above the plot: inside it, the legend would sit on the longest bars.
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
              labelcolor=INK, borderaxespad=0.2)
    path = FIGURES / "rapport_comparaison_familles.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def pipeline_figure() -> Path:
    """The chain from raw data to the published image, as boxes and arrows."""
    stages = [
        ("Données brutes", "2 jeux fournis\n+ prix FAOSTAT"),
        ("Préparation", "make data\nfusion contrôlée"),
        ("Entraînement", "make train\nsuivi MLflow"),
        ("Tests", "ruff + pytest\n112 tests"),
        ("Image de l'API", "docker build\ncontrôle /health"),
        ("Publication", "registre GHCR\npar commit"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 1.9), facecolor="white")
    ax.set_xlim(0, len(stages) * 1.55)
    ax.set_ylim(0, 1.3)
    ax.axis("off")
    for i, (title, detail) in enumerate(stages):
        x = i * 1.55 + 0.08
        local = i < 3
        ax.add_patch(FancyBboxPatch(
            (x, 0.2), 1.25, 0.9, boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor="#eaf2fc" if local else "#fdeee7", edgecolor=BLUE if local else ORANGE,
            linewidth=1.2))
        ax.text(x + 0.625, 0.82, title, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=INK)
        ax.text(x + 0.625, 0.47, detail, ha="center", va="center", fontsize=8, color=MUTED)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + 1.53, 0.65), xytext=(x + 1.29, 0.65),
                        arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.2})
    ax.text(0.08, 0.04, "en local, rejouable par make", fontsize=8, color=BLUE)
    ax.text(3 * 1.55 + 0.08, 0.04, "automatique à chaque push (GitHub Actions)", fontsize=8,
            color=ORANGE)
    path = FIGURES / "rapport_pipeline.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def stylesheet() -> str:
    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    return f"""
    @font-face {{ font-family: Report; src: url("{fonts / 'DejaVuSans.ttf'}"); }}
    @font-face {{ font-family: Report; font-weight: bold; src: url("{fonts / 'DejaVuSans-Bold.ttf'}"); }}
    @font-face {{ font-family: Report; font-style: italic; src: url("{fonts / 'DejaVuSans-Oblique.ttf'}"); }}
    @page {{ size: A4; margin: 2cm 2cm 2.2cm 2cm;
             @frame footer {{ -pdf-frame-content: footer; bottom: 1cm; margin-left: 2cm;
                              margin-right: 2cm; height: 0.8cm; }} }}
    body {{ font-family: Report; font-size: 10pt; line-height: 1.45; color: {INK}; }}
    h1 {{ font-size: 20pt; color: {INK}; margin: 0 0 4pt 0; }}
    h2 {{ font-size: 13.5pt; color: #1c5cab; margin: 16pt 0 5pt 0; -pdf-keep-with-next: true; }}
    h3 {{ font-size: 11pt; margin: 10pt 0 3pt 0; -pdf-keep-with-next: true; }}
    p {{ margin: 0 0 6pt 0; }}
    li {{ margin-bottom: 2pt; }}
    table {{ width: 100%; margin: 4pt 0 9pt 0; }}
    th {{ background-color: #eaf2fc; font-weight: bold; text-align: left; padding: 4pt 5pt 2pt 5pt;
          border-bottom: 0.8pt solid #86b6ef; font-size: 9pt; }}
    td {{ padding: 3pt 5pt 1pt 5pt; border-bottom: 0.4pt solid {GRID}; font-size: 9pt;
          vertical-align: top; }}
    img {{ margin: 4pt 0 2pt 0; }}
    em {{ color: {MUTED}; }}
    blockquote {{ margin: 6pt 0; padding: 5pt 9pt; background-color: #f4f8fe;
                  border-left: 2.5pt solid {BLUE}; }}
    code {{ font-family: Report; font-size: 9pt; color: #1c5cab; }}
    #footer {{ font-size: 8pt; color: {MUTED}; text-align: center; }}
    """


def render() -> None:
    body = markdown.markdown(SOURCE.read_text(encoding="utf-8"), extensions=["tables", "attr_list"])
    html = (
        f"<html><head><meta charset='utf-8'><style>{stylesheet()}</style></head><body>"
        f"{body}<div id='footer'>Agritech Answers — rapport métier — page <pdf:pagenumber/>"
        f"</div></body></html>"
    )

    def resolve(uri: str, _rel: str) -> str:
        """Image paths in the Markdown are relative to `reports/`."""
        path = Path(uri)
        return str(path if path.is_absolute() else (REPORTS / uri).resolve())

    with TARGET.open("wb") as target:
        status = pisa.CreatePDF(html, dest=target, link_callback=resolve, encoding="utf-8")
    if status.err:
        raise SystemExit(f"echec du rendu PDF ({status.err} erreurs)")


def main() -> int:
    if not COMPARISON.exists():
        print(f"{COMPARISON.relative_to(ROOT)} absent : lancer `make compare` d'abord")
        return 1
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (family_comparison_figure(), pipeline_figure()):
        print(f"ecrit {path.relative_to(ROOT)}")
    missing = [
        line.split("](")[1].split(")")[0]
        for line in SOURCE.read_text(encoding="utf-8").splitlines()
        if line.startswith("![") and not (REPORTS / line.split("](")[1].split(")")[0]).exists()
    ]
    if missing:
        print(f"images absentes, citees par le rapport : {missing}")
        return 1
    render()
    print(f"ecrit {TARGET.relative_to(ROOT)} ({TARGET.stat().st_size / 1024:.0f} Ko)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
