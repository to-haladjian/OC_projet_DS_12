"""Loading and preparation of the Agriculture CropYield dataset.

Preparation is split into two levels, so that the exploratory analysis never has to
apply a decision it is precisely meant to inform:

* :func:`typage` only fixes types (booleans stored as text). No row is touched:
  the EDA and PCA notebooks use it.
* :func:`nettoie` additionally applies the decisions deferred by the EDA — removal
  of negative yields. Only the build notebook uses it.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.data.rapport import Rapport

COLONNES_BOOLEENNES = ("Fertilizer_Used", "Irrigation_Used")
CIBLE = "Yield_tons_per_hectare"

COLONNES_CATEGORIELLES = ("Region", "Soil_Type", "Crop", "Weather_Condition")
COLONNES_CONTINUES = ("Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest")


def charge(chemin=None) -> pd.DataFrame:
    """Read the raw dataset as is, without any transformation.

    Booleans stay as text and negative yields are kept: this is exactly what the
    exploratory analysis must be able to observe.
    """
    return pd.read_csv(
        chemin or config.AGRICULTURE_CROP_YIELD,
        dtype={
            "Region": "category",
            "Soil_Type": "category",
            "Crop": "category",
            "Weather_Condition": "category",
            "Days_to_Harvest": "int16",
        },
    )


def typage(df: pd.DataFrame) -> pd.DataFrame:
    """Convert booleans stored as text, without removing any row.

    `Fertilizer_Used` and `Irrigation_Used` arrive as the strings `'True'` /
    `'False'`. This is a type fix, not an analysis choice.
    """
    out = df.copy()
    for colonne in COLONNES_BOOLEENNES:
        if out[colonne].dtype != bool:
            out[colonne] = out[colonne].map({"True": True, "False": False}).astype(bool)
    return out


def nettoie(df: pd.DataFrame) -> tuple[pd.DataFrame, Rapport]:
    """Apply the decisions deferred by the exploratory analysis.

    Two decisions, and their rationale:

    * **Negative yields removed.** A negative yield does not exist; it is the
      generator's additive noise dipping below zero. The affected rows all have
      neither fertilizer nor irrigation and almost all get under 200 mm of rain, which
      confirms the artificial floor. They make up 0.023 % of the data.
    * **`Days_to_Harvest` kept**, despite a null correlation with yield
      (r = -0.004). Dropping a variable belongs to modelling (step 2), not to
      preparation, and the brief warns against unjustified removals.
    """
    rapport = Rapport()
    n_depart = len(df)

    out = typage(df)
    rapport.ajoute("typage", "Fertilizer_Used, Irrigation_Used : texte -> booleen")

    doublons = int(out.duplicated().sum())
    rapport.ajoute(
        "doublons stricts",
        f"{doublons} — conserves le cas echeant (parcelles simulees distinctes)",
    )

    negatifs = out[CIBLE] < 0
    n_negatifs = int(negatifs.sum())
    out = out.loc[~negatifs].reset_index(drop=True)
    rapport.ajoute(
        "rendements negatifs",
        f"{n_negatifs} lignes supprimees ({n_negatifs / n_depart:.4%}) — "
        "physiquement impossibles",
    )

    rapport.ajoute("variables conservees", "Days_to_Harvest gardee (choix differe a l'etape 2)")
    rapport.ajoute("valeurs manquantes", f"{int(out.isna().sum().sum())} au total")
    rapport.ajoute("lignes", f"{n_depart:,} -> {len(out):,}")
    return out, rapport
