"""Loading and preparation of the Agriculture CropYield dataset.

Preparation is split into two levels, so that the exploratory analysis never has to
apply a decision it is precisely meant to inform:

* :func:`fix_types` only fixes types (booleans stored as text). No row is touched:
  the EDA and PCA notebooks use it.
* :func:`clean` additionally applies the decisions deferred by the EDA — removal
  of negative yields. Only the build notebook uses it.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.data.prep_log import PrepLog

BOOLEAN_COLUMNS = ("Fertilizer_Used", "Irrigation_Used")
TARGET = "Yield_tons_per_hectare"

CATEGORICAL_COLUMNS = ("Region", "Soil_Type", "Crop", "Weather_Condition")
CONTINUOUS_COLUMNS = ("Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest")


def load(path=None) -> pd.DataFrame:
    """Read the raw dataset as is, without any transformation.

    Booleans stay as text and negative yields are kept: this is exactly what the
    exploratory analysis must be able to observe.
    """
    return pd.read_csv(
        path or config.AGRICULTURE_CROP_YIELD,
        dtype={
            "Region": "category",
            "Soil_Type": "category",
            "Crop": "category",
            "Weather_Condition": "category",
            "Days_to_Harvest": "int16",
        },
    )


def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert booleans stored as text, without removing any row.

    `Fertilizer_Used` and `Irrigation_Used` arrive as the strings `'True'` /
    `'False'`. This is a type fix, not an analysis choice.
    """
    out = df.copy()
    for column in BOOLEAN_COLUMNS:
        if out[column].dtype != bool:
            out[column] = out[column].map({"True": True, "False": False}).astype(bool)
    return out


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, PrepLog]:
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
    prep_log = PrepLog()
    n_start = len(df)

    out = fix_types(df)
    prep_log.add("typage", "Fertilizer_Used, Irrigation_Used : texte -> booleen")

    duplicates = int(out.duplicated().sum())
    prep_log.add(
        "doublons stricts",
        f"{duplicates} — conserves le cas echeant (parcelles simulees distinctes)",
    )

    negative = out[TARGET] < 0
    n_negative = int(negative.sum())
    out = out.loc[~negative].reset_index(drop=True)
    prep_log.add(
        "rendements negatifs",
        f"{n_negative} lignes supprimees ({n_negative / n_start:.4%}) — "
        "physiquement impossibles",
    )

    prep_log.add("variables conservees", "Days_to_Harvest gardee (choix differe a l'etape 2)")
    prep_log.add("valeurs manquantes", f"{int(out.isna().sum().sum())} au total")
    prep_log.add("lignes", f"{n_start:,} -> {len(out):,}")
    return out, prep_log
