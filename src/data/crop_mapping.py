"""Crop vocabulary harmonisation between the two source datasets.

Dataset A (``Agriculture Crop Yield``) and dataset B (``Crop Yield Prediction``)
name their crops differently and do not cover the same species. This module is
the single place where that reconciliation is declared, so the notebook, the
training pipeline and the serving API all agree on the vocabulary. 

Only 4 of the 6 crops in dataset A exist in dataset B. Barley and Cotton have no
FAO counterpart, so their enrichment is imputed from a documented donor crop and
flagged through ``crop_ref_source``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Dataset B reports yields in hectograms per hectare; 1 t/ha = 10 000 hg/ha.
HG_PER_HA_TO_T_PER_HA = 1 / 10_000

# Provenance flag carried by every row of the consolidated dataset.
SOURCE_FAO = "FAO"
SOURCE_IMPUTED = "imputed"

#: Crop name in dataset A -> ``Item`` label in dataset B (FAO vocabulary).
#: ``None`` means the crop is absent from B and must go through IMPUTATION_RULES.
CROP_TO_FAO_ITEM: dict[str, str | None] = {
    "Maize": "Maize",
    "Wheat": "Wheat",
    "Rice": "Rice, paddy",
    "Soybean": "Soybeans",
    "Barley": None,
    "Cotton": None,
}


@dataclass(frozen=True)
class CropImputation:
    """How to build an enrichment profile for a crop missing from dataset B.

    The agronomic profile (climate optimum, pesticide intensity, yield spread) is
    copied from ``donor_item``. ``yield_override_t_ha`` replaces the donor's
    reference yield when the donor is only a climatic analogue and not a
    plausible yield proxy.
    """

    donor_item: str
    yield_override_t_ha: float | None
    rationale: str


IMPUTATION_RULES: dict[str, CropImputation] = {
    "Barley": CropImputation(
        donor_item="Wheat",
        yield_override_t_ha=None,
        rationale=(
            "Barley is a temperate cereal with the same growing calendar, input "
            "requirements and yield magnitude as wheat. Wheat's FAO profile is "
            "transferred whole, yield included."
        ),
    ),
    "Cotton": CropImputation(
        donor_item="Sorghum",
        yield_override_t_ha=2.2,
        rationale=(
            "Cotton is a fibre crop: no cereal can proxy its yield, so the "
            "reference yield is set to the FAO world average for seed cotton "
            "(~2.2 t/ha). Sorghum -- a warm, semi-arid crop -- only donates the "
            "climatic and pesticide-intensity profile."
        ),
    ),
}


def fao_item_for(crop: str) -> str | None:
    """Return the dataset B ``Item`` label for a dataset A crop, if any."""
    return CROP_TO_FAO_ITEM[crop]


def source_for(crop: str) -> str:
    """Return the provenance flag (``FAO`` or ``imputed``) for a dataset A crop."""
    return SOURCE_FAO if CROP_TO_FAO_ITEM[crop] is not None else SOURCE_IMPUTED


def known_crops() -> list[str]:
    """The 6 crops the application can predict and rank."""
    return list(CROP_TO_FAO_ITEM)


def _validate() -> None:
    """Every crop absent from B must have an imputation rule, and vice versa."""
    missing = {c for c, item in CROP_TO_FAO_ITEM.items() if item is None}
    if missing != set(IMPUTATION_RULES):
        raise ValueError(
            f"crops without an FAO item {sorted(missing)} do not match "
            f"imputation rules {sorted(IMPUTATION_RULES)}"
        )


_validate()
