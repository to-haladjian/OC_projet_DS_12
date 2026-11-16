"""Feature sets of the two models, declared once.

This module is the single reference: the notebooks, the training scripts and the API
draw the same lists from it. Do not redefine a feature set elsewhere.

It also holds the **dry matter table**, which makes yields comparable across crops
(see :data:`DRY_MATTER`).
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Practice-effects model — simulated dataset
# --------------------------------------------------------------------------- #

PRACTICE_TARGET = "Yield_tons_per_hectare"

#: The 4 variables carrying the signal, established by the exploration and confirmed by the
#: ablation of notebook 07. They are also the 4 the interface will ask for.
PRACTICE_SELECTED = [
    "Rainfall_mm",
    "Temperature_Celsius",
    "Fertilizer_Used",
    "Irrigation_Used",
]

#: Variables with no measurable effect, kept in the dataset but excluded from the model.
PRACTICE_NO_EFFECT = ["Days_to_Harvest"]
PRACTICE_CATEGORICAL = ["Region", "Soil_Type", "Crop", "Weather_Condition"]

#: Enrichment columns from the FAO dataset. The ablation shows they add nothing to this
#: target: the simulated yield does not depend on the crop, and they are collinear with
#: `Crop`. They stay in the consolidated dataset as documentation.
PRACTICE_ENRICHMENT = [
    "crop_ref_yield_t_ha",
    "crop_ref_yield_p25",
    "crop_ref_yield_p75",
    "crop_pesticides_intensity",
    "crop_temp_niche",
    "crop_rain_pctl",
    "temp_dev_from_niche",
]

#: Sets compared by the ablation of notebook 07. The names serve as MLflow `feature_set`.
PRACTICE_FEATURE_SETS = {
    "selected": PRACTICE_SELECTED,
    "selected_duration": PRACTICE_SELECTED + PRACTICE_NO_EFFECT,
    "selected_categorical": PRACTICE_SELECTED + PRACTICE_CATEGORICAL,
    "selected_enrichment": PRACTICE_SELECTED + PRACTICE_ENRICHMENT,
    "all": PRACTICE_SELECTED + PRACTICE_NO_EFFECT + PRACTICE_CATEGORICAL + PRACTICE_ENRICHMENT,
}

# --------------------------------------------------------------------------- #
# Yield model — historical dataset
# --------------------------------------------------------------------------- #

YIELD_TARGET = "yield_t_ha"

#: Original variables of the rebuilt dataset.
YIELD_CONDITIONS = ["crop", "avg_temp", "rain_mm", "pesticides_t"]

#: Derived variables, built by `src.models.history`.
YIELD_GEOGRAPHY = ["lat_abs", "geo_x", "geo_y", "geo_z"]
YIELD_HISTORY = ["temp_hist", "log_pest_hist"]

#: Sets compared by the ablations of notebooks 09 and 10.
#:
#: The first two measure the contribution of `iso3` and `year` **before** any feature
#: engineering: this is what allows stating, with measurements, that the final model
#: receives no country identifier. `year` is kept in the following sets — removing it
#: clearly costs RMSE, it carries the technological trend.
YIELD_FEATURE_SETS = {
    "conditions": YIELD_CONDITIONS,
    "conditions_year": YIELD_CONDITIONS + ["year"],
    "conditions_country": YIELD_CONDITIONS + ["iso3"],
    "conditions_country_year": YIELD_CONDITIONS + ["iso3", "year"],
    "log_pesticides": ["crop", "avg_temp", "rain_mm", "log_pesticides", "year"],
    "geography": ["crop", "avg_temp", "rain_mm", "log_pesticides", "year"] + YIELD_GEOGRAPHY,
    "history": ["crop", "rain_mm", "year"] + YIELD_HISTORY + YIELD_GEOGRAPHY,
}

#: The 10 crops of the FAO dataset, in alphabetical order.
YIELD_CROPS = [
    "Cassava",
    "Maize",
    "Plantains and others",
    "Potatoes",
    "Rice, paddy",
    "Sorghum",
    "Soybeans",
    "Sweet potatoes",
    "Wheat",
    "Yams",
]

# --------------------------------------------------------------------------- #
# Comparability of yields across crops
# --------------------------------------------------------------------------- #

#: Dry matter share, as a fraction of the harvested fresh weight.
#:
#: **Why this table exists.** A ranking in raw tonnes always puts a tuber first: potato
#: yields 16 t/ha against 2.4 t/ha for wheat. The gap is largely water — a potato is
#: 80 % water, a wheat grain 13 %. Comparing these tonnages means comparing different
#: things, and leads a recommendation service to answer "potato" whatever the
#: conditions.
#:
#: Bringing each yield back to its dry matter makes crops comparable. Usual agronomic
#: orders of magnitude (moisture content at harvest); they serve to **rank**, not to
#: compute a nutritional value.
DRY_MATTER: dict[str, float] = {
    "Maize": 0.87,
    "Wheat": 0.87,
    "Rice, paddy": 0.87,
    "Sorghum": 0.87,
    "Soybeans": 0.90,
    "Potatoes": 0.21,
    "Sweet potatoes": 0.30,
    "Cassava": 0.37,
    "Yams": 0.30,
    "Plantains and others": 0.35,
}

#: Agronomic group of each crop: a tuber is not ranked directly against a cereal,
#: even in dry matter.
CROP_GROUP: dict[str, str] = {
    "Maize": "cereale",
    "Wheat": "cereale",
    "Rice, paddy": "cereale",
    "Sorghum": "cereale",
    "Soybeans": "oleoproteagineux",
    "Potatoes": "tubercule",
    "Sweet potatoes": "tubercule",
    "Cassava": "tubercule",
    "Yams": "tubercule",
    "Plantains and others": "tubercule",
}


def to_dry_matter(yield_value, crop: str) -> float:
    """Convert a raw yield (t/ha) into tonnes of dry matter per hectare."""
    return yield_value * DRY_MATTER[crop]


def _check() -> None:
    """Every crop in the FAO dataset must have a dry matter share and a group."""
    for table, name in ((DRY_MATTER, "DRY_MATTER"), (CROP_GROUP, "CROP_GROUP")):
        missing = set(YIELD_CROPS) - set(table)
        if missing:
            raise ValueError(f"{name} : cultures absentes {sorted(missing)}")


_check()
