"""Feature sets of the two services, declared once.

This module is the single reference: the step 2 notebooks and the step 3 API draw
the same lists from it. Do not redefine a feature set elsewhere.

It also holds the **dry matter table**, which makes yields comparable across crops
(see :data:`MATIERE_SECHE`).
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# /predict — simulated dataset
# --------------------------------------------------------------------------- #

CIBLE_PREDICT = "Yield_tons_per_hectare"

#: The 4 variables carrying the signal, established in step 1 and confirmed by the
#: ablation of notebook 07. They are also the 4 the interface will ask for.
PREDICT_RETENUES = [
    "Rainfall_mm",
    "Temperature_Celsius",
    "Fertilizer_Used",
    "Irrigation_Used",
]

#: Variables with no measurable effect, kept in the dataset but excluded from the model.
PREDICT_SANS_EFFET = ["Days_to_Harvest"]
PREDICT_CATEGORIELLES = ["Region", "Soil_Type", "Crop", "Weather_Condition"]

#: Enrichment columns from the FAO dataset. The ablation shows they add nothing to this
#: target: the simulated yield does not depend on the crop, and they are collinear with
#: `Crop`. They stay in the consolidated dataset as documentation.
PREDICT_ENRICHISSEMENT = [
    "crop_ref_yield_t_ha",
    "crop_ref_yield_p25",
    "crop_ref_yield_p75",
    "crop_pesticides_intensity",
    "crop_temp_niche",
    "crop_rain_pctl",
    "temp_dev_from_niche",
]

#: Sets compared by the ablation of notebook 07. The names serve as MLflow `feature_set`.
PREDICT_JEUX = {
    "retenues": PREDICT_RETENUES,
    "retenues_duree": PREDICT_RETENUES + PREDICT_SANS_EFFET,
    "retenues_categorielles": PREDICT_RETENUES + PREDICT_CATEGORIELLES,
    "retenues_enrichissement": PREDICT_RETENUES + PREDICT_ENRICHISSEMENT,
    "toutes": PREDICT_RETENUES + PREDICT_SANS_EFFET + PREDICT_CATEGORIELLES + PREDICT_ENRICHISSEMENT,
}

# --------------------------------------------------------------------------- #
# /recommend — historical dataset
# --------------------------------------------------------------------------- #

CIBLE_RECOMMEND = "yield_t_ha"

#: Original variables of the rebuilt dataset.
RECOMMEND_CONDITIONS = ["crop", "avg_temp", "rain_mm", "pesticides_t"]

#: Derived variables, built by `src.models.historique`.
RECOMMEND_GEOGRAPHIE = ["lat_abs", "geo_x", "geo_y", "geo_z"]
RECOMMEND_HISTORIQUE = ["temp_hist", "log_pest_hist"]

#: Sets compared by the ablations of notebooks 09 and 10.
#:
#: The first two measure the contribution of `iso3` and `year` **before** any feature
#: engineering: this is what allows stating, with measurements, that the final model
#: receives no country identifier. `year` is kept in the following sets — removing it
#: clearly costs RMSE, it carries the technological trend.
RECOMMEND_JEUX = {
    "conditions": RECOMMEND_CONDITIONS,
    "conditions_annee": RECOMMEND_CONDITIONS + ["year"],
    "conditions_pays": RECOMMEND_CONDITIONS + ["iso3"],
    "conditions_pays_annee": RECOMMEND_CONDITIONS + ["iso3", "year"],
    "log_pesticides": ["crop", "avg_temp", "rain_mm", "log_pesticides", "year"],
    "geographie": ["crop", "avg_temp", "rain_mm", "log_pesticides", "year"] + RECOMMEND_GEOGRAPHIE,
    "historique": ["crop", "rain_mm", "year"] + RECOMMEND_HISTORIQUE + RECOMMEND_GEOGRAPHIE,
}

#: The 10 crops of the FAO dataset, in alphabetical order.
RECOMMEND_CULTURES = [
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
MATIERE_SECHE: dict[str, float] = {
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
GROUPE_CULTURE: dict[str, str] = {
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


def en_matiere_seche(rendement, culture: str) -> float:
    """Convert a raw yield (t/ha) into tonnes of dry matter per hectare."""
    return rendement * MATIERE_SECHE[culture]


def _controle() -> None:
    """Every crop in the FAO dataset must have a dry matter share and a group."""
    for table, nom in ((MATIERE_SECHE, "MATIERE_SECHE"), (GROUPE_CULTURE, "GROUPE_CULTURE")):
        manquantes = set(RECOMMEND_CULTURES) - set(table)
        if manquantes:
            raise ValueError(f"{nom} : cultures absentes {sorted(manquantes)}")


_controle()
