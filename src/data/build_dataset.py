"""Build of the consolidated dataset (step 1).

The two datasets share **no row-level join key**: Agriculture CropYield has
neither country nor year, CropYield Prediction has neither soil nor irrigation.
The crop is their only common dimension.

Agriculture CropYield therefore serves as the backbone — it carries the levers
the farmer controls — and receives, through a join on the harmonised crop, a
global agronomic profile derived from CropYield Prediction, where the gap between
crops reaches a factor of 13 while it is null in the simulated dataset.

**Known limitation.** The target of the consolidated dataset remains the yield of
the simulated dataset, which carries no crop effect. Since `crop_ref_yield_t_ha` is
collinear with `Crop`, a model trained on this target will learn to ignore it. The
`/recommend` ranking must therefore rely explicitly on `crop_ref_yield_t_ha` as a
documented post-processing step, rather than expect the model to infer the crop
order. See notebook 05.

Runnable as a script to regenerate the consolidated CSV.
"""

from __future__ import annotations

import gzip
import hashlib

import numpy as np
import pandas as pd

from src import config
from src.data import crop_yield_prediction
from src.data.crop_mapping import (
    CROP_TO_FAO_ITEM,
    IMPUTATION_RULES,
    SOURCE_FAO,
    SOURCE_IMPUTED,
)
from src.data.prep_log import PrepLog
from src.data.preparation import clean
from src.data.preparation import load as load_simulated

#: Agronomic profile columns added to each plot.
PROFILE_COLUMNS = [
    "crop_ref_yield_t_ha",
    "crop_ref_yield_p25",
    "crop_ref_yield_p75",
    "crop_pesticides_intensity",
    "crop_temp_niche",
    "crop_rain_pctl",
]


def fao_profiles(df_b: pd.DataFrame) -> pd.DataFrame:
    """Summarise CropYield Prediction into one agronomic profile per FAO crop.

    The climate descriptors describe the crop's **niche** — the conditions under
    which it is actually grown — not those where it yields best. Anchoring the
    measure on the most productive country-years seemed natural but measures
    **national wealth**, not agronomy: the top quartile is dominated by developed
    temperate countries, which put the wheat optimum at 12.9 degC and the maize one
    at 14.1 degC. The median growing temperature restores the expected order
    (wheat < soybean < maize < rice < cassava).

    Rainfall is expressed as a **percentile** of the CropYield Prediction
    distribution, never in millimetres: the simulated dataset caps at 1,000 mm while
    the CropYield Prediction median exceeds 1,000 mm — one is growing-season
    rainfall, the other an annual normal. The percentile is interpolated, because
    rainfall is constant per country: a raw rank would squash several crops onto
    the same value.
    """
    rains = np.sort(df_b["rain_mm"].dropna().to_numpy())
    ranks = 100.0 * np.arange(len(rains)) / (len(rains) - 1)

    rows = []
    for crop, group in df_b.groupby("crop"):
        median_rain = group["rain_mm"].median()
        rows.append(
            {
                "fao_item": crop,
                "crop_ref_yield_t_ha": group["yield_t_ha"].median(),
                "crop_ref_yield_p25": group["yield_t_ha"].quantile(0.25),
                "crop_ref_yield_p75": group["yield_t_ha"].quantile(0.75),
                "crop_pesticides_intensity": group["pesticides_t"].median(),
                "crop_temp_niche": group["avg_temp"].median(),
                "crop_rain_pctl": float(np.interp(median_rain, rains, ranks)),
                "n_observations": len(group),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("crop_ref_yield_t_ha", ascending=False)
        .reset_index(drop=True)
    )


def profiles_by_crop(df_b: pd.DataFrame) -> pd.DataFrame:
    """Map the FAO profiles onto the 6 crops of the simulated dataset.

    Four crops out of six have an FAO equivalent after harmonisation. Barley and
    cotton have none: each takes the profile of a documented donor crop and carries
    `crop_ref_source = 'imputed'`, so that provenance stays visible.
    """
    fao = fao_profiles(df_b).set_index("fao_item")

    rows = []
    for crop, item in CROP_TO_FAO_ITEM.items():
        if item is not None:
            profile = fao.loc[item, PROFILE_COLUMNS].to_dict()
            source, donor = SOURCE_FAO, None
        else:
            rule = IMPUTATION_RULES[crop]
            profile = fao.loc[rule.donor_item, PROFILE_COLUMNS].to_dict()
            if rule.yield_override_t_ha is not None:
                reference = rule.yield_override_t_ha
                spread = profile["crop_ref_yield_p75"] - profile["crop_ref_yield_p25"]
                profile["crop_ref_yield_t_ha"] = reference
                profile["crop_ref_yield_p25"] = reference - spread / 2
                profile["crop_ref_yield_p75"] = reference + spread / 2
            source, donor = SOURCE_IMPUTED, rule.donor_item
        rows.append(
            {"Crop": crop, **profile, "crop_ref_source": source, "donor_item": donor}
        )
    return pd.DataFrame(rows)


def merge_sources(df_a: pd.DataFrame, profiles: pd.DataFrame) -> tuple[pd.DataFrame, PrepLog]:
    """Join the per-crop profiles to the simulated dataset and derive the niche gap.

    `temp_dev_from_niche` is the only variable that truly crosses both sources row
    by row: the gap between the plot temperature and the crop's thermal niche.

    **Two reasons to distrust it**, both established in step 1.

    1. CropYield Prediction alone does not validate it: within that dataset the gap
       correlates *positively* with yield, through the country wealth bias.
    2. More fundamentally, the two temperatures **do not measure the same window**.
       The simulated dataset floors at exactly 15.0 degC, yet 44 % of real
       wheat-growing country-years sit below that in annual mean, and 53 of 137
       countries do — Russia at 4.0 degC, Kazakhstan at 5.2. An annual mean could
       not floor at 15 degC without excluding half the world's wheat belt, which an
       exhaustive catalogue would not do. The simulator therefore describes a
       **growing season**, like its rainfall; the FAO figure is a calendar-year mean.

    So this variable subtracts a growing-season temperature from an annual one. It is
    kept because it carries the agronomic intent `/recommend` needs, but it is a
    heuristic, not a measured agronomic gap. Step 2 must earn its place or drop it.
    """
    prep_log = PrepLog()
    profile_without_donor = profiles.drop(columns=["donor_item"])

    before = len(df_a)
    out = df_a.merge(profile_without_donor, on="Crop", how="left", validate="many_to_one")
    if len(out) != before:
        raise AssertionError(f"la jointure a change le nombre de lignes : {before} -> {len(out)}")
    prep_log.add("jointure", f"{before:,} lignes conservees, aucun fan-out")

    added = [c for c in profile_without_donor.columns if c != "Crop"]
    if out[added].isna().any().any():
        raise AssertionError("la jointure a introduit des valeurs manquantes")
    prep_log.add("completude", "aucune valeur manquante dans les colonnes de profil")

    out["temp_dev_from_niche"] = (out["Temperature_Celsius"] - out["crop_temp_niche"]).abs()
    out["crop_ref_source"] = out["crop_ref_source"].astype("category")
    prep_log.add("variable derivee", "temp_dev_from_niche = |temperature - niche|")

    breakdown = out["crop_ref_source"].value_counts().to_dict()
    prep_log.add("provenance", ", ".join(f"{k} : {v:,}" for k, v in breakdown.items()))
    return out, prep_log


def check_agronomy(profiles: pd.DataFrame) -> None:
    """Check that the profiles stay consistent with agronomic knowledge.

    These sanity checks guard against a silent regression: if the rebuild of the
    CropYield Prediction dataset changes, an aberrant order must make the build
    fail rather than go unnoticed.
    """
    profile = profiles.set_index("Crop")

    niches = profile["crop_temp_niche"]
    if not niches["Wheat"] < niches["Maize"] < niches["Rice"]:
        raise AssertionError(
            "niches thermiques incoherentes : attendu ble < mais < riz, obtenu "
            f"{niches['Wheat']:.1f} / {niches['Maize']:.1f} / {niches['Rice']:.1f}"
        )

    rains = profile["crop_rain_pctl"]
    if not rains["Wheat"] < rains["Rice"]:
        raise AssertionError(
            "percentiles de pluie incoherents : le ble doit etre plus sec que le riz "
            f"({rains['Wheat']:.1f} vs {rains['Rice']:.1f})"
        )

    yields = profile["crop_ref_yield_t_ha"]
    if not yields["Rice"] > yields["Maize"] > yields["Soybean"]:
        raise AssertionError(
            "rendements de reference incoherents : attendu riz > mais > soja, obtenu "
            f"{yields['Rice']:.2f} / {yields['Maize']:.2f} / {yields['Soybean']:.2f}"
        )


DICTIONARY: list[tuple[str, str, str, str, str]] = [
    # (column, type, unit, origin, description)
    ("Region", "category", "-", "A", "Zone abstraite (North/South/East/West), pas un pays"),
    ("Soil_Type", "category", "-", "A", "Texture du sol : Sandy, Loam, Chalky, Silt, Clay, Peaty"),
    ("Crop", "category", "-", "A", "Culture de la parcelle (6 especes)"),
    ("Rainfall_mm", "float64", "mm", "A", "Pluie de saison culturale (100-1000)"),
    ("Temperature_Celsius", "float64", "degC", "A",
     "Temperature moyenne de la parcelle sur la saison culturale (15-40)"),
    ("Fertilizer_Used", "bool", "-", "A", "Fertilisation appliquee"),
    ("Irrigation_Used", "bool", "-", "A", "Parcelle irriguee"),
    ("Weather_Condition", "category", "-", "A",
     "Sunny, Rainy ou Cloudy — independante de la pluie et de la temperature mesurees"),
    ("Days_to_Harvest", "int16", "jours", "A", "Duree du cycle (60-150) ; sans lien avec le rendement"),
    ("Yield_tons_per_hectare", "float64", "t/ha", "A", "CIBLE — rendement observe"),
    ("crop_ref_yield_t_ha", "float64", "t/ha", "B", "Rendement FAO median de la culture dans le monde"),
    ("crop_ref_yield_p25", "float64", "t/ha", "B", "1er quartile de la dispersion FAO"),
    ("crop_ref_yield_p75", "float64", "t/ha", "B", "3e quartile de la dispersion FAO"),
    ("crop_pesticides_intensity", "float64", "tonnes", "B",
     "Pesticides medians des pays-annees cultivant l'espece ; proxy d'intensification"),
    ("crop_temp_niche", "float64", "degC", "B",
     "Temperature mediane ANNUELLE des pays-annees cultivant l'espece (niche thermique)"),
    ("crop_rain_pctl", "float64", "percentile", "B",
     "Pluie mediane de la culture, en percentile de la distribution de B (jamais en mm)"),
    ("crop_ref_source", "category", "-", "derivee", "'FAO' ou 'imputed' (orge, coton)"),
    ("temp_dev_from_niche", "float64", "degC", "derivee",
     "abs(Temperature_Celsius - crop_temp_niche) ; croise A et B, mais soustrait une "
     "temperature de saison culturale a une moyenne annuelle : heuristique, pas un "
     "ecart agronomique mesure"),
]


def write_dictionary(path=None) -> None:
    """Write the column reference, basis of the step 3 Pydantic schemas."""
    path = path or config.DATA_DICTIONARY
    rows = [
        "# Dictionnaire des donnees — `crop_dataset_consolidated.csv`",
        "",
        "Genere par `src/data/build_dataset.py`. Origine : **A** = Agriculture CropYield,",
        "**B** = CropYield Prediction reconstruit, **derivee** = calculee lors de la fusion.",
        "",
        "| Colonne | Type | Unite | Origine | Description |",
        "|---|---|---|---|---|",
    ]
    rows += [f"| `{c}` | {t} | {u} | {o} | {d} |" for c, t, u, o, d in DICTIONARY]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def build(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chain the preparation of both datasets, the merge and the checks."""
    def trace(message: str) -> None:
        if verbose:
            print(message)

    trace("Preparation d'Agriculture CropYield ...")
    df_a, prep_log_a = clean(load_simulated())
    trace(str(prep_log_a))

    trace("\nReconstruction de CropYield Prediction ...")
    df_b, _ = crop_yield_prediction.build(verbose=False)
    trace(f"  {len(df_b):,} lignes x {df_b.shape[1]} colonnes, {df_b['iso3'].nunique()} pays")

    trace("\nProfils agronomiques par culture ...")
    profiles = profiles_by_crop(df_b)
    check_agronomy(profiles)
    trace(profiles.round(2).to_string(index=False))
    trace("  controles agronomiques : ordres respectes")

    trace("\nFusion ...")
    consolidated, merge_log = merge_sources(df_a, profiles)
    trace(str(merge_log))
    trace(f"\nconsolide : {len(consolidated):,} lignes x {consolidated.shape[1]} colonnes")
    return consolidated, profiles


def load_consolidated() -> pd.DataFrame:
    """Read the consolidated dataset, from whichever form is available.

    The plain CSV is preferred when present — it is faster to read and it is what the
    local work regenerates. A freshly cloned repository only holds the gzipped form,
    which is the versioned deliverable, and `read_csv` handles it from the extension
    alone. Either way the DataFrame is identical.
    """
    path = (
        config.CONSOLIDATED_DATASET_CSV
        if config.CONSOLIDATED_DATASET_CSV.exists()
        else config.CONSOLIDATED_DATASET_GZ
    )
    if not path.exists():
        raise FileNotFoundError(
            f"dataset consolide absent : ni {config.CONSOLIDATED_DATASET_CSV.name} ni "
            f"{config.CONSOLIDATED_DATASET_GZ.name} dans {config.PROCESSED_DATA}. "
            "Le regenerer avec `python -m src.data.build_dataset`."
        )
    return pd.read_csv(path)


def _digest(path, opener) -> str:
    """MD5 of a file's (decompressed) content, read in chunks."""
    digest = hashlib.md5()
    with opener(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gzip_is_stale() -> bool:
    """True when the gzipped dataset is missing or differs from the plain CSV."""
    if not config.CONSOLIDATED_DATASET_GZ.exists():
        return True
    return _digest(config.CONSOLIDATED_DATASET_GZ, gzip.open) != _digest(
        config.CONSOLIDATED_DATASET_CSV, open
    )


def main() -> None:
    config.PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    config.REPORTS.mkdir(parents=True, exist_ok=True)

    consolidated, _ = build()
    consolidated.to_csv(config.CONSOLIDATED_DATASET_CSV, index=False)
    # The gzipped file is the versioned deliverable: the plain CSV exceeds GitHub's
    # 100 MB per-file limit, the gzipped one fits in 43 MB. It is only rewritten when
    # its content changes: a gzip header carries a timestamp, so rewriting identical
    # data would still hand git a new 43 MB blob on every replay.
    if _gzip_is_stale():
        consolidated.to_csv(
            config.CONSOLIDATED_DATASET_GZ, index=False,
            compression={"method": "gzip", "mtime": 0},
        )
    write_dictionary()

    for path in (config.CONSOLIDATED_DATASET_CSV, config.CONSOLIDATED_DATASET_GZ):
        size_mb = path.stat().st_size / 1024**2
        print(f"\necrit {path.relative_to(config.ROOT)} ({size_mb:.1f} Mo)")
    print(f"ecrit {config.DATA_DICTIONARY.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
