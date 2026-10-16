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
from src.data.preparation import charge, nettoie
from src.data.rapport import Rapport

#: Agronomic profile columns added to each plot.
COLONNES_PROFIL = [
    "crop_ref_yield_t_ha",
    "crop_ref_yield_p25",
    "crop_ref_yield_p75",
    "crop_pesticides_intensity",
    "crop_temp_niche",
    "crop_rain_pctl",
]


def profils_fao(df_b: pd.DataFrame) -> pd.DataFrame:
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
    pluies = np.sort(df_b["rain_mm"].dropna().to_numpy())
    rangs = 100.0 * np.arange(len(pluies)) / (len(pluies) - 1)

    lignes = []
    for culture, groupe in df_b.groupby("crop"):
        pluie_mediane = groupe["rain_mm"].median()
        lignes.append(
            {
                "fao_item": culture,
                "crop_ref_yield_t_ha": groupe["yield_t_ha"].median(),
                "crop_ref_yield_p25": groupe["yield_t_ha"].quantile(0.25),
                "crop_ref_yield_p75": groupe["yield_t_ha"].quantile(0.75),
                "crop_pesticides_intensity": groupe["pesticides_t"].median(),
                "crop_temp_niche": groupe["avg_temp"].median(),
                "crop_rain_pctl": float(np.interp(pluie_mediane, pluies, rangs)),
                "n_observations": len(groupe),
            }
        )
    return (
        pd.DataFrame(lignes)
        .sort_values("crop_ref_yield_t_ha", ascending=False)
        .reset_index(drop=True)
    )


def profils_par_culture(df_b: pd.DataFrame) -> pd.DataFrame:
    """Map the FAO profiles onto the 6 crops of the simulated dataset.

    Four crops out of six have an FAO equivalent after harmonisation. Barley and
    cotton have none: each takes the profile of a documented donor crop and carries
    `crop_ref_source = 'imputed'`, so that provenance stays visible.
    """
    fao = profils_fao(df_b).set_index("fao_item")

    lignes = []
    for culture, item in CROP_TO_FAO_ITEM.items():
        if item is not None:
            profil = fao.loc[item, COLONNES_PROFIL].to_dict()
            source, donneuse = SOURCE_FAO, None
        else:
            regle = IMPUTATION_RULES[culture]
            profil = fao.loc[regle.donor_item, COLONNES_PROFIL].to_dict()
            if regle.yield_override_t_ha is not None:
                reference = regle.yield_override_t_ha
                etendue = profil["crop_ref_yield_p75"] - profil["crop_ref_yield_p25"]
                profil["crop_ref_yield_t_ha"] = reference
                profil["crop_ref_yield_p25"] = reference - etendue / 2
                profil["crop_ref_yield_p75"] = reference + etendue / 2
            source, donneuse = SOURCE_IMPUTED, regle.donor_item
        lignes.append(
            {"Crop": culture, **profil, "crop_ref_source": source, "donor_item": donneuse}
        )
    return pd.DataFrame(lignes)


def fusionne(df_a: pd.DataFrame, profils: pd.DataFrame) -> tuple[pd.DataFrame, Rapport]:
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
    rapport = Rapport()
    profil_sans_donneuse = profils.drop(columns=["donor_item"])

    avant = len(df_a)
    out = df_a.merge(profil_sans_donneuse, on="Crop", how="left", validate="many_to_one")
    if len(out) != avant:
        raise AssertionError(f"la jointure a change le nombre de lignes : {avant} -> {len(out)}")
    rapport.ajoute("jointure", f"{avant:,} lignes conservees, aucun fan-out")

    ajoutees = [c for c in profil_sans_donneuse.columns if c != "Crop"]
    if out[ajoutees].isna().any().any():
        raise AssertionError("la jointure a introduit des valeurs manquantes")
    rapport.ajoute("completude", "aucune valeur manquante dans les colonnes de profil")

    out["temp_dev_from_niche"] = (out["Temperature_Celsius"] - out["crop_temp_niche"]).abs()
    out["crop_ref_source"] = out["crop_ref_source"].astype("category")
    rapport.ajoute("variable derivee", "temp_dev_from_niche = |temperature - niche|")

    repartition = out["crop_ref_source"].value_counts().to_dict()
    rapport.ajoute("provenance", ", ".join(f"{k} : {v:,}" for k, v in repartition.items()))
    return out, rapport


def controle_agronomique(profils: pd.DataFrame) -> None:
    """Check that the profiles stay consistent with agronomic knowledge.

    These sanity checks guard against a silent regression: if the rebuild of the
    CropYield Prediction dataset changes, an aberrant order must make the build
    fail rather than go unnoticed.
    """
    profil = profils.set_index("Crop")

    niches = profil["crop_temp_niche"]
    if not niches["Wheat"] < niches["Maize"] < niches["Rice"]:
        raise AssertionError(
            "niches thermiques incoherentes : attendu ble < mais < riz, obtenu "
            f"{niches['Wheat']:.1f} / {niches['Maize']:.1f} / {niches['Rice']:.1f}"
        )

    pluies = profil["crop_rain_pctl"]
    if not pluies["Wheat"] < pluies["Rice"]:
        raise AssertionError(
            "percentiles de pluie incoherents : le ble doit etre plus sec que le riz "
            f"({pluies['Wheat']:.1f} vs {pluies['Rice']:.1f})"
        )

    rendements = profil["crop_ref_yield_t_ha"]
    if not rendements["Rice"] > rendements["Maize"] > rendements["Soybean"]:
        raise AssertionError(
            "rendements de reference incoherents : attendu riz > mais > soja, obtenu "
            f"{rendements['Rice']:.2f} / {rendements['Maize']:.2f} / {rendements['Soybean']:.2f}"
        )


DICTIONNAIRE: list[tuple[str, str, str, str, str]] = [
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
     "abs(Temperature_Celsius - crop_temp_niche) ; croise A et B, mais soustrait une\n      temperature de saison culturale a une moyenne annuelle : heuristique, pas un\n      ecart agronomique mesure"),
]


def ecrit_dictionnaire(chemin=None) -> None:
    """Write the column reference, basis of the step 3 Pydantic schemas."""
    chemin = chemin or config.DICTIONNAIRE_DONNEES
    lignes = [
        "# Dictionnaire des donnees — `crop_dataset_consolidated.csv`",
        "",
        "Genere par `src/data/build_dataset.py`. Origine : **A** = Agriculture CropYield,",
        "**B** = CropYield Prediction reconstruit, **derivee** = calculee lors de la fusion.",
        "",
        "| Colonne | Type | Unite | Origine | Description |",
        "|---|---|---|---|---|",
    ]
    lignes += [f"| `{c}` | {t} | {u} | {o} | {d} |" for c, t, u, o, d in DICTIONNAIRE]
    chemin.write_text("\n".join(lignes) + "\n", encoding="utf-8")


def construit(verbeux: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chain the preparation of both datasets, the merge and the checks."""
    def trace(message: str) -> None:
        if verbeux:
            print(message)

    trace("Preparation d'Agriculture CropYield ...")
    df_a, rapport_a = nettoie(charge())
    trace(str(rapport_a))

    trace("\nReconstruction de CropYield Prediction ...")
    df_b, _ = crop_yield_prediction.construit(verbeux=False)
    trace(f"  {len(df_b):,} lignes x {df_b.shape[1]} colonnes, {df_b['iso3'].nunique()} pays")

    trace("\nProfils agronomiques par culture ...")
    profils = profils_par_culture(df_b)
    controle_agronomique(profils)
    trace(profils.round(2).to_string(index=False))
    trace("  controles agronomiques : ordres respectes")

    trace("\nFusion ...")
    consolide, rapport_fusion = fusionne(df_a, profils)
    trace(str(rapport_fusion))
    trace(f"\nconsolide : {len(consolide):,} lignes x {consolide.shape[1]} colonnes")
    return consolide, profils


def main() -> None:
    config.DATA_TRAITE.mkdir(parents=True, exist_ok=True)
    config.RAPPORTS.mkdir(parents=True, exist_ok=True)

    consolide, _ = construit()
    consolide.to_csv(config.DATASET_CONSOLIDE_CSV, index=False)
    ecrit_dictionnaire()

    taille = config.DATASET_CONSOLIDE_CSV.stat().st_size / 1024**2
    print(f"\necrit {config.DATASET_CONSOLIDE_CSV.relative_to(config.RACINE)} ({taille:.1f} Mo)")
    print(f"ecrit {config.DICTIONNAIRE_DONNEES.relative_to(config.RACINE)}")


if __name__ == "__main__":
    main()
