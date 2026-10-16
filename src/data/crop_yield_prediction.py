"""Rebuild of the CropYield Prediction dataset from its four raw sources.

The brief provides `yield_df.csv`, already merged. We do not use it, for two
measured reasons:

1. **Coverage.** It was assembled by joining on the literal country name, which
   only matches for 101 countries across the four sources. The United States,
   China, Russia, Iran, Viet Nam, South Korea, Tanzania, Czechia and Turkey are
   missing from it. Matching on ISO3 code keeps 120.
2. **Duplicates.** `temp.csv` contains up to 41 readings for a single
   country-year pair. Joined without prior aggregation, it multiplies rows: this
   is the source of the 3,837 duplicated keys in `yield_df.csv`. We aggregate
   before joining, and every join is checked against any fan-out.

Output granularity: **iso3 x year x crop**, over 1990-2013.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.data import geo
from src.data.rapport import Rapport

#: Key of an observation in the rebuilt dataset.
CLE = ["iso3", "year", "crop"]

#: FAO aggregate including Taiwan, Hong Kong and Macao. The three explanatory sources
#: only describe mainland China, already present as `China, mainland`:
#: keeping both would duplicate it over a different scope.
AGREGAT_CHINE = "China"


def _ajoute_iso3(df: pd.DataFrame, colonne_pays: str) -> pd.DataFrame:
    """Add an `iso3` column derived from the country name."""
    correspondances = geo.vers_iso3(df[colonne_pays].astype(str).unique())
    return df.assign(iso3=df[colonne_pays].astype(str).map(correspondances))


def charge_sources() -> dict[str, pd.DataFrame]:
    """Read the four raw files without any transformation."""
    pluie = pd.read_csv(config.SOURCE_PLUIE)
    # The country column header starts with a space in this file.
    pluie.columns = pluie.columns.str.strip()
    return {
        "rendement": pd.read_csv(config.SOURCE_RENDEMENT),
        "temperature": pd.read_csv(config.SOURCE_TEMPERATURE),
        "pluie": pluie,
        "pesticides": pd.read_csv(config.SOURCE_PESTICIDES),
    }


def _periode(df: pd.DataFrame, colonne: str) -> pd.DataFrame:
    return df[df[colonne].between(config.ANNEE_DEBUT, config.ANNEE_FIN)]


def prepare_rendement(df: pd.DataFrame) -> tuple[pd.DataFrame, Rapport]:
    """Yield per country, year and crop, converted to tonnes per hectare."""
    rapport = Rapport()
    n_depart = len(df)

    sans_agregat = df[df["Area"] != AGREGAT_CHINE]
    rapport.ajoute(
        "agregat Chine",
        f"{n_depart - len(sans_agregat)} lignes `{AGREGAT_CHINE}` ecartees "
        "au profit de `China, mainland`",
    )

    out = (
        _periode(sans_agregat, "Year")
        .pipe(_ajoute_iso3, "Area")
        .rename(columns={"Area": "area", "Year": "year", "Item": "crop"})
        .assign(yield_t_ha=lambda d: d["Value"] / 10_000)
        .loc[:, ["iso3", "area", "year", "crop", "yield_t_ha"]]
    )
    rapport.ajoute("unite", "hg/ha -> t/ha (division par 10 000)")

    sans_code = out["iso3"].isna().sum()
    out = out.dropna(subset=["iso3"])
    rapport.ajoute(
        "sans ISO3",
        f"{sans_code} lignes ecartees (entites historiques : URSS, Tchecoslovaquie, "
        "Belgique-Luxembourg, Yougoslavie, Serbie-et-Montenegro, Soudan ex)",
    )

    # Two distinct labels may point to the same country: we settle it with the mean.
    doublons = int(out.duplicated(CLE).sum())
    if doublons:
        out = out.groupby(CLE, as_index=False).agg(
            area=("area", "first"), yield_t_ha=("yield_t_ha", "mean")
        )
    rapport.ajoute("doublons sur la cle", f"{doublons} resolus par moyenne")
    rapport.ajoute("lignes", f"{n_depart:,} -> {len(out):,}")
    return out.reset_index(drop=True), rapport


def prepare_temperature(df: pd.DataFrame) -> tuple[pd.DataFrame, Rapport]:
    """Mean temperature per country and year.

    This is the source of the `yield_df.csv` fan-out: up to 41 readings for a single
    country-year pair. We drop exact duplicates, discard empty readings, then
    average — without geographic weighting, for lack of metadata to tell the
    stations apart.
    """
    rapport = Rapport()
    n_depart = len(df)

    stricts = df.drop_duplicates()
    rapport.ajoute("doublons stricts", f"{n_depart - len(stricts)} lignes retirees")

    vides = int(stricts["avg_temp"].isna().sum())
    renseignes = stricts.dropna(subset=["avg_temp"])
    rapport.ajoute("releves vides", f"{vides} lignes sans temperature ecartees")

    avant_agregation = _periode(renseignes, "year").pipe(_ajoute_iso3, "country")
    max_releves = int(avant_agregation.groupby(["iso3", "year"]).size().max())
    rapport.ajoute(
        "releves par pays-annee",
        f"jusqu'a {max_releves} — c'est ce fan-out qui corrompt `yield_df.csv`",
    )

    out = (
        avant_agregation.dropna(subset=["iso3"])
        .groupby(["iso3", "year"], as_index=False)["avg_temp"]
        .mean()
    )
    rapport.ajoute("agregation", "moyenne des releves par pays-annee")
    rapport.ajoute("lignes", f"{n_depart:,} -> {len(out):,}")
    return out, rapport


def prepare_pluie(df: pd.DataFrame) -> tuple[pd.DataFrame, Rapport]:
    """Rainfall per country, represented as the national constant it is.

    `rainfall.csv` does not provide a yearly measurement but a **climate normal that
    is constant per country**: 188 countries out of 192 have exactly the same value
    every year. Two defects follow, which this representation fixes in one go.

    1. **The file has no row for 2003** (it runs from 1985 to 2017, 2003 missing).
       Joined as is, that year loses its rainfall for every country. By keeping one
       value per country and broadcasting it over all years, the gap disappears
       without inventing any value.
    2. **Four countries carry one aberrant year** out of 31: India has 1,940 mm one
       year against 1,083 mm the other thirty — 1,940 being the value of Iceland, the
       alphabetically preceding country. Paraguay likewise takes that of Papua New
       Guinea. Keeping the country's **modal value** discards these aberrations, the
       majority winning over the accident.

    One defect remains that the median does not fix: eleven countries whose whole
    series is copied from the preceding country (see :data:`geo.PLUIE_NON_FIABLE`).
    Their `rain_mm` is set to `NaN`: the whole series is wrong, there is nothing to keep.

    The mode is computed over **the whole file**, 1985-2017, not only over the
    retained period: more years make the majority value more reliable.
    """
    rapport = Rapport()
    n_depart = len(df)

    converti = df.assign(
        rain_mm=pd.to_numeric(df["average_rain_fall_mm_per_year"], errors="coerce")
    )
    non_numeriques = int(converti["rain_mm"].isna().sum())
    rapport.ajoute("conversion", f"{non_numeriques} valeurs non numeriques ou vides -> NaN")

    renseigne = (
        _ajoute_iso3(converti, "Area").dropna(subset=["iso3", "rain_mm"])
    )

    # One value per country: the modal one. Countries with several values are those
    # with an aberrant year; the mode fixes them.
    valeurs_distinctes = renseigne.groupby("iso3")["rain_mm"].nunique()
    repares = sorted(valeurs_distinctes[valeurs_distinctes > 1].index)
    modes = renseigne.groupby("iso3")["rain_mm"].agg(lambda s: s.mode().iloc[0])
    rapport.ajoute(
        "annees aberrantes",
        f"{len(repares)} pays a plusieurs valeurs ({', '.join(repares)}) "
        "-> valeur modale retenue",
    )

    ecartes = sorted(set(modes.index) & geo.PLUIE_NON_FIABLE)
    modes = modes.drop(index=ecartes)
    rapport.ajoute(
        "series non fiables",
        f"{len(ecartes)} pays ecartes ({', '.join(ecartes)}) : serie entiere copiee "
        "du pays alphabetiquement precedent",
    )

    annees = range(config.ANNEE_DEBUT, config.ANNEE_FIN + 1)
    out = pd.DataFrame(
        [{"iso3": iso3, "year": annee, "rain_mm": valeur}
         for iso3, valeur in modes.items() for annee in annees]
    )
    rapport.ajoute(
        "diffusion",
        f"1 valeur x {len(modes)} pays x {len(annees)} annees -> aucun trou, 2003 comprise",
    )
    rapport.ajoute("lignes", f"{n_depart:,} -> {len(out):,}")
    return out, rapport


def prepare_pesticides(df: pd.DataFrame) -> tuple[pd.DataFrame, Rapport]:
    """Total pesticide use per country and year, in tonnes of active ingredient."""
    rapport = Rapport()
    n_depart = len(df)

    out = (
        _periode(df, "Year")
        .pipe(_ajoute_iso3, "Area")
        .dropna(subset=["iso3"])
        .rename(columns={"Year": "year", "Value": "pesticides_t"})
        .groupby(["iso3", "year"], as_index=False)["pesticides_t"]
        .mean()
    )
    rapport.ajoute("lignes", f"{n_depart:,} -> {len(out):,}")
    return out, rapport


def controle_avant_fusion(sources: dict[str, tuple[pd.DataFrame, list[str]]]) -> pd.DataFrame:
    """Check that each source is unique on its key, before any join."""
    lignes = []
    for nom, (df, cle) in sources.items():
        doublons = int(df.duplicated(cle).sum())
        lignes.append(
            {
                "source": nom,
                "lignes": len(df),
                "pays": df["iso3"].nunique(),
                "cle": " + ".join(cle),
                "doublons": doublons,
            }
        )
        if doublons:
            raise AssertionError(f"{nom} : la cle {cle} n'est pas unique ({doublons})")
    return pd.DataFrame(lignes)


def fusionne(
    rendement: pd.DataFrame,
    temperature: pd.DataFrame,
    pluie: pd.DataFrame,
    pesticides: pd.DataFrame,
) -> tuple[pd.DataFrame, Rapport]:
    """Join the three explanatory sources onto the yield base.

    Each join is followed by a fan-out check: a left join must never change the
    number of rows. This is exactly the check that was missing from the build of
    `yield_df.csv`.
    """
    rapport = Rapport()
    dataset = rendement
    rapport.ajoute("socle rendement", f"{len(dataset):,} lignes")

    for nom, source in [
        ("temperature", temperature),
        ("pluie", pluie),
        ("pesticides", pesticides),
    ]:
        avant = len(dataset)
        dataset = dataset.merge(source, on=["iso3", "year"], how="left", validate="many_to_one")
        if len(dataset) != avant:
            raise AssertionError(f"fan-out lors de la jointure {nom} : {avant} -> {len(dataset)}")
        rapport.ajoute(f"+ {nom}", f"{avant:,} lignes conservees, aucun fan-out")

    doublons = int(dataset.duplicated(CLE).sum())
    if doublons:
        raise AssertionError(f"la cle finale n'est pas unique ({doublons} doublons)")
    rapport.ajoute("cle finale", f"{' + '.join(CLE)} unique")
    return dataset.reset_index(drop=True), rapport


def couverture(df: pd.DataFrame, colonne: str) -> pd.Series:
    """Split a column's missing values between whole countries and isolated gaps.

    The distinction drives the treatment. If a column is missing for **an entire**
    country, there is nothing to recover and the country must be discarded. If it is
    only missing for some years, the country's known value can fill the gap without
    inventing anything.
    """
    par_pays = df.groupby("iso3")[colonne].agg(["count", "size"])
    aucune = par_pays[par_pays["count"] == 0]
    partielle = par_pays[(par_pays["count"] > 0) & (par_pays["count"] < par_pays["size"])]
    return pd.Series(
        {
            "lignes manquantes": int(df[colonne].isna().sum()),
            "pays sans aucune valeur": len(aucune),
            "lignes de ces pays": int(aucune["size"].sum()),
            "pays avec quelques annees manquantes": len(partielle),
        },
        name=colonne,
    )


def table_couverture(df: pd.DataFrame, colonnes: list[str] | None = None) -> pd.DataFrame:
    """Apply :func:`couverture` to several columns, for side-by-side reading."""
    colonnes = colonnes or ["avg_temp", "rain_mm", "pesticides_t"]
    return pd.DataFrame({c: couverture(df, c) for c in colonnes})


#: Variables whose training domain is bounded.
VARIABLES_DOMAINE = ["yield_t_ha", "avg_temp", "rain_mm", "pesticides_t"]


def domaine_apprentissage(
    df: pd.DataFrame, annee_test: int | None = None, quantiles: tuple[float, float] = (0.05, 0.95)
) -> pd.DataFrame:
    """Usual bounds of each variable, crop by crop.

    The 5 % and 95 % quantiles describe the central 90 % observed. The step 3 API
    will use them to flag an unusual input **without rejecting it**: falling outside
    these bounds is not an error, it is a zone where the model learned little.

    `annee_test` is excluded from the computation: the year held out for the final
    `/recommend` test must not influence any upstream decision, interface bounds
    included.
    """
    annee_test = config.ANNEE_TEST if annee_test is None else annee_test
    avant = df[df["year"] < annee_test]
    bas, haut = quantiles

    lignes = []
    for culture, groupe in avant.groupby("crop"):
        for variable in VARIABLES_DOMAINE:
            valeurs = groupe[variable].dropna()
            if valeurs.empty:
                continue
            lignes.append(
                {
                    "crop": culture,
                    "variable": variable,
                    "q05": valeurs.quantile(bas),
                    "q95": valeurs.quantile(haut),
                    "min": valeurs.min(),
                    "max": valeurs.max(),
                    "n": len(valeurs),
                }
            )
    return pd.DataFrame(lignes)


def construit(verbeux: bool = True) -> tuple[pd.DataFrame, dict[str, Rapport]]:
    """Chain reading, preparation, checks and merge."""
    brut = charge_sources()
    rendement, r_rendement = prepare_rendement(brut["rendement"])
    temperature, r_temp = prepare_temperature(brut["temperature"])
    pluie, r_pluie = prepare_pluie(brut["pluie"])
    pesticides, r_pest = prepare_pesticides(brut["pesticides"])

    controle = controle_avant_fusion(
        {
            "rendement": (rendement, CLE),
            "temperature": (temperature, ["iso3", "year"]),
            "pluie": (pluie, ["iso3", "year"]),
            "pesticides": (pesticides, ["iso3", "year"]),
        }
    )

    dataset, r_fusion = fusionne(rendement, temperature, pluie, pesticides)

    rapports = {
        "rendement": r_rendement,
        "temperature": r_temp,
        "pluie": r_pluie,
        "pesticides": r_pest,
        "fusion": r_fusion,
    }
    if verbeux:
        for nom, rapport in rapports.items():
            print(f"-- {nom}\n{rapport}\n")
        print(controle.to_string(index=False))
        print(f"\ndataset : {len(dataset):,} lignes x {dataset.shape[1]} colonnes")
    return dataset, rapports


def main() -> None:
    config.DATA_TRAITE.mkdir(parents=True, exist_ok=True)
    dataset, _ = construit()
    dataset.to_csv(config.CROP_YIELD_PREDICTION_CSV, index=False)
    print(f"\necrit {config.CROP_YIELD_PREDICTION_CSV.relative_to(config.RACINE)}")

    domaine = domaine_apprentissage(dataset)
    domaine.to_csv(config.DOMAINE_APPRENTISSAGE_CSV, index=False)
    print(
        f"ecrit {config.DOMAINE_APPRENTISSAGE_CSV.relative_to(config.RACINE)} "
        f"({len(domaine)} bornes, annee {config.ANNEE_TEST} exclue)"
    )


if __name__ == "__main__":
    main()
