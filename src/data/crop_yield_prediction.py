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
from src.data.prep_log import PrepLog

#: Key of an observation in the rebuilt dataset.
KEY = ["iso3", "year", "crop"]

#: FAO aggregate including Taiwan, Hong Kong and Macao. The three explanatory sources
#: only describe mainland China, already present as `China, mainland`:
#: keeping both would duplicate it over a different scope.
CHINA_AGGREGATE = "China"


def _add_iso3(df: pd.DataFrame, country_column: str) -> pd.DataFrame:
    """Add an `iso3` column derived from the country name."""
    matches = geo.to_iso3(df[country_column].astype(str).unique())
    return df.assign(iso3=df[country_column].astype(str).map(matches))


def load_sources() -> dict[str, pd.DataFrame]:
    """Read the four raw files without any transformation."""
    rain = pd.read_csv(config.SOURCE_RAIN)
    # The country column header starts with a space in this file.
    rain.columns = rain.columns.str.strip()
    return {
        "rendement": pd.read_csv(config.SOURCE_YIELD),
        "temperature": pd.read_csv(config.SOURCE_TEMPERATURE),
        "pluie": rain,
        "pesticides": pd.read_csv(config.SOURCE_PESTICIDES),
    }


def _period(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df[df[column].between(config.START_YEAR, config.END_YEAR)]


def prepare_yield(df: pd.DataFrame) -> tuple[pd.DataFrame, PrepLog]:
    """Yield per country, year and crop, converted to tonnes per hectare."""
    prep_log = PrepLog()
    n_start = len(df)

    without_aggregate = df[df["Area"] != CHINA_AGGREGATE]
    prep_log.add(
        "agregat Chine",
        f"{n_start - len(without_aggregate)} lignes `{CHINA_AGGREGATE}` ecartees "
        "au profit de `China, mainland`",
    )

    out = (
        _period(without_aggregate, "Year")
        .pipe(_add_iso3, "Area")
        .rename(columns={"Area": "area", "Year": "year", "Item": "crop"})
        .assign(yield_t_ha=lambda d: d["Value"] / 10_000)
        .loc[:, ["iso3", "area", "year", "crop", "yield_t_ha"]]
    )
    prep_log.add("unite", "hg/ha -> t/ha (division par 10 000)")

    without_code = out["iso3"].isna().sum()
    out = out.dropna(subset=["iso3"])
    prep_log.add(
        "sans ISO3",
        f"{without_code} lignes ecartees (entites historiques : URSS, Tchecoslovaquie, "
        "Belgique-Luxembourg, Yougoslavie, Serbie-et-Montenegro, Soudan ex)",
    )

    # Two distinct labels may point to the same country: we settle it with the mean.
    duplicates = int(out.duplicated(KEY).sum())
    if duplicates:
        out = out.groupby(KEY, as_index=False).agg(
            area=("area", "first"), yield_t_ha=("yield_t_ha", "mean")
        )
    prep_log.add("doublons sur la cle", f"{duplicates} resolus par moyenne")
    prep_log.add("lignes", f"{n_start:,} -> {len(out):,}")
    return out.reset_index(drop=True), prep_log


def prepare_temperature(df: pd.DataFrame) -> tuple[pd.DataFrame, PrepLog]:
    """Mean temperature per country and year.

    This is the source of the `yield_df.csv` fan-out: up to 41 readings for a single
    country-year pair. We drop exact duplicates, discard empty readings, then
    average — without geographic weighting, for lack of metadata to tell the
    stations apart.
    """
    prep_log = PrepLog()
    n_start = len(df)

    strict = df.drop_duplicates()
    prep_log.add("doublons stricts", f"{n_start - len(strict)} lignes retirees")

    n_empty = int(strict["avg_temp"].isna().sum())
    filled_rows = strict.dropna(subset=["avg_temp"])
    prep_log.add("releves vides", f"{n_empty} lignes sans temperature ecartees")

    before_aggregation = _period(filled_rows, "year").pipe(_add_iso3, "country")
    max_readings = int(before_aggregation.groupby(["iso3", "year"]).size().max())
    prep_log.add(
        "releves par pays-annee",
        f"jusqu'a {max_readings} — c'est ce fan-out qui corrompt `yield_df.csv`",
    )

    out = (
        before_aggregation.dropna(subset=["iso3"])
        .groupby(["iso3", "year"], as_index=False)["avg_temp"]
        .mean()
    )
    prep_log.add("agregation", "moyenne des releves par pays-annee")
    prep_log.add("lignes", f"{n_start:,} -> {len(out):,}")
    return out, prep_log


def prepare_rain(df: pd.DataFrame) -> tuple[pd.DataFrame, PrepLog]:
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
    series is copied from the preceding country (see :data:`geo.UNRELIABLE_RAIN`).
    Their `rain_mm` is set to `NaN`: the whole series is wrong, there is nothing to keep.

    The mode is computed over **the whole file**, 1985-2017, not only over the
    retained period: more years make the majority value more reliable.
    """
    prep_log = PrepLog()
    n_start = len(df)

    converted = df.assign(
        rain_mm=pd.to_numeric(df["average_rain_fall_mm_per_year"], errors="coerce")
    )
    non_numeric = int(converted["rain_mm"].isna().sum())
    prep_log.add("conversion", f"{non_numeric} valeurs non numeriques ou vides -> NaN")

    filled = (
        _add_iso3(converted, "Area").dropna(subset=["iso3", "rain_mm"])
    )

    # One value per country: the modal one. Countries with several values are those
    # with an aberrant year; the mode fixes them.
    distinct_values = filled.groupby("iso3")["rain_mm"].nunique()
    repaired = sorted(distinct_values[distinct_values > 1].index)
    modes = filled.groupby("iso3")["rain_mm"].agg(lambda s: s.mode().iloc[0])
    prep_log.add(
        "annees aberrantes",
        f"{len(repaired)} pays a plusieurs valeurs ({', '.join(repaired)}) "
        "-> valeur modale retenue",
    )

    discarded = sorted(set(modes.index) & geo.UNRELIABLE_RAIN)
    modes = modes.drop(index=discarded)
    prep_log.add(
        "series non fiables",
        f"{len(discarded)} pays ecartes ({', '.join(discarded)}) : serie entiere copiee "
        "du pays alphabetiquement precedent",
    )

    years = range(config.START_YEAR, config.END_YEAR + 1)
    out = pd.DataFrame(
        [{"iso3": iso3, "year": year, "rain_mm": value}
         for iso3, value in modes.items() for year in years]
    )
    prep_log.add(
        "diffusion",
        f"1 valeur x {len(modes)} pays x {len(years)} annees -> aucun trou, 2003 comprise",
    )
    prep_log.add("lignes", f"{n_start:,} -> {len(out):,}")
    return out, prep_log


def prepare_pesticides(df: pd.DataFrame) -> tuple[pd.DataFrame, PrepLog]:
    """Total pesticide use per country and year, in tonnes of active ingredient."""
    prep_log = PrepLog()
    n_start = len(df)

    out = (
        _period(df, "Year")
        .pipe(_add_iso3, "Area")
        .dropna(subset=["iso3"])
        .rename(columns={"Year": "year", "Value": "pesticides_t"})
        .groupby(["iso3", "year"], as_index=False)["pesticides_t"]
        .mean()
    )
    prep_log.add("lignes", f"{n_start:,} -> {len(out):,}")
    return out, prep_log


def check_before_merge(sources: dict[str, tuple[pd.DataFrame, list[str]]]) -> pd.DataFrame:
    """Check that each source is unique on its key, before any join."""
    rows = []
    for name, (df, key) in sources.items():
        duplicates = int(df.duplicated(key).sum())
        rows.append(
            {
                "source": name,
                "lignes": len(df),
                "pays": df["iso3"].nunique(),
                "cle": " + ".join(key),
                "doublons": duplicates,
            }
        )
        if duplicates:
            raise AssertionError(f"{name} : la cle {key} n'est pas unique ({duplicates})")
    return pd.DataFrame(rows)


def merge_sources(
    yield_value: pd.DataFrame,
    temperature: pd.DataFrame,
    rain: pd.DataFrame,
    pesticides: pd.DataFrame,
) -> tuple[pd.DataFrame, PrepLog]:
    """Join the three explanatory sources onto the yield base.

    Each join is followed by a fan-out check: a left join must never change the
    number of rows. This is exactly the check that was missing from the build of
    `yield_df.csv`.
    """
    prep_log = PrepLog()
    dataset = yield_value
    prep_log.add("socle rendement", f"{len(dataset):,} lignes")

    for name, source in [
        ("temperature", temperature),
        ("pluie", rain),
        ("pesticides", pesticides),
    ]:
        before = len(dataset)
        dataset = dataset.merge(source, on=["iso3", "year"], how="left", validate="many_to_one")
        if len(dataset) != before:
            raise AssertionError(f"fan-out lors de la jointure {name} : {before} -> {len(dataset)}")
        prep_log.add(f"+ {name}", f"{before:,} lignes conservees, aucun fan-out")

    duplicates = int(dataset.duplicated(KEY).sum())
    if duplicates:
        raise AssertionError(f"la cle finale n'est pas unique ({duplicates} doublons)")
    prep_log.add("cle finale", f"{' + '.join(KEY)} unique")
    return dataset.reset_index(drop=True), prep_log


def coverage(df: pd.DataFrame, column: str) -> pd.Series:
    """Split a column's missing values between whole countries and isolated gaps.

    The distinction drives the treatment. If a column is missing for **an entire**
    country, there is nothing to recover and the country must be discarded. If it is
    only missing for some years, the country's known value can fill the gap without
    inventing anything.
    """
    by_country = df.groupby("iso3")[column].agg(["count", "size"])
    no_value = by_country[by_country["count"] == 0]
    partial = by_country[(by_country["count"] > 0) & (by_country["count"] < by_country["size"])]
    return pd.Series(
        {
            "lignes manquantes": int(df[column].isna().sum()),
            "pays sans aucune valeur": len(no_value),
            "lignes de ces pays": int(no_value["size"].sum()),
            "pays avec quelques annees manquantes": len(partial),
        },
        name=column,
    )


def coverage_table(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Apply :func:`coverage` to several columns, for side-by-side reading."""
    columns = columns or ["avg_temp", "rain_mm", "pesticides_t"]
    return pd.DataFrame({c: coverage(df, c) for c in columns})


#: Variables whose training domain is bounded.
DOMAIN_VARIABLES = ["yield_t_ha", "avg_temp", "rain_mm", "pesticides_t"]


def training_domain(
    df: pd.DataFrame, test_year: int | None = None, quantiles: tuple[float, float] = (0.05, 0.95)
) -> pd.DataFrame:
    """Usual bounds of each variable, crop by crop.

    The 5 % and 95 % quantiles describe the central 90 % observed. The API
    will use them to flag an unusual input **without rejecting it**: falling outside
    these bounds is not an error, it is a zone where the model learned little.

    `test_year` is excluded from the computation: the year held out for the final
    test of the yield model must not influence any upstream decision, interface bounds
    included.
    """
    test_year = config.TEST_YEAR if test_year is None else test_year
    before = df[df["year"] < test_year]
    low, high = quantiles

    rows = []
    for crop, group in before.groupby("crop"):
        for variable in DOMAIN_VARIABLES:
            values = group[variable].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "crop": crop,
                    "variable": variable,
                    "q05": values.quantile(low),
                    "q95": values.quantile(high),
                    "min": values.min(),
                    "max": values.max(),
                    "n": len(values),
                }
            )
    return pd.DataFrame(rows)


def build(verbose: bool = True) -> tuple[pd.DataFrame, dict[str, PrepLog]]:
    """Chain reading, preparation, checks and merge."""
    raw = load_sources()
    yield_value, r_yield = prepare_yield(raw["rendement"])
    temperature, r_temp = prepare_temperature(raw["temperature"])
    rain, r_rain = prepare_rain(raw["pluie"])
    pesticides, r_pest = prepare_pesticides(raw["pesticides"])

    check = check_before_merge(
        {
            "rendement": (yield_value, KEY),
            "temperature": (temperature, ["iso3", "year"]),
            "pluie": (rain, ["iso3", "year"]),
            "pesticides": (pesticides, ["iso3", "year"]),
        }
    )

    dataset, r_fusion = merge_sources(yield_value, temperature, rain, pesticides)

    prep_logs = {
        "rendement": r_yield,
        "temperature": r_temp,
        "pluie": r_rain,
        "pesticides": r_pest,
        "fusion": r_fusion,
    }
    if verbose:
        for name, prep_log in prep_logs.items():
            print(f"-- {name}\n{prep_log}\n")
        print(check.to_string(index=False))
        print(f"\ndataset : {len(dataset):,} lignes x {dataset.shape[1]} colonnes")
    return dataset, prep_logs


def main() -> None:
    config.PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    dataset, _ = build()
    dataset.to_csv(config.CROP_YIELD_PREDICTION_CSV, index=False)
    print(f"\necrit {config.CROP_YIELD_PREDICTION_CSV.relative_to(config.ROOT)}")

    domain = training_domain(dataset)
    domain.to_csv(config.TRAINING_DOMAIN_CSV, index=False)
    print(
        f"ecrit {config.TRAINING_DOMAIN_CSV.relative_to(config.ROOT)} "
        f"({len(domain)} bornes, annee {config.TEST_YEAR} exclue)"
    )


if __name__ == "__main__":
    main()
