"""Derived `/recommend` variables: geography and past conditions.

All functions are **row-wise or per-country** transformations: they learn nothing
and never look at the target. They can therefore be applied before the split without
creating leakage, and in the same way on an API request.

Two decisive precautions:

* **Next year's conditions are unknown at recommendation time.** So the model cannot
  be given the temperature of the year it must predict. The variables `temp_hist` and
  `log_pest_hist` summarise the country's **previous years**.
* **The computation must not fill gaps.** Lags are obtained by shifting the
  country-year table then joining on `(iso3, year)`, not with `shift()`: a missing
  year stays missing instead of being replaced by the previous one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import geo

#: Lags used to summarise a country's past.
RETARDS = (1, 2, 3)


def ajoute_geographie(df: pd.DataFrame) -> pd.DataFrame:
    """Add `lat_abs`, `geo_x`, `geo_y` and `geo_z` from the `iso3` code.

    * `lat_abs`: absolute latitude, i.e. the distance to the equator. It separates
      temperate from tropical countries, which signed latitude does not — a country at
      -40 and one at +40 have comparable climates.
    * `geo_x`, `geo_y`, `geo_z`: position on the unit sphere. Two neighbouring countries
      have neighbouring values, **including on either side of the +-180 degree
      meridian**, which raw longitude does not guarantee.

    Giving the position rather than the country code lets the model share information
    between nearby countries, instead of treating each country as an isolated level. A
    code without coordinates raises an error: no silently wrong geography.
    """
    manquants = sorted(set(df["iso3"]) - set(geo.COORDONNEES))
    if manquants:
        raise KeyError(f"coordonnees absentes pour : {manquants}")

    longitudes = df["iso3"].map(lambda code: geo.COORDONNEES[code][0])
    latitudes = df["iso3"].map(lambda code: geo.COORDONNEES[code][1])
    lon, lat = np.radians(longitudes), np.radians(latitudes)
    return df.assign(
        lat_abs=latitudes.abs(),
        geo_x=np.cos(lat) * np.cos(lon),
        geo_y=np.cos(lat) * np.sin(lon),
        geo_z=np.sin(lat),
    )


def ajoute_log_pesticides(df: pd.DataFrame) -> pd.DataFrame:
    """Add `log_pesticides = log1p(pesticides_t)`.

    National tonnage spans seven orders of magnitude, from under one tonne to over a
    million: in raw value, almost all rows bunch up at the bottom of the scale.
    `log1p` rather than `log` because values close to zero exist.
    """
    return df.assign(log_pesticides=np.log1p(df["pesticides_t"]))


def ajoute_conditions_passees(df: pd.DataFrame, retards: tuple[int, ...] = RETARDS) -> pd.DataFrame:
    """Add `temp_hist`, `pest_hist` and `log_pest_hist`: means over previous years.

    For year *t* of a country, mean of `avg_temp` and `pesticides_t` over the
    **available** years *t−1*, *t−2* and *t−3*. Year *t* never enters its own
    history: no temporal leakage.

    The country-year table is shifted by each lag then joined on `(iso3, year)`. This
    detour avoids `shift()`, which would take the previous row of the table — not the
    previous year — and silently fill a missing year.

    The first year of each country has no history: its values are missing and the row
    will be left out of training.

    `df` must contain all relevant years of each country. At inference, provide it
    with the country's latest known years in addition to the row to predict.
    """
    pays_annee = df[["iso3", "year", "avg_temp", "pesticides_t"]].drop_duplicates(["iso3", "year"])
    decalees = pd.concat(
        [pays_annee.assign(year=pays_annee["year"] + retard) for retard in retards]
    )
    passe = decalees.groupby(["iso3", "year"])[["avg_temp", "pesticides_t"]].mean()
    passe.columns = ["temp_hist", "pest_hist"]

    resultat = df.join(passe, on=["iso3", "year"])
    return resultat.assign(log_pest_hist=np.log1p(resultat["pest_hist"]))


def prepare(df: pd.DataFrame, retards: tuple[int, ...] = RETARDS) -> pd.DataFrame:
    """Chain the three transformations: geography, log pesticides, past conditions."""
    return (
        df.pipe(ajoute_geographie)
        .pipe(ajoute_log_pesticides)
        .pipe(ajoute_conditions_passees, retards=retards)
    )


def controle_absence_de_fuite(df: pd.DataFrame, retards: tuple[int, ...] = RETARDS) -> None:
    """Check that `temp_hist` contains no information from the current year.

    Recomputes the expected history from prior years only and compares. This check is
    the safeguard of the temporal protocol: if it passes, no validation row knows its
    own future.
    """
    reference = df[["iso3", "year", "avg_temp"]].drop_duplicates(["iso3", "year"])
    attendu = {}
    temperatures = {(r.iso3, r.year): r.avg_temp for r in reference.itertuples()}
    for (iso3, annee) in temperatures:
        passees = [
            temperatures[(iso3, annee - retard)]
            for retard in retards
            if (iso3, annee - retard) in temperatures
        ]
        attendu[(iso3, annee)] = float(np.mean(passees)) if passees else np.nan

    obtenu = df.dropna(subset=["temp_hist"]).drop_duplicates(["iso3", "year"])
    for ligne in obtenu.itertuples():
        cible = attendu[(ligne.iso3, ligne.year)]
        if not np.isclose(ligne.temp_hist, cible, equal_nan=True):
            raise AssertionError(
                f"{ligne.iso3} {ligne.year} : temp_hist={ligne.temp_hist:.4f} "
                f"au lieu de {cible:.4f} — fuite temporelle possible"
            )
