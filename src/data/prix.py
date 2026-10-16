"""Producer prices, to turn an expected yield into an expected gross product.

**Why this module exists.** The brief asks for "the most profitable crop" and lists
profitability among the metrics. Neither dataset carries an economic variable. Rather
than giving up on the question, we bring in a third source, documented and free:
FAOSTAT producer prices (`Producer Price (USD/tonne)`), per country, crop and year.

What this allows, and what it does not:

* **gross product** = expected yield (t/ha) x producer price (USD/t), in USD/ha. It is
  what the harvest sells for at the farm gate;
* **not a margin**: production costs (seed, fertiliser, labour, irrigation) are
  unknown. A potato sells for more per hectare than wheat, but also costs much more to
  grow. The interface must say "gross product", never "profit".

Prices are a **post-processing hypothesis**, not a model input: they are applied to
the model's predicted yields, and the user can override them.

Three decisions, each measured in `main()`:

1. **Unit errors are discarded.** A value more than 10 times above or below the world
   median of the same crop and year is a probable local-currency figure. Mauritanian
   maize, 1991-1998, sits around 150 times the median. This removes 0.5 % of the values.
2. **A country's price is the median of its last 5 known years**, not a single year:
   producer prices swing with harvests and exchange rates.
3. **Fallback to the world median** when the country publishes no price for the crop,
   flagged in the `prix_source` column.
"""

from __future__ import annotations

import zipfile

import pandas as pd

from src import config
from src.data import geo

#: FAO crop label of the yield dataset -> FAOSTAT price item code.
#: "Rice, paddy" is item 27 ("Rice"), whose CPC code 0113 is paddy rice. "Plantains and
#: others" maps to item 489, the only plantain item covering 1991-2013.
CODE_PRIX: dict[str, int] = {
    "Cassava": 125,
    "Maize": 56,
    "Plantains and others": 489,
    "Potatoes": 116,
    "Rice, paddy": 27,
    "Sorghum": 83,
    "Soybeans": 236,
    "Sweet potatoes": 122,
    "Wheat": 15,
    "Yams": 137,
}

ELEMENT_USD_TONNE = 5532
VALEUR_ANNUELLE = 7021

#: Ratio to the world median of the same crop and year beyond which a value is
#: considered a unit error.
FACTEUR_ABERRANT = 10

#: Number of years over which a country's reference price is taken.
FENETRE_ANNEES = 5

SOURCE_PAYS = "pays"
SOURCE_MONDE = "médiane mondiale"


def extrait(chemin=None) -> pd.DataFrame:
    """Read the FAOSTAT archive and keep annual USD/t prices of the 10 crops, 1990-2013.

    The file is read in chunks: uncompressed, it weighs over 200 MB.
    """
    chemin = config.SOURCE_PRIX if chemin is None else chemin
    if not chemin.exists():
        raise FileNotFoundError(
            f"{chemin} absent : telecharger l'archive FAOSTAT depuis {config.SOURCE_PRIX_URL}"
        )

    vers_culture = {code: culture for culture, code in CODE_PRIX.items()}
    morceaux = []
    with zipfile.ZipFile(chemin) as archive, archive.open(
        "Prices_E_All_Data_(Normalized).csv"
    ) as fichier:
        for bloc in pd.read_csv(
            fichier, chunksize=500_000, encoding="utf-8",
            usecols=["Area", "Item Code", "Element Code", "Year", "Months Code", "Value"],
        ):
            garde = (
                (bloc["Element Code"] == ELEMENT_USD_TONNE)
                & (bloc["Months Code"] == VALEUR_ANNUELLE)
                & bloc["Item Code"].isin(vers_culture)
                & bloc["Year"].between(config.ANNEE_DEBUT, config.ANNEE_FIN)
            )
            morceaux.append(bloc[garde])

    brut = pd.concat(morceaux, ignore_index=True)
    correspondances = geo.vers_iso3(brut["Area"].unique())
    return pd.DataFrame(
        {
            "iso3": brut["Area"].map(correspondances),
            "crop": brut["Item Code"].map(vers_culture),
            "year": brut["Year"].astype(int),
            "prix_usd_t": brut["Value"].astype(float),
        }
    )


def nettoie(prix: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop rows without a country code and probable unit errors.

    Returns the cleaned table and the count of each removal, to be printed.
    """
    sans_pays = prix["iso3"].isna()
    prix = prix[~sans_pays]

    mediane = prix.groupby(["crop", "year"])["prix_usd_t"].transform("median")
    ratio = prix["prix_usd_t"] / mediane
    aberrant = (ratio > FACTEUR_ABERRANT) | (ratio < 1 / FACTEUR_ABERRANT)

    propre = prix[~aberrant].sort_values(["iso3", "crop", "year"]).reset_index(drop=True)
    if propre.duplicated(["iso3", "crop", "year"]).any():
        raise AssertionError("plusieurs prix pour un meme couple pays x culture x annee")
    return propre, {
        "lignes lues": int(len(sans_pays)),
        "entites sans code pays (agregats, entites disparues)": int(sans_pays.sum()),
        f"erreurs d'unite probables (facteur > {FACTEUR_ABERRANT})": int(aberrant.sum()),
        "lignes conservees": int(len(propre)),
    }


def charge() -> pd.DataFrame:
    """Read the cleaned prices produced by `main()`."""
    return pd.read_csv(config.PRIX_PRODUCTEUR_CSV)


def prix_reference(
    prix: pd.DataFrame, iso3: str, annee: int, cultures=None
) -> pd.DataFrame:
    """Reference price of each crop for a country, as known at `annee`.

    Median of the country's prices over the `FENETRE_ANNEES` years up to `annee`
    included; failing that, world median over the same window. No price after
    `annee` is ever used.
    """
    cultures = list(CODE_PRIX) if cultures is None else list(cultures)
    fenetre = prix[prix["year"].between(annee - FENETRE_ANNEES + 1, annee)]
    du_pays = fenetre[fenetre["iso3"] == iso3].groupby("crop")["prix_usd_t"].median()
    monde = fenetre.groupby("crop")["prix_usd_t"].median()

    lignes = []
    for culture in cultures:
        if culture in du_pays.index:
            lignes.append((culture, float(du_pays[culture]), SOURCE_PAYS))
        elif culture in monde.index:
            lignes.append((culture, float(monde[culture]), SOURCE_MONDE))
        else:
            raise KeyError(f"aucun prix pour {culture} entre {annee - FENETRE_ANNEES + 1} et {annee}")
    return pd.DataFrame(lignes, columns=["crop", "prix_usd_t", "prix_source"]).set_index("crop")


def prix_par_ligne(prix: pd.DataFrame, lignes: pd.DataFrame) -> pd.Series:
    """Reference price for each (iso3, year, crop) row, vectorised for evaluation.

    Same rule as :func:`prix_reference`, applied to every row of `lignes`: the price
    known at the row's year, never a later one.
    """
    resultat = pd.Series(index=lignes.index, dtype=float)
    for (iso3, annee), bloc in lignes.groupby(["iso3", "year"]):
        reference = prix_reference(prix, iso3, int(annee), bloc["crop"].unique())
        resultat.loc[bloc.index] = bloc["crop"].map(reference["prix_usd_t"]).to_numpy()
    return resultat


def main() -> None:
    propre, bilan = nettoie(extrait())
    for libelle, n in bilan.items():
        print(f"{libelle:55} {n:7,}")
    print(f"{'pays couverts':55} {propre['iso3'].nunique():7,}")
    config.DATA_TRAITE.mkdir(parents=True, exist_ok=True)
    propre.to_csv(config.PRIX_PRODUCTEUR_CSV, index=False)
    print(f"\necrit {config.PRIX_PRODUCTEUR_CSV.relative_to(config.RACINE)}")


if __name__ == "__main__":
    main()
