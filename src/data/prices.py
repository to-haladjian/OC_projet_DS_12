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
   flagged in the `price_source` column.
"""

from __future__ import annotations

import zipfile

import pandas as pd

from src import config
from src.data import geo

#: FAO crop label of the yield dataset -> FAOSTAT price item code.
#: "Rice, paddy" is item 27 ("Rice"), whose CPC code 0113 is paddy rice. "Plantains and
#: others" maps to item 489, the only plantain item covering 1991-2013.
PRICE_CODE: dict[str, int] = {
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
ANNUAL_VALUE = 7021

#: Ratio to the world median of the same crop and year beyond which a value is
#: considered a unit error.
OUTLIER_FACTOR = 10

#: Number of years over which a country's reference price is taken.
YEAR_WINDOW = 5

SOURCE_COUNTRY = "pays"
SOURCE_WORLD = "médiane mondiale"


def extract(path=None) -> pd.DataFrame:
    """Read the FAOSTAT archive and keep annual USD/t prices of the 10 crops, 1990-2013.

    The file is read in chunks: uncompressed, it weighs over 200 MB.
    """
    path = config.SOURCE_PRICES if path is None else path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} absent : telecharger l'archive FAOSTAT depuis {config.SOURCE_PRICES_URL}"
        )

    to_crop = {code: crop for crop, code in PRICE_CODE.items()}
    chunks = []
    with zipfile.ZipFile(path) as archive, archive.open(
        "Prices_E_All_Data_(Normalized).csv"
    ) as file:
        for block in pd.read_csv(
            file, chunksize=500_000, encoding="utf-8",
            usecols=["Area", "Item Code", "Element Code", "Year", "Months Code", "Value"],
        ):
            kept = (
                (block["Element Code"] == ELEMENT_USD_TONNE)
                & (block["Months Code"] == ANNUAL_VALUE)
                & block["Item Code"].isin(to_crop)
                & block["Year"].between(config.START_YEAR, config.END_YEAR)
            )
            chunks.append(block[kept])

    raw = pd.concat(chunks, ignore_index=True)
    matches = geo.to_iso3(raw["Area"].unique())
    return pd.DataFrame(
        {
            "iso3": raw["Area"].map(matches),
            "crop": raw["Item Code"].map(to_crop),
            "year": raw["Year"].astype(int),
            "price_usd_t": raw["Value"].astype(float),
        }
    )


def clean(prices: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop rows without a country code and probable unit errors.

    Returns the cleaned table and the count of each removal, to be printed.
    """
    without_country = prices["iso3"].isna()
    prices = prices[~without_country]

    median_value = prices.groupby(["crop", "year"])["price_usd_t"].transform("median")
    ratio = prices["price_usd_t"] / median_value
    outlier = (ratio > OUTLIER_FACTOR) | (ratio < 1 / OUTLIER_FACTOR)

    cleaned = prices[~outlier].sort_values(["iso3", "crop", "year"]).reset_index(drop=True)
    if cleaned.duplicated(["iso3", "crop", "year"]).any():
        raise AssertionError("plusieurs prix pour un meme couple pays x culture x annee")
    return cleaned, {
        "lignes lues": int(len(without_country)),
        "entites sans code pays (agregats, entites disparues)": int(without_country.sum()),
        f"erreurs d'unite probables (facteur > {OUTLIER_FACTOR})": int(outlier.sum()),
        "lignes conservees": int(len(cleaned)),
    }


def load() -> pd.DataFrame:
    """Read the cleaned prices produced by `main()`."""
    return pd.read_csv(config.PRODUCER_PRICES_CSV)


def reference_prices(
    prices: pd.DataFrame, iso3: str, year: int, crops=None
) -> pd.DataFrame:
    """Reference price of each crop for a country, as known at `year`.

    Median of the country's prices over the `YEAR_WINDOW` years up to `year`
    included; failing that, world median over the same window. No price after
    `year` is ever used.
    """
    crops = list(PRICE_CODE) if crops is None else list(crops)
    window = prices[prices["year"].between(year - YEAR_WINDOW + 1, year)]
    of_country = window[window["iso3"] == iso3].groupby("crop")["price_usd_t"].median()
    world = window.groupby("crop")["price_usd_t"].median()

    rows = []
    for crop in crops:
        if crop in of_country.index:
            rows.append((crop, float(of_country[crop]), SOURCE_COUNTRY))
        elif crop in world.index:
            rows.append((crop, float(world[crop]), SOURCE_WORLD))
        else:
            raise KeyError(f"aucun prix pour {crop} entre {year - YEAR_WINDOW + 1} et {year}")
    return pd.DataFrame(rows, columns=["crop", "price_usd_t", "price_source"]).set_index("crop")


def price_per_row(prices: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """Reference price for each (iso3, year, crop) row, vectorised for evaluation.

    Same rule as :func:`reference_prices`, applied to every row of `rows`: the price
    known at the row's year, never a later one.
    """
    result = pd.Series(index=rows.index, dtype=float)
    for (iso3, year), block in rows.groupby(["iso3", "year"]):
        reference = reference_prices(prices, iso3, int(year), block["crop"].unique())
        result.loc[block.index] = block["crop"].map(reference["price_usd_t"]).to_numpy()
    return result


def main() -> None:
    cleaned, tally = clean(extract())
    for label, n in tally.items():
        print(f"{label:55} {n:7,}")
    print(f"{'pays couverts':55} {cleaned['iso3'].nunique():7,}")
    config.PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(config.PRODUCER_PRICES_CSV, index=False)
    print(f"\necrit {config.PRODUCER_PRICES_CSV.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
