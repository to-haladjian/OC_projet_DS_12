"""Data preparation: the rebuild of the FAO dataset and the simulated one.

Two kinds of tests. The first run on small hand-made frames and need no file: they pin
the decisions that make the rebuilt dataset trustworthy (ISO3 matching, no fan-out,
units). The second read the datasets actually produced and are skipped where the raw
data is absent — a fresh clone, or the CI.
"""

import pandas as pd
import pytest

from src import config
from src.data import crop_yield_prediction as cyp
from src.data import geo, preparation, prices
from src.data.crop_mapping import CROP_TO_FAO_ITEM, IMPUTATION_RULES

needs_fao_dataset = pytest.mark.skipif(
    not config.CROP_YIELD_PREDICTION_CSV.exists(), reason="jeu FAO non reconstruit (make data-fao)"
)


# --------------------------------------------------------------------------- #
# Country matching
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name, iso3", [
    ("United States of America", "USA"),
    ("Viet Nam", "VNM"), ("Vietnam", "VNM"),
    ("Turkey", "TUR"),
    ("Côte D'Ivoire", "CIV"),
    ("China, mainland", "CHN"), ("China", "CHN"),
    ("South Korea ", "KOR"),
])
def test_names_from_every_source_resolve(name, iso3):
    """The spellings a literal join loses are matched to their ISO3 code."""
    assert geo.to_iso3([name])[name] == iso3


@pytest.mark.parametrize("name", ["USSR", "Czechoslovakia", "Yugoslav SFR"])
def test_defunct_entities_have_no_code(name):
    assert geo.to_iso3([name])[name] is None


def test_position_never_defaults():
    with pytest.raises(KeyError):
        geo.position("ZZZ")


# --------------------------------------------------------------------------- #
# Rebuild of the FAO dataset, on hand-made sources
# --------------------------------------------------------------------------- #

def test_yield_is_converted_to_tonnes_and_the_china_aggregate_dropped():
    raw = pd.DataFrame({
        "Area": ["France", "China", "China, mainland"],
        "Item": ["Wheat"] * 3,
        "Year": [2000] * 3,
        "Value": [70_000, 40_000, 38_000],   # hg/ha
    })
    out, _ = cyp.prepare_yield(raw)
    assert set(out["iso3"]) == {"FRA", "CHN"}
    assert out.set_index("iso3").loc["FRA", "yield_t_ha"] == pytest.approx(7.0)
    assert out.set_index("iso3").loc["CHN", "yield_t_ha"] == pytest.approx(3.8)


def test_temperature_readings_are_aggregated_before_any_join():
    """Several readings for one country-year: the source of the provided file's duplicates."""
    raw = pd.DataFrame({
        "year": [2000, 2000, 2000, 2001],
        "country": ["India", "India", "India", "India"],
        "avg_temp": [24.0, 26.0, 28.0, 25.0],
    })
    out, _ = cyp.prepare_temperature(raw)
    assert len(out) == 2 and not out.duplicated(["iso3", "year"]).any()
    assert out.set_index("year").loc[2000, "avg_temp"] == pytest.approx(26.0)


def test_rain_is_one_value_per_country_with_no_missing_year():
    """The modal value discards an aberrant year; 2003, absent from the file, is filled."""
    years = [y for y in range(1990, 2014) if y != 2003]
    raw = pd.DataFrame({
        "Area": ["France"] * len(years),
        "Year": years,
        "average_rain_fall_mm_per_year": ["867"] * (len(years) - 1) + ["1940"],
    })
    out, _ = cyp.prepare_rain(raw)
    assert set(out["rain_mm"]) == {867.0}
    assert sorted(out["year"]) == list(range(config.START_YEAR, config.END_YEAR + 1))


def test_merge_keeps_the_row_count_and_a_unique_key():
    base = pd.DataFrame({"iso3": ["FRA", "FRA"], "area": ["France"] * 2, "year": [2000, 2000],
                         "crop": ["Wheat", "Maize"], "yield_t_ha": [7.0, 9.0]})
    by_year = pd.DataFrame({"iso3": ["FRA"], "year": [2000]})
    merged, _ = cyp.merge_sources(
        base, by_year.assign(avg_temp=12.0), by_year.assign(rain_mm=867.0),
        by_year.assign(pesticides_t=60_000.0),
    )
    assert len(merged) == len(base)
    assert not merged.duplicated(cyp.KEY).any()


def test_a_source_with_a_duplicated_key_is_refused():
    twice = pd.DataFrame({"iso3": ["FRA", "FRA"], "year": [2000, 2000], "avg_temp": [12.0, 13.0]})
    with pytest.raises(AssertionError):
        cyp.check_before_merge({"temperature": (twice, ["iso3", "year"])})


# --------------------------------------------------------------------------- #
# Simulated dataset and vocabulary
# --------------------------------------------------------------------------- #

def test_cleaning_drops_negative_yields_only():
    raw = pd.DataFrame({
        "Region": ["North"] * 3, "Soil_Type": ["Clay"] * 3, "Crop": ["Wheat"] * 3,
        "Rainfall_mm": [500.0, 120.0, 800.0], "Temperature_Celsius": [20.0, 16.0, 30.0],
        "Fertilizer_Used": ["True", "False", "False"], "Irrigation_Used": ["False"] * 3,
        "Weather_Condition": ["Sunny"] * 3, "Days_to_Harvest": [100, 110, 120],
        "Yield_tons_per_hectare": [4.5, -0.3, 5.1],
    })
    out, _ = preparation.clean(raw)
    assert len(out) == 2 and (out[preparation.TARGET] >= 0).all()
    assert out["Fertilizer_Used"].dtype == bool


def test_every_crop_without_an_fao_item_has_an_imputation_rule():
    assert {c for c, item in CROP_TO_FAO_ITEM.items() if item is None} == set(IMPUTATION_RULES)


def test_unit_errors_in_prices_are_discarded():
    """A price more than 10 times the median of its crop and year is a unit error."""
    table = pd.DataFrame({
        "iso3": ["FRA", "DEU", "ITA", "MRT"], "crop": ["Maize"] * 4, "year": [2000] * 4,
        "price_usd_t": [150.0, 160.0, 155.0, 23_000.0],
    })
    cleaned, _ = prices.clean(table)
    assert set(cleaned["iso3"]) == {"FRA", "DEU", "ITA"}


# --------------------------------------------------------------------------- #
# The datasets actually produced
# --------------------------------------------------------------------------- #

@needs_fao_dataset
def test_rebuilt_dataset_key_units_and_coverage():
    df = pd.read_csv(config.CROP_YIELD_PREDICTION_CSV)
    assert not df.duplicated(cyp.KEY).any()
    # Countries with every variable known: the 117 the API serves. The file provided
    # lost the ones below to a join on the literal country name.
    complete = df.dropna()
    assert complete["iso3"].nunique() == 117
    assert {"USA", "CHN", "RUS", "TUR", "VNM"} <= set(complete["iso3"])
    assert df["year"].between(config.START_YEAR, config.END_YEAR).all()
    # Tonnes per hectare: a value in hg/ha would be four orders of magnitude above.
    assert 0 < df["yield_t_ha"].median() < 20 and df["yield_t_ha"].max() < 100


def test_consolidated_dataset_is_clean():
    """The versioned deliverable: no missing value, no negative yield, 18 columns."""
    from src.data.build_dataset import DICTIONARY, load_consolidated

    df = load_consolidated()
    assert list(df.columns) == [column for column, *_ in DICTIONARY]
    assert not df.isna().any().any()
    assert (df["Yield_tons_per_hectare"] >= 0).all()
    assert set(df["Crop"]) == set(CROP_TO_FAO_ITEM)
