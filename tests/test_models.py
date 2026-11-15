"""Training code: splits, history features, metrics, tracking schema and a small retrain.

Everything runs on small generated frames, so none of it needs the raw data: these
tests guard the code the served numbers come from, wherever the suite runs.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

from src import config
from src.models import evaluation, features, history, practice_model, protocol
from src.models import yield_model as ym


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """A small country x crop x year panel shaped like the rebuilt FAO dataset."""
    rng = np.random.default_rng(config.SEED)
    countries = ["FRA", "DEU", "ITA", "ESP", "BRA", "IND", "USA", "CHN", "EGY", "ZAF"]
    rows = []
    for iso3 in countries:
        level = rng.uniform(0.6, 1.4)
        for crop, base in (("Wheat", 3.0), ("Maize", 5.0), ("Potatoes", 18.0)):
            for year in range(2000, config.TEST_YEAR + 1):
                rows.append({
                    "iso3": iso3, "area": iso3, "year": year, "crop": crop,
                    "yield_t_ha": base * level * (1 + 0.01 * (year - 2000)) + rng.normal(0, 0.1),
                    "avg_temp": 15 + rng.normal(0, 0.5), "rain_mm": 800.0,
                    "pesticides_t": 1000 * level,
                })
    return history.prepare(pd.DataFrame(rows)).dropna().reset_index(drop=True)


# --------------------------------------------------------------------------- #
# History features: no knowledge of the future
# --------------------------------------------------------------------------- #

def test_history_never_contains_the_current_year():
    df = pd.DataFrame({
        "iso3": ["FRA"] * 4, "year": [2000, 2001, 2002, 2003],
        "avg_temp": [10.0, 20.0, 30.0, 99.0], "pesticides_t": [1.0, 2.0, 3.0, 4.0],
    })
    out = history.add_past_conditions(df).set_index("year")
    assert np.isnan(out.loc[2000, "temp_hist"])                 # no past at all
    assert out.loc[2001, "temp_hist"] == pytest.approx(10.0)
    assert out.loc[2003, "temp_hist"] == pytest.approx(20.0)    # 99.0 is not in it
    history.check_no_leakage(out.reset_index())


def test_a_missing_year_is_not_filled_by_the_previous_row():
    df = pd.DataFrame({
        "iso3": ["FRA"] * 3, "year": [2000, 2001, 2005],
        "avg_temp": [10.0, 20.0, 30.0], "pesticides_t": [1.0, 2.0, 3.0],
    })
    out = history.add_past_conditions(df).set_index("year")
    assert np.isnan(out.loc[2005, "temp_hist"])   # 2002-2004 are unknown, not 2001


def test_leakage_is_detected():
    df = pd.DataFrame({"iso3": ["FRA"] * 3, "year": [2000, 2001, 2002],
                       "avg_temp": [10.0, 20.0, 30.0], "temp_hist": [np.nan, 20.0, 15.0]})
    with pytest.raises(AssertionError, match="fuite"):
        history.check_no_leakage(df)


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #

def test_temporal_folds_train_on_the_past_only(panel):
    years = panel["year"].to_numpy()
    for (training, validation), year in zip(
        protocol.temporal_cv(panel["year"], [2010, 2011, 2012]), [2010, 2011, 2012]
    ):
        assert (years[training] < year).all() and (years[validation] == year).all()


def test_the_test_year_cannot_be_a_validation_year(panel):
    with pytest.raises(AssertionError):
        protocol.temporal_cv(panel["year"], [config.TEST_YEAR])


def test_temporal_split_isolates_the_test_year(panel):
    X_train, X_test, _, _ = protocol.temporal_split(panel, ["year"], features.YIELD_TARGET)
    assert (X_train["year"] < config.TEST_YEAR).all() and (X_test["year"] == config.TEST_YEAR).all()
    assert len(X_train) + len(X_test) == len(panel)


def test_country_folds_never_share_a_country(panel):
    countries = panel["iso3"].to_numpy()
    seen = []
    for training, validation in protocol.country_cv(panel["iso3"], n_folds=5):
        assert not set(countries[training]) & set(countries[validation])
        seen.extend(set(countries[validation]))
    assert sorted(seen) == sorted(set(countries))   # each country validated exactly once


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def test_cross_validation_yields_the_tracked_metrics(panel):
    folds = evaluation.validate_by_fold(
        ym.pipeline(LinearRegression()), panel[ym.VARIABLES], panel[features.YIELD_TARGET],
        protocol.temporal_cv(panel["year"], [2011, 2012]),
    )
    assert set(evaluation.summarise(folds)) == set(evaluation.CV_METRICS)


def test_nrmse_has_no_unit():
    observed = np.array([2.0, 4.0, 6.0])
    assert evaluation.nrmse(observed, observed + 1.0) == pytest.approx(0.25)
    assert evaluation.nrmse(observed * 10, (observed + 1.0) * 10) == pytest.approx(0.25)


def test_ranking_quality_counts_first_places_and_regret():
    observed = pd.DataFrame({
        "iso3": ["FRA"] * 3 + ["IND"] * 3, "year": [2012] * 6,
        "crop": ["Wheat", "Maize", "Rice"] * 2,
        "yield_t_ha": [7.0, 9.0, 5.0, 3.0, 2.5, 3.5],
    })
    predicted = pd.Series([6.0, 8.0, 5.0, 3.2, 2.0, 3.0])   # right in FRA, wrong in IND
    quality = evaluation.ranking_quality(observed, predicted)
    assert quality["top1"] == pytest.approx(0.5)
    assert quality["regret"] == pytest.approx((0.0 + 0.5) / 2)
    assert quality["groups"] == 2


def test_paired_comparison_counts_improved_folds():
    fold = lambda rmse: pd.DataFrame({"rmse": rmse, "mae": rmse, "r2": [0.9] * 3})  # noqa: E731
    table = evaluation.compare_paired(
        {"reference": fold([1.0, 1.0, 1.0]), "candidate": fold([0.9, 1.1, 0.8])},
        reference="reference",
    )
    assert table.loc["candidate", "plis ameliores"] == "2/3"


# --------------------------------------------------------------------------- #
# A small retrain
# --------------------------------------------------------------------------- #

def test_yield_pipeline_learns_the_crop_levels(panel):
    """Trained on the past of a small panel, the pipeline ranks the crops correctly."""
    X_train, X_test, y_train, y_test = protocol.temporal_split(
        panel, ym.VARIABLES, features.YIELD_TARGET)
    model = ym.pipeline(ExtraTreesRegressor(n_estimators=30, random_state=config.SEED))
    predictions = model.fit(X_train, y_train).predict(X_test)
    assert evaluation.metrics(y_test, predictions)["test_r2"] > 0.9

    test = panel[panel["year"] == config.TEST_YEAR]
    quality = evaluation.ranking_quality(test, pd.Series(predictions, index=test.index))
    assert quality["top1"] == 1.0


def test_served_model_matches_its_metadata():
    """The pickle that is served was trained on the variables its metadata declares."""
    import json

    metadata = json.loads((config.MODELS_DIR / "yield_model_metadata.json").read_text())
    assert metadata["variables"] == ym.VARIABLES
    assert metadata["settings"] == ym.SETTINGS
    assert set(metadata["cv_temporal"]) == set(evaluation.CV_METRICS)
    assert metadata["cv_leave_country_out"]["cv_rmse_mean"] > metadata["cv_temporal"]["cv_rmse_mean"]


def test_practice_model_recovers_the_generator():
    """On plots drawn from the documented generator, the coefficients come back."""
    rng = np.random.default_rng(config.SEED)
    n = 20_000
    X = pd.DataFrame({
        "Rainfall_mm": rng.uniform(100, 1000, n), "Temperature_Celsius": rng.uniform(15, 40, n),
        "Fertilizer_Used": rng.integers(0, 2, n).astype(float),
        "Irrigation_Used": rng.integers(0, 2, n).astype(float),
    })
    generator = practice_model.GENERATOR
    y = sum(X[name] * effect for name, effect in generator.items()) + rng.normal(0, 0.5, n)
    model = practice_model.pipeline(LinearRegression(), list(X.columns)).fit(X, y)
    practice_model.check_generator(model, list(X.columns), tolerance=0.05)

    wrong = practice_model.pipeline(LinearRegression(), list(X.columns)).fit(X, 2 * y)
    with pytest.raises(AssertionError):
        practice_model.check_generator(wrong, list(X.columns), tolerance=0.05)


# --------------------------------------------------------------------------- #
# Tracking schema
# --------------------------------------------------------------------------- #

@pytest.fixture
def tracking(tmp_path, monkeypatch):
    """The tracking module, pointed at a throwaway store."""
    pytest.importorskip("mlflow")
    from src import tracking as module

    monkeypatch.setattr(config, "MLFLOW_DB", tmp_path / "mlflow.db")
    monkeypatch.setattr(config, "MLFLOW_ARTIFACTS", tmp_path / "mlruns")
    module.configure(module.EXPERIMENT_YIELD)
    return module


def _cv_metrics(rmse: float) -> dict[str, float]:
    return {name: rmse for name in evaluation.CV_METRICS}


def test_family_is_read_through_the_pipeline(tracking):
    assert tracking.family(ym.selected_model()) == "ExtraTreesRegressor"
    assert tracking.family(LinearRegression()) == "LinearRegression"


def test_an_incomplete_run_is_refused(tracking):
    description = tracking.describe(LinearRegression(), tracking.DATASET_FAO, "history", "temporal")
    with pytest.raises(ValueError, match="metriques manquantes"):
        tracking.log_evaluation("x", description, {"cv_rmse_mean": 1.0}, "reference")
    with pytest.raises(ValueError, match="description incomplete"):
        tracking.log_evaluation("x", {"model_family": "x"}, _cv_metrics(1.0), "reference")
    with pytest.raises(ValueError, match="metriques manquantes"):   # a final run needs test metrics
        tracking.log_evaluation("x", description, _cv_metrics(1.0), "final_evaluation")


def test_every_run_lands_in_the_comparison_table(tracking):
    for model, rmse in ((LinearRegression(), 2.0), (ym.selected_model(), 1.0)):
        tracking.log_evaluation(
            tracking.family(model),
            tracking.describe(model, tracking.DATASET_FAO, "history", "temporal"),
            _cv_metrics(rmse), "reference",
        )
    table = tracking.comparison_table((tracking.EXPERIMENT_YIELD,))
    assert list(table["model_family"]) == ["ExtraTreesRegressor", "LinearRegression"]   # best first
    assert table[["dataset", "feature_set", "protocol", "phase", "cv_nrmse_mean"]].notna().all().all()


def test_search_candidates_carry_the_full_metrics(tracking, panel):
    from src.models import tuning

    pipe = ym.pipeline(ExtraTreesRegressor(n_estimators=10, random_state=config.SEED))
    table, best = tuning.random_search(
        pipe, {"model__max_features": [0.5, 1.0]}, panel[ym.VARIABLES],
        panel[features.YIELD_TARGET], protocol.temporal_cv(panel["year"], [2011, 2012]),
        tracking.describe(pipe, tracking.DATASET_FAO, "history", "temporal"), n_iter=2,
    )
    assert set(evaluation.CV_METRICS) <= set(table.columns)
    assert best == {"max_features": table.loc[0, "max_features"]}
    runs = tracking.comparison_table((tracking.EXPERIMENT_YIELD,))
    assert len(runs) == 3 and (runs["phase"] == "optimisation").all()   # parent + 2 candidates
