"""Project paths and shared constants.

A single place declares where the data lives, so that the notebooks, the training
pipeline and the API never redefine a path.
"""

from __future__ import annotations

from pathlib import Path

#: Single seed, so that everything is reproducible end to end.
SEED = 42

ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = ROOT / "data" / "raw"
PROCESSED_DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

#: Dataset A: simulated observations at the plot level.
AGRICULTURE_CROP_YIELD = RAW_DATA / "Agriculture Crop Yield" / "crop_yield.csv"

#: Dataset B: the four raw sources of CropYield Prediction.
CROP_YIELD_PREDICTION = RAW_DATA / "Crop Yield Prediction Dataset"
SOURCE_YIELD = CROP_YIELD_PREDICTION / "yield.csv"
SOURCE_TEMPERATURE = CROP_YIELD_PREDICTION / "temp.csv"
SOURCE_RAIN = CROP_YIELD_PREDICTION / "rainfall.csv"
SOURCE_PESTICIDES = CROP_YIELD_PREDICTION / "pesticides.csv"

#: The pre-merged file shipped with the brief. Kept for comparison only:
#: it was built by joining on the literal country name and keeps only
#: 101 of them (see src/data/geo.py). We rebuild it instead.
SOURCE_PREMERGED = CROP_YIELD_PREDICTION / "yield_df.csv"

#: Source C, outside the brief: FAOSTAT producer prices (domain "Prices", bulk download,
#: free). It only feeds the economic reading of the crop ranking; no model
#: learns from it.
SOURCE_PRICES = RAW_DATA / "FAOSTAT Prices" / "Prices_E_All_Data_(Normalized).zip"
SOURCE_PRICES_URL = (
    "https://bulks-faostat.fao.org/production/Prices_E_All_Data_(Normalized).zip"
)

#: Outputs.
CROP_YIELD_PREDICTION_CSV = PROCESSED_DATA / "crop_yield_prediction.csv"
PRODUCER_PRICES_CSV = PROCESSED_DATA / "producer_prices.csv"
CONSOLIDATED_DATASET_CSV = PROCESSED_DATA / "crop_dataset_consolidated.csv"

#: Same dataset, gzipped. This is the versioned deliverable: the plain CSV weighs
#: 174 MB, past GitHub's 100 MB per-file limit, while the gzipped one weighs 43 MB.
#: Nothing is lost — pandas reads either transparently — so the repository can be
#: cloned and used without regenerating anything.
CONSOLIDATED_DATASET_GZ = PROCESSED_DATA / "crop_dataset_consolidated.csv.gz"
DATA_DICTIONARY = REPORTS / "data_dictionary.md"

#: Bounds of the training domain, per crop and per variable. The API
#: uses them to flag an unusual input without rejecting it.
TRAINING_DOMAIN_CSV = PROCESSED_DATA / "training_domain.csv"

#: Period shared by the four CropYield Prediction sources.
START_YEAR, END_YEAR = 1990, 2013

#: Experiment tracking. The store is a **local SQLite database**: a single file,
#: no server, no remote service, no cost. MLflow put its old "file" store into
#: maintenance mode as of version 3.16; SQLite is the supported path and stays
#: entirely local. Artifacts stay in `mlruns/`.
MLFLOW_DB = ROOT / "mlflow.db"
MLFLOW_ARTIFACTS = ROOT / "mlruns"

#: Trained models and their metadata.
MODELS_DIR = ROOT / "models"
PRACTICE_MODEL_PATH = MODELS_DIR / "practice_model.joblib"
YIELD_MODEL_PATH = MODELS_DIR / "yield_model.joblib"

#: Last year of the historical dataset, held out for the final test of the yield model.
#: The training domain bounds are computed excluding it, so that no information
#: from the test year leaks upstream.
TEST_YEAR = 2013
