"""Project paths and shared constants.

A single place declares where the data lives, so that the notebooks, the training
pipeline (step 2) and the API (step 3) never redefine a path.
"""

from __future__ import annotations

from pathlib import Path

#: Single seed, so that everything is reproducible end to end.
SEED = 42

RACINE = Path(__file__).resolve().parents[1]

DATA_BRUT = RACINE / "data" / "raw"
DATA_TRAITE = RACINE / "data" / "processed"
RAPPORTS = RACINE / "reports"
FIGURES = RAPPORTS / "figures"

#: Dataset A: simulated observations at the plot level.
AGRICULTURE_CROP_YIELD = DATA_BRUT / "Agriculture Crop Yield" / "crop_yield.csv"

#: Dataset B: the four raw sources of CropYield Prediction.
CROP_YIELD_PREDICTION = DATA_BRUT / "Crop Yield Prediction Dataset"
SOURCE_RENDEMENT = CROP_YIELD_PREDICTION / "yield.csv"
SOURCE_TEMPERATURE = CROP_YIELD_PREDICTION / "temp.csv"
SOURCE_PLUIE = CROP_YIELD_PREDICTION / "rainfall.csv"
SOURCE_PESTICIDES = CROP_YIELD_PREDICTION / "pesticides.csv"

#: The pre-merged file shipped with the brief. Kept for comparison only:
#: it was built by joining on the literal country name and keeps only
#: 101 of them (see src/data/geo.py). We rebuild it instead.
SOURCE_PREFUSIONNEE = CROP_YIELD_PREDICTION / "yield_df.csv"

#: Source C, outside the brief: FAOSTAT producer prices (domain "Prices", bulk download,
#: free). It only feeds the economic reading of the `/recommend` ranking; no model
#: learns from it.
SOURCE_PRIX = DATA_BRUT / "FAOSTAT Prices" / "Prices_E_All_Data_(Normalized).zip"
SOURCE_PRIX_URL = (
    "https://bulks-faostat.fao.org/production/Prices_E_All_Data_(Normalized).zip"
)

#: Outputs.
CROP_YIELD_PREDICTION_CSV = DATA_TRAITE / "crop_yield_prediction.csv"
PRIX_PRODUCTEUR_CSV = DATA_TRAITE / "prix_producteur.csv"
DATASET_CONSOLIDE_CSV = DATA_TRAITE / "crop_dataset_consolidated.csv"
DICTIONNAIRE_DONNEES = RAPPORTS / "data_dictionary.md"

#: Bounds of the training domain, per crop and per variable. The step 3 API
#: uses them to flag an unusual input without rejecting it.
DOMAINE_APPRENTISSAGE_CSV = DATA_TRAITE / "domaine_apprentissage.csv"

#: Period shared by the four CropYield Prediction sources.
ANNEE_DEBUT, ANNEE_FIN = 1990, 2013

#: Experiment tracking. The store is a **local SQLite database**: a single file,
#: no server, no remote service, no cost. MLflow put its old "file" store into
#: maintenance mode as of version 3.16; SQLite is the supported path and stays
#: entirely local. Artifacts stay in `mlruns/`.
MLFLOW_BASE = RACINE / "mlflow.db"
MLFLOW_ARTEFACTS = RACINE / "mlruns"

#: Trained models and their metadata.
MODELES = RACINE / "models"
MODELE_PREDICT = MODELES / "predict_model.joblib"
MODELE_RECOMMEND = MODELES / "recommend_model.joblib"

#: Last year of the historical dataset, held out for the final `/recommend` test (step 2).
#: The training domain bounds are computed excluding it, so that no information
#: from the test year leaks upstream.
ANNEE_TEST = 2013
