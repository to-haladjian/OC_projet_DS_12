# Replay every stage of the project. `make help` lists the targets.
#
# Each stage is its own target, so one can be replayed alone; the aggregates
# (`data`, `train`, `all`) chain them in dependency order.

PY := .venv/bin/python
export MLFLOW_DISABLE_AGENT_HINT := 1

.DEFAULT_GOAL := help
.PHONY: help install data data-fao data-consolidated data-prices \
        train train-practice train-yield assets \
        select-yield compare report deliverables notebooks check-notebooks test lint all mlflow-ui api app up down

help:  ## List the targets
	@grep -E '^[a-z][a-z-]*:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "} {printf "  make %-18s %s\n", $$1, $$2}'

install:  ## Create the virtual environment and install every extra
	uv venv
	uv pip install -e ".[notebook,api,app,dev,report]"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

data-fao:  ## Rebuild the FAO dataset from its four raw sources
	$(PY) -m src.data.crop_yield_prediction

data-consolidated:  ## Build the consolidated dataset (simulated plots + FAO profiles)
	$(PY) -m src.data.build_dataset

data-prices:  ## Clean the FAOSTAT producer prices
	$(PY) -m src.data.prices

data: data-fao data-consolidated data-prices  ## All three datasets, in order

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

train-practice:  ## Train the practice-effects model (simulated plots)
	$(PY) -m src.models.practice_model

train-yield:  ## Train the yield model served by /predict and /recommend (FAO data)
	$(PY) -m src.models.yield_model

assets:  ## Write the JSON assets the API serves next to the models
	$(PY) -m src.models.serving_assets

train: train-practice train-yield assets  ## Both models, then the serving assets

select-yield:  ## Compare the candidate families of the yield model, under both protocols
	$(PY) -m src.models.selection

# ---------------------------------------------------------------------------
# Notebooks, tests
# ---------------------------------------------------------------------------

notebooks:  ## Re-execute the ten notebooks in place (long)
	cd notebooks && for nb in *.ipynb; do \
		../$(PY) -m jupyter nbconvert --execute --to notebook --inplace "$$nb" || exit 1; \
	done

check-notebooks:  ## Check that every notebook compiles, ran, and raised no error (local tool, not versioned)
	$(PY) scripts/check_notebooks.py

test:  ## Run the test suite
	$(PY) -m pytest -q

lint:  ## Check the code with ruff
	$(PY) -m ruff check .

all: data train lint test  ## From the raw data to linted, tested models

# ---------------------------------------------------------------------------
# Experiment tracking, serving
# ---------------------------------------------------------------------------

compare:  ## Print every tracked run on the shared metrics, and write the CSV
	$(PY) -m src.tracking

report:  ## Build the business report PDF (after `make compare`) (local tool, not versioned)
	$(PY) scripts/build_report.py

deliverables: report check-notebooks  ## Rebuild the report, check the notebooks, zip the deliverables and test the zip (local tool, not versioned)
	$(PY) scripts/build_deliverables.py --verify

mlflow-ui:  ## Browse the tracked runs on http://127.0.0.1:5000
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db

api:  ## Serve the API on http://127.0.0.1:8000
	$(PY) -m uvicorn api.main:app --reload

app:  ## Run the interface on http://localhost:8501 (API started first)
	$(PY) -m streamlit run app/app.py

up:  ## API and interface together, in containers
	docker compose up --build

down:  ## Stop the containers
	docker compose down
