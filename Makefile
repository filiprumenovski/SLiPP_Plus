.PHONY: help install ingest train eval figures all scratch build-caver-t12 build-lipid-boundary hierarchical-lipid test lint typecheck clean

PY ?= uv run
CFG ?= configs/day1.yaml

help:
	@echo "slipp_plus targets:"
	@echo "  install     uv sync (installs deps + dev extras)"
	@echo "  ingest      CSV + xlsx -> validated parquets (Rule 1 gate)"
	@echo "  train       25 iterations x 3 models (RF, XGB, LGBM)"
	@echo "  eval        per-class + binary-collapse metrics + holdouts"
	@echo "  figures     confusion / ROC / PCA / comparison plots"
	@echo "  build-caver-t12  build persisted-output-first CAVER Tier 1-2 parquet"
	@echo "  build-lipid-boundary  build v_lipid_boundary feature parquet"
	@echo "  hierarchical-lipid  run staged lipid hierarchy over v_sterol artifacts"
	@echo "  all         ingest -> train -> eval -> figures"
	@echo "  scratch     Day 7+: download PDBs, run fpocket, re-ingest from raw"
	@echo "  test        pytest + ruff + mypy"
	@echo "  clean       remove processed/, models/, reports/*.png"

install:
	uv sync --extra dev

ingest:
	$(PY) python -m slipp_plus.cli ingest --config $(CFG)

train:
	$(PY) python -m slipp_plus.cli train --config $(CFG)

eval:
	$(PY) python -m slipp_plus.cli eval --config $(CFG)

figures:
	$(PY) python -m slipp_plus.cli figures --config $(CFG)

build-caver-t12:
	@echo "Use: make build-caver-t12 BASE=... MANIFEST=... OUT=... [ANALYSIS_ROOT=...] [HOLDOUT=1]"
	$(PY) python -m slipp_plus.cli build-caver-t12 --base-parquet $(BASE) --manifest $(MANIFEST) --output $(OUT) $(if $(ANALYSIS_ROOT),--analysis-root $(ANALYSIS_ROOT),) $(if $(HOLDOUT),--holdout,)

build-lipid-boundary:
	$(PY) python -m slipp_plus.cli build-lipid-boundary \
		--base-parquet $(or $(BASE),processed/v_sterol/full_pockets.parquet) \
		--source-pdbs-root $(or $(SOURCE_PDBS_ROOT),data/structures/source_pdbs) \
		--structural-join-parquet $(or $(STRUCTURAL_JOIN),processed/v49/full_pockets.parquet) \
		--output $(or $(OUT),processed/v_lipid_boundary/full_pockets.parquet) \
		--reports-dir $(or $(REPORTS_DIR),reports/v_lipid_boundary) \
		--workers $(or $(WORKERS),6)

hierarchical-lipid:
	$(PY) python -m slipp_plus.cli hierarchical-lipid \
		--full-pockets $(or $(FULL),processed/v_sterol/full_pockets.parquet) \
		--predictions $(or $(PREDICTIONS),processed/v_sterol/predictions/test_predictions.parquet) \
		--splits-dir $(or $(SPLITS_DIR),processed/v_sterol/splits) \
		--model-bundle $(or $(MODEL_BUNDLE),models/v_sterol/xgb_multiclass.joblib) \
		--output-report $(or $(REPORT),reports/v_sterol/hierarchical_lipid_report.md) \
		--output-metrics $(or $(METRICS),reports/v_sterol/hierarchical_lipid_metrics.parquet) \
		--output-predictions $(or $(OUT),processed/v_sterol/predictions/hierarchical_lipid_predictions.parquet) \
		--stage1-source $(or $(STAGE1_SOURCE),ensemble) \
		--ste-threshold $(or $(STE_THRESHOLD),0.40) \
		--workers $(or $(WORKERS),8)

all: ingest train eval figures

scratch:
	@echo "Day 7+ from-scratch reproduction."
	@echo "Requires: uv sync --extra scratch, system fpocket >= 4.0."
	$(PY) python -m slipp_plus.cli scratch --config $(CFG)

test:
	$(PY) pytest -q
	$(PY) ruff check .
	$(PY) mypy src

lint:
	$(PY) ruff check .
	$(PY) ruff format --check .

typecheck:
	$(PY) mypy src

clean:
	rm -rf processed/ models/ reports/raw_metrics.parquet reports/*.png reports/*.pdf
	find . -type d -name __pycache__ -exec rm -rf {} +
