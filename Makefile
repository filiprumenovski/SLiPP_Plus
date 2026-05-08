.PHONY: help install ingest train eval figures all scratch build-caver-t12 build-caver-batch build-lipid-boundary build-v-sterol-ablation hierarchical-lipid test test-slow lint typecheck clean

PY ?= uv run
CFG ?= configs/day1.yaml
HIER_CFG ?= configs/v_sterol_boundary_refactor.yaml
MYPY_TARGETS ?= src/slipp_plus/cli.py src/slipp_plus/splits.py src/slipp_plus/schemas.py src/slipp_plus/run_metadata.py

help:
	@echo "slipp_plus targets:"
	@echo "  install     uv sync (installs deps + dev extras)"
	@echo "  ingest      CSV + xlsx -> validated parquets (Rule 1 gate)"
	@echo "  train       25 iterations x 3 models (RF, XGB, LGBM)"
	@echo "  eval        per-class + binary-collapse metrics + holdouts"
	@echo "  figures     confusion / ROC / PCA / comparison plots"
	@echo "  build-caver-t12  build persisted-output-first CAVER Tier 1-2 parquet"
	@echo "  build-caver-batch  run v_tunnel CAVER on one structure batch (BATCH=0 BATCH_SIZE=100)"
	@echo "  build-lipid-boundary  build v_lipid_boundary feature parquet"
	@echo "  build-v-sterol-ablation  materialize a v_sterol ablation feature set"
	@echo "  hierarchical-lipid  ingest + train + eval the config-driven boundary-head hierarchy"
	@echo "  all         ingest -> train -> eval -> figures"
	@echo "  scratch     Day 7+: download PDBs, run fpocket, re-ingest from raw"
	@echo "  test        pytest + ruff + mypy"
	@echo "  test-slow   pytest slow Day 1 regression checks"
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

build-caver-batch:
	$(PY) python -m slipp_plus.tunnel_features training \
		--base-parquet $(or $(BASE),processed/v_sterol/full_pockets.parquet) \
		--source-pdbs-root $(or $(SOURCE_PDBS_ROOT),data/structures/source_pdbs) \
		--caver-jar $(or $(CAVER_JAR),tools/caver/caver.jar) \
		--output $(or $(OUT),processed/v_tunnel/batches/batch_$(or $(BATCH),0).parquet) \
		--reports-dir $(or $(REPORTS_DIR),reports/v_tunnel/batches/batch_$(or $(BATCH),0)) \
		--workers $(or $(WORKERS),6) \
		--cache-dir $(or $(CACHE_DIR),processed/v_tunnel/structure_json) \
		--batch-index $(or $(BATCH),0) \
		--batch-size $(or $(BATCH_SIZE),100) \
		--max-missing-structure-frac $(or $(MAX_MISSING_STRUCTURE_FRAC),0.02) \
		--min-context-present-frac $(or $(MIN_CONTEXT_PRESENT_FRAC),0.0) \
		--min-profile-present-frac $(or $(MIN_PROFILE_PRESENT_FRAC),0.0)

build-lipid-boundary:
	$(PY) python -m slipp_plus.cli build-lipid-boundary \
		--base-parquet $(or $(BASE),processed/v_sterol/full_pockets.parquet) \
		--source-pdbs-root $(or $(SOURCE_PDBS_ROOT),data/structures/source_pdbs) \
		--structural-join-parquet $(or $(STRUCTURAL_JOIN),processed/v49/full_pockets.parquet) \
		--output $(or $(OUT),processed/v_lipid_boundary/full_pockets.parquet) \
		--reports-dir $(or $(REPORTS_DIR),reports/v_lipid_boundary) \
		--workers $(or $(WORKERS),6)

build-v-sterol-ablation:
	@echo "Use: make build-v-sterol-ablation FEATURE_SET=... OUT=... [V_STEROL_DIR=...] [TRAINING_CSV=...]"
	$(PY) python -m slipp_plus.cli ablate-v-sterol \
		--feature-set $(FEATURE_SET) \
		--v-sterol-dir $(or $(V_STEROL_DIR),processed/v_sterol) \
		--output-dir $(OUT) \
		$(if $(TRAINING_CSV),--training-csv $(TRAINING_CSV),)

hierarchical-lipid:
	$(PY) python -m slipp_plus.cli ingest --config $(HIER_CFG)
	$(PY) python -m slipp_plus.cli train --config $(HIER_CFG)
	$(PY) python -m slipp_plus.cli eval --config $(HIER_CFG)

all: ingest train eval figures

scratch:
	@echo "Day 7+ from-scratch reproduction."
	@echo "Requires: uv sync --extra scratch, system fpocket >= 4.0."
	$(PY) python -m slipp_plus.cli scratch --config $(CFG)

test:
	$(PY) pytest -q
	$(PY) ruff check .
	$(PY) mypy $(MYPY_TARGETS)

test-slow:
	$(PY) pytest -q --runslow

lint:
	$(PY) ruff check .
	$(PY) ruff format --check .

typecheck:
	$(PY) mypy $(MYPY_TARGETS)

clean:
	rm -rf processed/ models/ reports/raw_metrics.parquet reports/*.png reports/*.pdf
	find . -type d -name __pycache__ -exec rm -rf {} +
