.PHONY: help install ingest train eval figures all scratch test lint typecheck clean

PY ?= uv run
CFG ?= configs/day1.yaml

help:
	@echo "slipp_plus targets:"
	@echo "  install     uv sync (installs deps + dev extras)"
	@echo "  ingest      CSV + xlsx -> validated parquets (Rule 1 gate)"
	@echo "  train       25 iterations x 3 models (RF, XGB, LGBM)"
	@echo "  eval        per-class + binary-collapse metrics + holdouts"
	@echo "  figures     confusion / ROC / PCA / comparison plots"
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
