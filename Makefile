.PHONY: help install install-dev quickstart smoke test compile validate check lint shell package format clean

UV ?= uv
UV_CACHE_DIR ?= .uv-cache
PYTHON ?= $(UV) run python
UV_RUN_DEV ?= $(UV) run --extra dev
export UV_CACHE_DIR

help:
	@echo "MARQUIS developer commands"
	@echo "  make install      Sync editable package with uv"
	@echo "  make install-dev  Sync editable package with uv and dev extras"
	@echo "  make quickstart   Run no-model command smoke checks"
	@echo "  make smoke        Alias for quickstart"
	@echo "  make test         Run unit and smoke tests"
	@echo "  make compile      Compile package sources"
	@echo "  make validate     Validate sample contracts and fixtures"
	@echo "  make check        Run release checks"
	@echo "  make lint         Run Ruff checks"
	@echo "  make shell        Check shell and SLURM template syntax"
	@echo "  make package      Build source and wheel distributions"
	@echo "  make format       Format Python files with Ruff"
	@echo "  make clean        Remove local Python caches"

install:
	$(UV) sync

install-dev:
	$(UV) sync --extra dev

quickstart:
	$(PYTHON) -m marquis.retrieval.cli --help >/dev/null
	$(PYTHON) -m marquis.information_extraction.cli --help >/dev/null
	$(PYTHON) -m marquis.article_generation.cli --help >/dev/null
	$(PYTHON) -m marquis.rlm_controller.cli --help >/dev/null
	$(PYTHON) -m marquis.evaluation.cli --help >/dev/null
	$(PYTHON) -m marquis.common.validate_contracts

smoke: quickstart

test:
	$(UV_RUN_DEV) python -m pytest -q

compile:
	$(PYTHON) -m compileall -q src

validate:
	$(PYTHON) -m marquis.common.validate_contracts

check: lint shell test compile validate
	git diff --check

lint:
	$(UV_RUN_DEV) ruff check src tests

shell:
	bash -n scripts/*.sh slurm/templates/*.sh slurm/templates/*.sbatch

package:
	rm -rf dist build src/marquis.egg-info
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run --with build python -m build --sdist --wheel

format:
	$(UV_RUN_DEV) ruff format src tests

clean:
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
