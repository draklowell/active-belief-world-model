PYTHON ?= $(shell command -v python3 || command -v python)
VENV := .venv
BIN := $(VENV)/bin
VENV_PYTHON := $(BIN)/python
PIP := $(VENV_PYTHON) -m pip

.PHONY: setup reproduce

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	$(PIP) install -e .
	$(BIN)/pre-commit install

reproduce: setup
	$(BIN)/python scripts/download_data.py
	$(BIN)/python -m wine_origin train
	$(BIN)/python -m wine_origin evaluate
