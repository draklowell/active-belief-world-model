.PHONY: setup

ifneq (,$(wildcard ./.env))
    include .env
    export
endif

setup:
	uv python install 3.14
	uv sync
	uv run pre-commit install

mlflow-headless:
	uv run mlflow server \
		--backend-store-uri sqlite:///$(MLFLOW_BASE_PATH)mlflow.db \
		--default-artifact-root $(MLFLOW_BASE_PATH)mlruns \
		--host 127.0.0.1 \
		--port 5000 \
		--workers 1 \
        --gunicorn-opts "--workers=1 --threads=1 --worker-class=sync --no-sendfile"

mlflow:
	uv run mlflow server \
		--backend-store-uri sqlite:///$(MLFLOW_BASE_PATH)mlflow.db \
		--default-artifact-root $(MLFLOW_BASE_PATH)mlruns \
		--host 127.0.0.1 \
		--port 5000 \
		--workers 1
