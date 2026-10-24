.PHONY: setup

setup:
	uv python install 3.14
	uv sync
	uv run pre-commit install

mlflow-headless:
	uv run mlflow server \
		--backend-store-uri sqlite:///.artifacts/mlflow.db \
		--default-artifact-root ./.artifacts/mlruns \
		--host 127.0.0.1 \
		--port 5000 \
		--workers 1 \
        --gunicorn-opts "--workers=1 --threads=1 --worker-class=sync --no-sendfile"

mlflow:
	uv run mlflow server \
		--backend-store-uri sqlite:///.artifacts/mlflow.db \
		--default-artifact-root ./.artifacts/mlruns \
		--host 127.0.0.1 \
		--port 5000 \
		--workers 1
