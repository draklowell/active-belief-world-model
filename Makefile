.PHONY: setup reproduce

setup:
	uv python install 3.14
	uv sync
	uv run pre-commit install

reproduce: setup
	uv run python scripts/download_data.py
	uv run python -m wine_origin train
	uv run python -m wine_origin evaluate
