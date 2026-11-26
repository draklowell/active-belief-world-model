.PHONY: setup

ifneq (,$(wildcard ./.env))
    include .env
    export
endif

setup:
	uv python install 3.14
	uv sync
	uv run pre-commit install
