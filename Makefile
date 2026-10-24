.PHONY: setup db schema ingest ask eval test

setup:            ## install dependencies into .venv
	uv sync --extra dev

db:               ## start local pgvector
	docker compose up -d

schema:           ## create tables and indexes
	uv run python -m labellens.db

ingest:           ## fetch ~2000 prescription labels from openFDA and embed them (cached in data/raw)
	uv run python -m labellens.ingest --limit 2000

ask:              ## make ask Q="What are the contraindications of metformin?"
	uv run python -m labellens.cli ask "$(Q)"

eval:             ## run retrieval eval over eval/questions.jsonl
	uv run python eval/run_eval.py

test:
	uv run pytest -q
