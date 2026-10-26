.PHONY: setup db schema ingest ask eval eval-answer eval-extract check-questions test api

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

check-questions:  ## verify every eval question against the indexed label text
	uv run python eval/check_questions.py

eval:             ## retrieval eval over eval/questions.jsonl, all configs
	uv run python eval/run_eval.py

eval-answer:      ## end-to-end answer eval: citations, required terms, refusals, latency
	uv run python eval/run_answer_eval.py

test:
	uv run pytest -q

eval-faith:       ## judged faithfulness of cited sentences (uses eval/results/answers.json), plus judge validation
	uv run python eval/run_faithfulness.py

eval-extract:     ## extraction eval on N real labels: make eval-extract N=15 SECTION=contraindications
	uv run python eval/run_extract_eval.py --n $(or $(N),15) --section $(or $(SECTION),contraindications)

api:              ## HTTP API + web page on http://localhost:8000 (docs at /docs)
	uv sync --extra api && uv run uvicorn labellens.api:app --reload
