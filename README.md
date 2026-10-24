# LabelLens

Ask a question about any FDA drug label and get a cited answer. Extract its adverse reactions,
contraindications and dosage into a typed schema, scored against openFDA's own structured fields.

<!-- week 4: 20-second GIF of ask → cited answer → extract → eval table -->

## Architecture

```
openFDA label API ──► ingest (chunk by section, bge-small embeddings) ──► Postgres + pgvector
                                                                              │
question ──► retrieve top-k ──► scope guardrail ──► LLM with numbered sources ──► cited answer
                                                                              │
label ──────────────────────────────────────► extract (pydantic schema) ◄── compared to openFDA fields
                                                                              │
eval/questions.jsonl ──► make eval ──► recall@k, faithfulness, extraction F1, latency, cost
```

## Results

<!-- pasted from eval/results/latest.json after `make eval`. Never typed by hand. -->

| config | recall@5 | faithfulness | extraction F1 | p50 latency | cost / query |
|---|---|---|---|---|---|
| vector only, local model | _tbd_ | _tbd_ | _tbd_ | _tbd_ | $0 |
| hybrid (BM25 + vector), local model | _tbd_ | _tbd_ | _tbd_ | _tbd_ | $0 |
| hybrid, hosted model | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

Week-1 retrieval baseline on the 8 seed questions (2,000 labels, 56,686 chunks): naive top-5 **6/8**,
one chunk per label **6/8**, one chunk per (drug, section) **7/8**. The remaining miss is a lexical
one ("boxed warning" appears in the question, not in the nearest chunk), which is what hybrid search
is for. Sample question with llama3.2 locally: 682 prompt + 101 completion tokens, 6.9 s.

## Run it

```bash
cp .env.example .env         # defaults work with Docker + Ollama, no accounts needed
make setup                   # uv sync
make db                      # pgvector in Docker (port 5433)
make schema
make ingest                  # ~2,000 prescription labels, cached in data/raw
make ask Q="What are the contraindications of metformin?"
make eval
make test
```

Local LLM: `ollama pull llama3.2`. Hosted LLM: set `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` in `.env`.

## Design decisions

- **Chunk by label section, not by fixed window.** The section name travels with every chunk, so retrieval can be scored per section and citations say *where* in the label the answer came from.
- **pgvector, not a separate vector database.** One Postgres holds the raw labels, the chunks and the vectors. Fewer moving parts and the extraction ground truth is a JSONB column away.
- **Refuse before calling the model.** If the nearest chunk is too far away the question is out of scope and the LLM is never called. Cheaper and safer than asking the model to decline.
- **The eval set drives ingestion.** Every drug named in `eval/questions.jsonl` is guaranteed to be indexed, so a miss is a retrieval problem, never a coverage problem.
- **OpenAI-compatible client only.** Switching between a free local model and a hosted one is three env vars, which is what makes the cost/quality comparison row possible.

## Deliberately not built

- **Fine-tuning.** Not until the eval shows retrieval is no longer the bottleneck.
- **Agents / multi-step tool use.** One question, one retrieval, one answer covers the use case.
- **Federated or zero-ETL query layer.** The data is 2,000 JSON documents; a copy is fine.
- **Multi-tenant auth.** This is a reference implementation, not a product.

## Limitations

- openFDA labels are US-only and can lag the manufacturer's current label.
- Answers are label summaries, not clinical guidance. The footer on every answer says so.

See [AI_USAGE.md](AI_USAGE.md) for how AI assistants were used in building this.
