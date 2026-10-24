# LabelLens

[![ci](https://github.com/Sreyasivan/labellens/actions/workflows/ci.yml/badge.svg)](https://github.com/Sreyasivan/labellens/actions/workflows/ci.yml)

Ask a question about an FDA drug label and get an answer that cites the exact label section it came
from. Pull a label's contraindications into a typed list. Measure both against the real labels,
every time the code changes.

Runs entirely on a laptop with a free local model. Switch to a hosted model by changing three
environment variables.

---

## The problem

Every prescription drug sold in the US has an FDA-approved label (the *Prescribing Information*,
published as Structured Product Labeling and mirrored by [openFDA](https://open.fda.gov/)). It is the
single source of truth for what a drug is for, who must not take it, what it interacts with and what
it does to people. It is also 20,000 to 60,000 words long, written for lawyers as much as clinicians,
and a typical generic drug has a dozen near-identical versions from different manufacturers.

People who answer questions from these labels all day (hospital pharmacists, medical-information
teams at pharmaceutical companies, nurses on a ward, poison-control staff) have two hard rules:

1. **The answer has to be on-label.** Medical-information responses must be consistent with the
   approved labeling; anything else is a regulatory problem, not just a quality problem.
2. **The answer has to be traceable.** "The label says X" is useless without *where* the label says X.

A general-purpose chatbot breaks both rules. It answers from memory, it does not cite, and when it is
wrong it is confidently wrong. In a regulated setting that is worse than no answer. Many of these
organisations also cannot send label text, or the questions asked about it, to a third-party API at all.

LabelLens is a small, measurable system built for those rules.

## What it does

**Cited question answering.** Retrieval over 2,000 prescription labels from openFDA, chunked by label
section, with hybrid dense + full-text search. The model is only shown the retrieved sections and must
cite them by number. If nothing in the index is close to the question, it refuses *before* the model
is called.

```
$ make ask Q="What are the contraindications of metformin?"

According to the FDA label, the contraindications for metformin are:

* Severe renal impairment (eGFR below 30 mL/min/1.73 m²) [see Warnings and Precautions (5.1)].
* Hypersensitivity to metformin.
* Acute or chronic metabolic acidosis, including diabetic ketoacidosis, with or without coma.

These contraindications are listed in sections [1], [4], and [5].

This is a summary of FDA label text, not medical advice. Talk to a pharmacist or doctor before acting on it.

Sources:
  [1] Metformin Hydrochloride / contraindications (distance 0.159)
  [2] methimazole / contraindications (distance 0.165)
  [3] Acyclovir / contraindications (distance 0.165)
  [4] Metformin / contraindications (distance 0.171)
  [5] Glipizide and Metformin Hydrochloride / contraindications (distance 0.188)

llama3.2: 682+101 tokens, 6901 ms
```

Output above is verbatim. Two of the five retrieved sources were other drugs' contraindications
(dense retrieval matched "this is a contraindications section" more than the drug name); the model
used only the three metformin sources and said which ones. Hybrid retrieval, added after this run,
pushes the drug name up the ranking.

**Typed extraction.** Give it a label and get the contraindications back as a validated list
(`pydantic`), ready for a database column instead of a paragraph of prose.

**An evaluation harness that runs against the real data.** Retrieval recall per label section,
extraction grounding and coverage, latency and token counts. The numbers below come from
`eval/results/*.json`; none of them are typed by hand.

## Results

Index: 2,000 prescription labels, 56,686 section chunks. Model: `llama3.2` (3B) via Ollama on a laptop CPU.

**Retrieval** (recall@5 on the question set in `eval/questions.jsonl`):

| retrieval configuration | recall@5 |
|---|---|
| dense vectors, naive top-k | 6/8 |
| dense vectors + one chunk per (drug, section) | 7/8 |
| **hybrid dense + full-text, fused with RRF, + dedupe** | **8/8** |

Each step fixed a specific failure. Naive top-k returned the same passage from five manufacturers'
copies of the same label. Dense vectors alone ranked the sertraline *boxed warning* 24th for the
question "What is the boxed warning for sertraline?", because embeddings blur exact terms that
full-text search catches instantly.

**Extraction** (contraindications, 15 randomly sampled labels):

| metric | result | meaning |
|---|---|---|
| grounding precision | 53/53 = 100% | no extracted item contained words that are not in the label section |
| sentence coverage | 30/37 = 81% | share of section sentences represented in the extraction |
| parse failures | 0/15 | every response was valid JSON matching the schema |
| mean latency | 652 ms | per label, local 3B model |

The 7 uncovered sentences are section headings, cross-references such as *(see WARNINGS)* and
explanatory sentences, not missed contraindications. See [`eval/README.md`](eval/README.md) for
exactly what these metrics can and cannot prove.

**Caveat that matters:** 8 questions and 15 labels are small samples. They are large enough to catch
the retrieval failures above and to show the extraction does not invent content; they are not large
enough to quote as general accuracy figures. The harness is built so the sets can grow without
changing any code.

## How it works

```
openFDA label API ──► ingest: split by label section ──► bge-small embeddings ──► Postgres
                      (boxed warning, contraindications,      384-dim, local        + pgvector (HNSW)
                       adverse reactions, interactions, …)                          + tsvector (GIN)

question ──► dense top-50 ─┐
         └─► full-text top-50 ─┴─► reciprocal rank fusion ──► one chunk per (drug, section) ──► top-5
                                                                                                 │
                              closest chunk too far? ──► refuse, model never called ◄────────────┤
                                                                                                 ▼
                                                     numbered sources ──► LLM ──► answer with [n] citations
                                                                                  + fixed safety footer

label section ──► LLM in JSON mode ──► pydantic model ──► scored against the section text
```

Three guardrails, in order:

1. **Scope check before the model.** If the closest retrieved chunk is further than a cosine-distance
   threshold, the question is out of scope and the system says so. Cheaper and more predictable than
   asking the model to decline.
2. **Closed-book prompt.** The model sees only the numbered sources and must cite every claim.
   It is told to quote the label rather than recommend anything for a specific person.
3. **Fixed footer.** Every answer ends with the same not-medical-advice sentence, added in code, not by
   the model.

## Quick start

Requirements: Docker, Python 3.11+, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com/).

```bash
git clone https://github.com/Sreyasivan/labellens && cd labellens
cp .env.example .env                 # defaults need no accounts
ollama pull llama3.2

make setup                           # install dependencies
make db                              # Postgres + pgvector in Docker, port 5433
make schema
make ingest                          # 2,000 labels from openFDA; cached in data/raw; embeds on CPU, ~10 min first run

make ask Q="Does warfarin have a boxed warning?"
make eval                            # retrieval recall, all configurations
make eval-extract N=15               # extraction grounding and coverage
make test
```

### Configuration

All settings are environment variables (see `.env.example`):

| variable | default | notes |
|---|---|---|
| `DATABASE_URL` | local Docker Postgres | any Postgres with the `vector` extension, including Supabase |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | any OpenAI-compatible endpoint |
| `LLM_API_KEY` | `ollama` | |
| `LLM_MODEL` | `llama3.2` | |
| `RETRIEVAL` | `hybrid` | `vector` for dense only |
| `OPENFDA_API_KEY` | empty | optional, raises the openFDA rate limit |

To use a hosted model, point the three `LLM_*` variables at it. Nothing else changes, which is what
makes a like-for-like cost and quality comparison possible.

### Project layout

```
labellens/
  config.py    settings, indexed label sections, retrieval thresholds
  ingest.py    openFDA fetch (cached), section chunking, embedding, upsert
  embed.py     bge-small embeddings, loaded on first use
  rag.py       dense + full-text retrieval, RRF, dedupe, scope check, cited answer
  extract.py   JSON-mode extraction into a pydantic model
  llm.py       one OpenAI-compatible client
  cli.py       `ask` command
eval/
  questions.jsonl        retrieval question set
  run_eval.py            recall@k per retrieval configuration
  run_extract_eval.py    extraction grounding and coverage on sampled labels
  scoring.py             deterministic token-overlap scorer
sql/schema.sql           tables, HNSW and GIN indexes
tests/                   unit tests, no database or model required
```

## Design decisions

**Chunk by label section, not by a fixed window.** FDA labels already have a standard structure
(boxed warning, 1 Indications, 2 Dosage, 4 Contraindications, 6 Adverse Reactions, 7 Interactions, …).
Keeping the section name on every chunk means citations say *where* in the label the answer lives,
retrieval can be scored per section, and the eval can tell "wrong drug" from "right drug, wrong section".

**Postgres for everything.** Raw label JSON, chunks, dense vectors (pgvector, HNSW index) and full-text
vectors (`tsvector`, GIN index) live in one database. One container, one connection string, joins
between them are plain SQL, and the extraction ground truth is a JSONB column away. A dedicated
vector database would add a service and a sync problem for 56k vectors that Postgres handles in
milliseconds.

**Hybrid retrieval with reciprocal rank fusion.** Dense vectors understand paraphrase; full-text search
understands exact terms. Drug questions are full of exact terms (drug names, "boxed warning", "eGFR",
"QT prolongation"). RRF combines the two ranked lists without having to calibrate their scores against
each other, which is the step that usually goes wrong in hybrid search.

**One chunk per (drug, section).** Generic drugs have many manufacturers and nearly identical labels.
Without dedupe the top-5 was the same paragraph five times, and the eval showed it immediately.

**Refuse before the model, not with the model.** A distance threshold is deterministic, free, and
testable. Asking a language model to decide whether it should answer is none of those.

**Local embeddings and a local model by default.** `bge-small` (384 dimensions) runs on CPU and
embeds 56k chunks in minutes. A 3B local model answers in under a second for extraction and a few
seconds for Q&A. Label text never has to leave the machine, which is a hard requirement in parts of
this industry, and it makes the project reproducible for anyone without an API key.

**No orchestration framework.** The whole pipeline is about 400 lines of plain Python and SQL. Every
step is visible, every prompt is in the repo, and there is nothing between the code and the behaviour
that would need to be debugged through someone else's abstraction.

**The eval drives the data.** Every drug named in `eval/questions.jsonl` is guaranteed to be ingested,
so a retrieval miss is always a retrieval problem, never a coverage problem.

## FAQ

**Why not just ask ChatGPT or Claude about the drug?**
They answer from training data, not from the current approved label, they do not cite, and they cannot
tell you *which manufacturer's label, which version, which section*. For a casual question that is
fine. For a pharmacist documenting a response, it is not an answer at all. LabelLens uses a model only
as a reader of retrieved text, never as a source of facts.

**Can I use the OpenAI or Anthropic API instead of a local model?**
Yes: set `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL`. Any OpenAI-compatible endpoint works. The
local default exists so the project runs with no account and so that label text stays on the machine.

**Why is the default model so small?**
Because the design does the heavy lifting, not the model. With the right five sections in front of it, a
3B model produced 53/53 grounded extraction items. A bigger model improves wording and recall at the
margins; it does not fix retrieval, and retrieval is where the real failures were.

**Why Postgres full-text search instead of a BM25 library?**
The chunks are already in Postgres. A generated `tsvector` column and a GIN index give ranked lexical
search with one `ALTER TABLE`, no second index to keep in sync, and the hybrid query is a SQL join.

**Why not LangChain or LlamaIndex?**
They are good at getting to a first demo. This project needed to know exactly what text reaches the
model, exactly how candidates are ranked, and exactly why a question missed, so that the eval could
point at a specific line. Plain code made that easier than configuring a framework to expose it.

**Why not fine-tune a model on the labels?**
Fine-tuning teaches style and format, not facts, and the facts change every time a label is updated.
Retrieval over the current label text is the right tool for "what does the label say". Fine-tuning
might help the extraction's segmentation later; the eval will show whether it is needed.

**Why is extraction scored against the section text and not against a gold list?**
Because openFDA publishes the contraindications section as prose. There is no gold list. What *can* be
measured honestly is whether every extracted item is grounded in the text (nothing invented) and
whether every sentence of the text is represented (nothing dropped). Those two numbers are reported;
a clinical-correctness number is not claimed.

**Is the data real?**
Yes. The labels are the FDA-approved Prescribing Information as published by openFDA. The questions in
the eval set were written by hand and checked against the label text.

**Is this medical advice?**
No. It summarises label text and says where it came from. Every answer carries that disclaimer, added
in code so the model cannot leave it out.

**How do I add a drug or a question?**
Add a line to `eval/questions.jsonl` with the drug's generic name and the section that answers it, then
run `make ingest` (it fetches any drug named in the eval set that is not yet indexed) and `make eval`.

## Limitations

- US labels only. The EU equivalent (SmPC) has a different structure and is not indexed.
- openFDA mirrors labels with some delay; a very recent label change may not be reflected.
- The scope check is a single distance threshold tuned on the current question set.
- Extraction covers one section (contraindications). The pattern extends to others; the scoring does too.
- Sample sizes in the results table are small. See the caveat under Results.

## Roadmap

In order of what the evaluation says matters most:

1. **Answer faithfulness.** Judge whether each cited claim is supported by its cited chunk, so the
   Q&A side gets a grounding number like the extraction side already has.
2. **Larger question set.** 100 hand-verified questions, including out-of-scope ones for the refusal path.
3. **Extraction for more sections.** Boxed warning present/absent, dosage adjustments by renal function,
   drug-interaction pairs.
4. **Eval in CI.** Run `make eval` against a pgvector service on every pull request and fail below the
   current numbers.
5. **Hosted-model comparison row.** Same eval, same questions, hosted model, with cost per query.
6. **A small web UI** over the existing functions, once the numbers above are in place.

Out of scope by design: agents and multi-step tool use (one question, one retrieval, one answer covers the
job), fine-tuning (see FAQ), and a separate vector database (see Design decisions).

## Licence

MIT. See [LICENSE](LICENSE).
