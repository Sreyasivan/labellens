# LabelLens

[![ci](https://github.com/Sreyasivan/labellens/actions/workflows/ci.yml/badge.svg)](https://github.com/Sreyasivan/labellens/actions/workflows/ci.yml)

Ask a question about an FDA drug label and get an answer that cites the exact label section it came
from. Pull a label's contraindications, interactions or warnings into a typed list. Measure both
against the real labels, every time the code changes. Use it from the command line, an HTTP API with
generated docs, or a small web page.

Runs entirely on a laptop with a free local model and no accounts. Switch to OpenAI or Claude by
changing environment variables; nothing else moves.

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
cite them by number. If the question does not name a drug in the index, or nothing in the index is
close to it, it refuses *before* the model is called.

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

**Typed extraction.** Give it a label and a section (contraindications, drug interactions, warnings)
and get the items back as a validated list (`pydantic`), ready for a database column instead of a
paragraph of prose. Each item is scored for whether its words actually occur in the section.

**An evaluation harness that runs against the real data.** Retrieval recall per label section, the
full answer path (right drug cited, required terms present, citations valid, off-topic questions
refused), extraction grounding and coverage, latency and token counts. The question set is itself
checked against the label text before any of it runs. The numbers below come from
`eval/results/*.json`; none of them are typed by hand.

## Results

Index: 2,000 prescription labels, 56,686 section chunks. Model: `llama3.2` (3B) via Ollama on a laptop
CPU. Question set: 100 questions in `eval/questions.jsonl`, 84 in scope across 9 label sections and
16 off-topic (8 general, 8 that use clinical vocabulary), every in-scope one verified against the label
text by `make check-questions`.

**Retrieval** (recall@5; an off-topic question counts as a hit only when it would be refused):

| retrieval configuration | recall@5 |
|---|---|
| dense vectors, naive top-k | 70/100 |
| dense vectors + one chunk per (drug, section) | 81/100 |
| hybrid dense + full-text, fused with RRF, + dedupe | 82/100 |
| **hybrid + dedupe + the named drug's chunks first** | **97/100** |

Per section with the final configuration: boxed warning 35/37, contraindications 17/18, adverse
reactions 10/10, drug interactions 8/8, dosage 3/3, overdosage 3/3, specific populations 2/2, warnings
1/1, off-topic refused 16/16. All three misses are "right drug, wrong section": the question names the
drug, its chunks are retrieved, but another of its sections outranks the one that holds the answer.

Each step fixed a failure the eval exposed, and the larger set exposed new ones:

- Naive top-k returned the same passage from five manufacturers' copies of one label.
- Dense vectors ranked the sertraline *boxed warning* 24th because embeddings blur exact terms.
- Postgres' default full-text query ANDs every word, so "What does the boxed warning for lithium say?"
  matched nothing. Ranking by the number of distinct matching terms fixed it.
- Rivaroxaban's boxed warning says "XARELTO" and never "rivaroxaban", so neither leg found it from
  the generic name. The full-text column now carries the drug's generic and brand names.
- Bupropion has 18 labels in the index; one section of them filled a leg's whole candidate pool, so
  its boxed warning never reached the fusion step. Each leg is now de-duplicated before fusion.
- Generic passages ("the following reactions are discussed elsewhere") sit close to every question in
  embedding space and crowded out the drug the question named. If the question names an indexed
  drug, that drug's chunks now come first.
- The distance threshold alone let 6 of 8 clinical-vocabulary off-topic questions through ("What
  are the symptoms of type 2 diabetes?" is 0.25 from a liraglutide passage). Scope now also requires
  that the question names a drug in the index.

**Answers** (the full ask path on the same 100 questions):

| check | result |
|---|---|
| cited the right drug | 84/84 = 100% |
| answer contains the required terms | 74/84 = 88% |
| every `[n]` citation points at a source that was shown | 72/84 = 86% |
| all three at once | 64/84 = 76% |
| in-scope questions wrongly refused | 0 |
| off-topic questions refused before the model | 16/16 = 100% |
| latency p50 / p95 | 2.2 s / 3.7 s |
| tokens per answer, prompt / completion (mean) | 1,559 / 85 |

**Faithfulness** (a judge model reads each cited sentence next to the source it cites; `make eval-faith`):

| | result |
|---|---|
| sentences in the 84 answers | 181 |
| sentences carrying a `[n]` citation | 99 |
| cited sentences the judge finds supported by their source | 77/99 = 78% |
| sentences with no citation at all | 82/181 = 45% |
| judge agreement with 29 hand-labelled pairs | 23/29 = 79% |

The judge is `llama3.2` too, and it is only trusted as far as that last row. Its first prompt agreed
with the hand labels 13 of 29 times, calling verbatim quotes "adds a fact"; asking it to quote the
supporting span first, and discarding any verdict whose quote is not actually in the source, brought it
to 23 of 29. The remaining disagreements are the judge accepting a sentence whose facts are true of
the drug but absent from the specific chunk cited. So the 78% is a floor with a known-imperfect ruler,
and the 45% uncited is the number that matters more: a 3B model, told to end every sentence with a
citation, does so for about half of them.

Retrieval and scope are no longer the bottleneck; the model is. Its failures are of three kinds: it
paraphrases past the required word ("suicidal thoughts and behaviours" becomes "mental health risks"),
it drops the `[n]` markers or cites the label's own section numbers, and it occasionally attaches a
true fact to the wrong source. The next step is not more prompt work; it is the hosted-model rows on
the roadmap, which the provider switch already supports.

**Extraction** (15 randomly sampled labels per section):

| section | grounding precision | sentence coverage | parse failures | mean latency |
|---|---|---|---|---|
| contraindications | 53/53 = 100% | 30/37 = 81% | 0/15 | 0.8 s |
| drug interactions | 100/100 = 100% | 51/83 = 61% | 0/15 | 2.6 s |
| warnings and cautions | 68/68 = 100% | 93/125 = 74% | 1/15 | 2.4 s |

Nothing invented in 221 items. Coverage is lower on the longer sections because the model summarises
tables and pharmacokinetic detail into one item per drug, and because headings and cross-references
count as sentences. The one parse failure is a label (cocaine hydrochloride) whose section the model
answered with truncated JSON; the output cap that ended it exists because an earlier run looped for
189 KB. Before the parser accepted `{"drug": ..., "effect": ...}` objects as items, 10 of the 30
interaction and warning extractions failed on shape alone. See [`eval/README.md`](eval/README.md) for
exactly what these metrics can and cannot prove.

**Caveat that matters:** 100 questions and 15 labels per section are enough to catch the failures
described above and to show that extraction does not invent content. They are not enough to quote as
general accuracy figures, the scope threshold was set on this same question set, and the judge was
validated on 29 pairs. The harness is built so the sets can grow without changing any code.

## How it works

```
openFDA label API ──► ingest: split by label section ──► bge-small embeddings ──► Postgres
                      (boxed warning, contraindications,      384-dim, local        + pgvector (HNSW)
                       adverse reactions, interactions, …)                          + tsvector (GIN)

question ──► dense top-50 (deduped) ─┐
         └─► full-text top-50 (deduped) ─┴─► reciprocal rank fusion ──► named drug first ──► top-5
                                                                                                 │
          too far, or no indexed drug named? ──► refuse, model never called ◄────────────────────┤
                                                                                                 ▼
                                                     numbered sources ──► LLM ──► answer with [n] citations
                                                                                  + fixed safety footer

label section ──► LLM in JSON mode ──► pydantic model ──► scored against the section text

CLI · HTTP API (FastAPI, /docs) · web page  ──► all call the same answer() and extract_items()
```

Three guardrails, in order:

1. **Scope check before the model.** Two conditions, both deterministic: the closest retrieved chunk
   is within a cosine-distance threshold, and the question names a drug that is in the index. Fail
   either and the system says so without calling the model. Cheaper and more predictable than asking
   the model to decline.
2. **Closed-book prompt.** The model sees only the numbered sources and must cite every claim.
   It is told to quote the label rather than recommend anything for a specific person.
3. **Fixed footer.** Every answer ends with the same not-medical-advice sentence, added in code, not by
   the model.

## Quick start

Requirements: Docker, Python 3.11+, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com/).
About 15 minutes the first time, most of it the model download and the embedding run.

```bash
git clone https://github.com/Sreyasivan/labellens && cd labellens
cp .env.example .env                 # defaults need no accounts
ollama pull llama3.2                 # 2 GB, once

make setup                           # install dependencies into .venv
make db                              # Postgres + pgvector in Docker, port 5433
make schema                          # tables and indexes
make ingest                          # 2,000 labels from openFDA; cached in data/raw; embeds on CPU, ~10 min

make ask Q="Does warfarin have a boxed warning?"
make api                             # http://localhost:8000  (web page) and /docs (API)
```

### Check it yourself

Every number in this README comes from a command below. Run them and compare.

```bash
make test                            # 46 unit tests, no database or model needed, under 3 s
make check-questions                 # every eval question verified against the indexed label text
make eval                            # retrieval recall@5 for all four configurations
make eval-answer                     # full ask path: citations, required terms, refusals, latency
make eval-faith                      # judge each cited sentence against its source; validates the judge first
make eval-extract N=15 SECTION=contraindications     # also drug_interactions, warnings_and_cautions
```

What you should see: `make ask` prints an answer with `[n]` citations, a sources list with the drug,
section and cosine distance of each, and a token/latency line. An off-topic question
(`make ask Q="What is the capital of France?"`) is refused without calling the model. `make eval`
prints the table under Results, plus a per-section breakdown and the top-3 chunks for every miss so a
failure can be read, not guessed at. Results are also written to `eval/results/*.json`.

To try the API without the page:

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "What is the boxed warning for sertraline?"}' | python -m json.tool
curl -s localhost:8000/extract -H 'content-type: application/json' \
     -d '{"generic_name": "metformin", "section": "contraindications"}' | python -m json.tool
```

### Configuration

All settings are environment variables (see `.env.example`):

| variable | default | notes |
|---|---|---|
| `DATABASE_URL` | local Docker Postgres | any Postgres with the `vector` extension, including Supabase |
| `LLM_PROVIDER` | `openai_compatible` | or `anthropic` |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | any OpenAI-compatible endpoint: Ollama, OpenAI, vLLM, Groq |
| `LLM_API_KEY` | `ollama` | the endpoint's key |
| `LLM_MODEL` | `llama3.2` | `claude-opus-5-5` when the provider is `anthropic` |
| `ANTHROPIC_API_KEY` | unset | read by the Anthropic SDK when `LLM_PROVIDER=anthropic` |
| `RETRIEVAL` | `hybrid` | `vector` for dense only |
| `OPENFDA_API_KEY` | empty | optional, raises the openFDA rate limit |

Three ways to run the model, all behind the same `chat()` function:

| | set | needs |
|---|---|---|
| **Local (default)** | nothing | Ollama running; data never leaves the machine |
| **OpenAI or compatible** | `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | an API key |
| **Claude** | `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`, optionally `LLM_MODEL` | `uv sync --extra anthropic` |

Retrieval, extraction and every eval are identical across the three, which is what makes a
like-for-like quality and cost comparison possible: change the variables, run `make eval-answer`.

### Project layout

```
labellens/
  config.py        settings, indexed label sections, retrieval thresholds
  ingest.py        openFDA fetch (cached, rate-limit aware), section chunking, embedding, upsert
  embed.py         bge-small embeddings, loaded on first use
  rag.py           dense + full-text retrieval, RRF, dedupe, scope check, cited answer
  extract.py       JSON-mode extraction into a pydantic model, one rule per section
  llm.py           chat() over an OpenAI-compatible endpoint or the Anthropic SDK
  cli.py           `ask` command
  api.py           FastAPI: POST /ask, POST /extract, GET /health, generated docs at /docs
  static/index.html  the web page; plain HTML + fetch(), served by the API
eval/
  questions.jsonl        100 questions: 84 in scope, 16 off-topic for the refusal path
  check_questions.py     verifies every question against the indexed label text; CI runs it first
  run_eval.py            recall@5 per retrieval configuration; --min-recall is the CI gate
  run_answer_eval.py     end-to-end: right drug cited, required terms present, citations valid, refusals, latency
  run_faithfulness.py    judge model on each cited sentence vs its source; validated against judge_labels.jsonl
  judge_labels.jsonl     29 hand-labelled (sentence, source) pairs the judge is scored against
  run_extract_eval.py    extraction grounding and coverage on sampled labels, per section
  scoring.py             deterministic token-overlap scorer
sql/schema.sql           tables, HNSW and GIN indexes
tests/                   unit tests, no database or model required
.github/workflows/ci.yml unit tests, then the retrieval gate against a real pgvector service
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

**Lexical search matches any term and ranks by how many terms matched.** Postgres' default query
ANDs every word, so "What does the boxed warning for lithium say?" matched nothing because no label
says "say". A plain OR query over-ranked long chunks that repeat one common word. Ranking by the
number of distinct question terms a chunk contains, then by density, fixed both. The section name is
part of the indexed text so "boxed warning" finds boxed-warning chunks, whose text starts with
"WARNING:" and never contains the word "boxed".

**Scope is two signals, both deterministic.** The nearest chunk must be within 0.35 cosine distance
(measured: in-scope questions never exceed 0.32, general off-topic ones never come under 0.39), and
the question must name a drug that is in the index, matched against the generic and brand names of the
retrieved chunks with salt and dosage-form words ignored. The distance check alone let 7 of 8
general and 6 of 8 clinical-vocabulary off-topic questions through at different points; each was
caught the day those questions were added. The cost is a product rule: a question has to name its
drug. "Which drugs carry a suicidality boxed warning?" is refused, and the refusal says why.

**The named drug's chunks come first.** Once retrieval knows which drug the question is about, every
chunk of that drug outranks everything else, in fused order. This is what turned 82/100 into 97/100:
without it, generic passages that sit close to every question in embedding space filled the top-5.

**One chunk per (drug, section).** Generic drugs have many manufacturers and nearly identical labels.
Without dedupe the top-5 was the same paragraph five times, and the eval showed it immediately.

**Refuse before the model, not with the model.** A distance threshold is deterministic, free, and
testable. Asking a language model to decide whether it should answer is none of those.

**Local embeddings and a local model by default.** `bge-small` (384 dimensions) runs on CPU and
embeds 56k chunks in minutes. A 3B local model answers in under a second for extraction and a few
seconds for Q&A. Label text never has to leave the machine, which is a hard requirement in parts of
this industry, and it makes the project reproducible for anyone without an API key.

**No orchestration framework.** The whole pipeline is about 500 lines of plain Python and SQL. Every
step is visible, every prompt is in the repo, and there is nothing between the code and the behaviour
that would need to be debugged through someone else's abstraction.

**An API with a plain web page, not a notebook-style UI.** Streamlit and Gradio get to a demo fastest,
but they rerun a script per click, hold state awkwardly, and the result is a demo, not a service. Here
the UI is a client of a typed FastAPI service: the same functions serve the CLI, the evals, the API and
the page; the API documents itself at `/docs`; and it is tested in CI with FastAPI's test client. The
page is one HTML file with `fetch()`, no build step. Chainlit was ruled out as chat-only, Reflex and
NiceGUI as more framework than the two forms need.

**One provider switch, two client libraries.** Ollama, OpenAI, vLLM and Groq all speak the OpenAI
chat API, so one client covers them. Claude is called through the official Anthropic SDK instead of a
compatibility shim, so its own features work (server-side refusal fallback is on by default). Both
return the same `Completion`, so the rest of the code does not know which one ran.

**The eval drives the data.** Every drug named in `eval/questions.jsonl` is guaranteed to be ingested,
so a retrieval miss is always a retrieval problem, never a coverage problem.

## FAQ

**Why not just ask ChatGPT or Claude about the drug? They can search the web now.**
For a one-off question from a consumer, they are good enough, and this project does not try to beat
them there. The gap is everything a team needs around the answer:

- *Pinned source.* A web search returns whichever page ranks that day: DailyMed, a manufacturer PDF,
  Drugs.com, a forum. LabelLens answers from one known set of FDA-approved labels, and says which
  label and which section. In a regulated setting "a web page" is not a citable source; the current
  Prescribing Information is.
- *Reproducible and testable.* Ask a chatbot the same question next week and the sources, the answer
  and the wording change. Here the same 100 questions run on every commit and produce the same recall
  number, so a change that breaks retrieval fails CI instead of being noticed by a user.
- *Data stays in-house.* The question often carries context (a patient's other medicines, a renal
  value). Many pharma and hospital IT policies do not allow that text to leave the organisation.
  The default setup runs the model on the machine that holds the labels.
- *Batch, not chat.* "Extract the contraindications of all 2,000 labels into one table" is a job, not a
  conversation. The extraction path here is a function with a schema and a grounding score.
- *Cost and control.* A 3B model on a CPU answers in about a second for free. Hosted models are a
  three-variable switch when their quality is worth paying for; the eval shows whether it is.

The honest summary: a chatbot with web search is a better general assistant. LabelLens is a narrow,
measured, auditable component that a team could put behind a form or a pipeline.

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
Add a line to `eval/questions.jsonl` with the drug's generic name, the section that answers it and the
words a correct answer must contain. Run `uv run python -m labellens.ingest --only-eval` (fetches any
eval drug not yet indexed), `make check-questions` (proves the label actually says what the question
assumes) and `make eval`. A question that fails the check is wrong about the label and is not an eval
case until fixed.

**Why FastAPI and a plain HTML page rather than Streamlit or Gradio?**
See Design decisions. Short version: the UI should be a client of the same API a pipeline would call,
and both should be testable without a browser. Streamlit or Gradio would have been a one-afternoon
front end on an untested service.

**Why does CI index only the eval drugs?**
Embedding 2,000 labels on a CI runner would take long enough that nobody would wait for it. The gate
indexes the ~60 drugs the questions name (about 2,900 chunks, a few minutes), verifies every question
against the label text, and fails if recall@5 of the final configuration drops below 90%. The headline
numbers come from the full index and are reproduced with `make eval`.

## Limitations

- US labels only. The EU equivalent (SmPC) has a different structure and is not indexed.
- openFDA mirrors labels with some delay; a very recent label change may not be reflected.
- Scope requires the question to name an indexed drug. General questions across drugs are refused,
  and a question that names a drug for an unrelated reason ("how much caffeine is in coffee" names
  caffeine, which is in butalbital combinations) is answered from that label.
- The three retrieval misses are all "right drug, wrong section": nothing yet maps "boxed warning" in
  a question to the boxed-warning section when another section of the same drug scores higher.
- The faithfulness judge is a 3B model validated on 29 hand-labelled pairs at 79% agreement. Its
  number is indicative, not authoritative, until a stronger judge or more labels replace it.
- `must_mention` checks that the answer contains the right words; it does not prove the sentence
  around them is correct. That is what the judged faithfulness number is for, with the caveat above.
- Extraction covers three sections. Others follow the same one-line pattern.
- 100 questions and 15 labels per section are enough to catch the failures described here, not enough
  to quote as general accuracy figures. See the caveat under Results.

## Roadmap

In order of what the evaluation says matters most:

1. **Hosted-model rows.** Same 100 questions on Claude and on an OpenAI model, with cost per query,
   next to the local 3B row: answer checks, uncited-sentence rate and judged faithfulness. The switch
   exists; the table does not yet. A stronger judge comes from the same switch.
2. **Section intent.** When a question says "boxed warning" or "contraindications", prefer that
   section of the named drug. This is the only kind of retrieval miss left.
3. **Drug and section in the embedded text.** Prepending "rivaroxaban (Xarelto), boxed warning" to
   each chunk before embedding would give the dense leg what the full-text leg now has. Costs a
   re-embedding run; measured before it is kept.
4. **More judge labels.** 29 pairs validate the judge; 100 would let the faithfulness number carry a
   confidence interval.
5. **Extraction for dosage and renal adjustment**, where the output is numeric and can be checked
   exactly.
6. **A deployed instance** behind the existing API, once a hosted embedding store is chosen.

Out of scope by design: agents and multi-step tool use (one question, one retrieval, one answer covers the
job), fine-tuning (see FAQ), and a separate vector database (see Design decisions).

## Licence

MIT. See [LICENSE](LICENSE).
