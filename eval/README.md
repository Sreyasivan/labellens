# Evaluation

Three evals and one check, all run against the real index. The numbers in the top-level README are
copied from `eval/results/*.json`; nothing there is typed by hand.

## Check: `make check-questions`

Before any eval number is trusted, every in-scope question is verified against the indexed labels:
the drug is indexed, a label of that drug has the expected section, and that section contains every
`must_mention` term. A question that fails is wrong about the label, not a retrieval failure. CI runs
this first and stops if anything fails.

## Retrieval: `make eval`

`questions.jsonl`, one JSON object per line:

| field | meaning |
|---|---|
| `id` | stable identifier |
| `question` | what a pharmacist, nurse or patient would actually ask |
| `expected_generic_name` | the drug the answer must come from. Ingestion guarantees one label for every drug listed here. `null` marks an out-of-scope question, which passes only if the system refuses |
| `expected_section` | the label section that holds the answer (see `SECTIONS` in `labellens/config.py`). A list when old- and new-format labels place the fact in different sections |
| `must_mention` | lower-case terms a correct answer has to contain, checked by the answer eval |

Metric: recall@k. A question is a hit when any of the top-k chunks comes from the expected drug *and*
the expected section. The script runs every retrieval configuration (dense only, dense + dedupe,
hybrid) on the same questions so they can be compared directly.

Out-of-scope questions have `expected_generic_name: null` and count as a hit only when the scope check
would refuse them. `--min-recall 0.9` makes the script exit 1 below that hybrid recall, which is how CI
uses it.

## Answers: `make eval-answer`

Runs the complete ask path (retrieve, scope check, model, footer) on every question and checks the
answer itself:

| check | passes when |
|---|---|
| right drug cited | at least one cited source is a chunk of the expected drug |
| mentions required terms | the answer text contains every `must_mention` term |
| citations valid | every `[n]` in the answer points at a source that was actually shown to the model |
| refused (off-topic only) | the scope check stopped the question before the model was called |

"Answered correctly" means all checks passed. Latency p50/p95 and mean token counts are reported so a
model or retrieval change shows its cost. What this cannot measure: whether the sentence around a
required term is faithful to the source. That needs a judged metric (see the roadmap).

## Extraction: `make eval-extract SECTION=...`

Samples N labels that have the chosen section (contraindications, drug_interactions or
warnings_and_cautions), extracts a list of items, and scores the list against the section text
(`scoring.py`, token overlap, no LLM judge):

| metric | what it measures |
|---|---|
| grounding precision | items whose content words all occur in the section ÷ items. 100% means nothing was invented |
| sentence coverage | sentences of the section represented by at least one item ÷ sentences. 100% means nothing was left out |

What it cannot measure: whether items are segmented correctly or clinically complete. openFDA
publishes the section as prose, not as a gold list, so there is no gold list to compare against.

## Adding questions

Write the question first, then open the actual label on
[DailyMed](https://dailymed.nlm.nih.gov/) and confirm which section answers it before filling in
`expected_section`. A question whose answer you have not checked against the label is not an eval
case, it is a guess.
