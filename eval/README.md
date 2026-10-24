# Evaluation

Two evals, both run against the real index. The numbers in the top-level README are copied from
`eval/results/*.json`; nothing there is typed by hand.

## Retrieval: `make eval`

`questions.jsonl`, one JSON object per line:

| field | meaning |
|---|---|
| `id` | stable identifier |
| `question` | what a pharmacist, nurse or patient would actually ask |
| `expected_generic_name` | the drug the answer must come from. Ingestion guarantees one label for every drug listed here. `null` marks an out-of-scope question, which passes only if the system refuses |
| `expected_section` | the label section that holds the answer (see `SECTIONS` in `labellens/config.py`) |
| `must_mention` | lower-case words a correct answer has to contain, for the answer-faithfulness check |

Metric: recall@k. A question is a hit when any of the top-k chunks comes from the expected drug *and*
the expected section. The script runs every retrieval configuration (dense only, dense + dedupe,
hybrid) on the same questions so they can be compared directly.

## Extraction: `make eval-extract`

Samples N labels with a contraindications section, extracts a list of contraindications, and scores
the list against the section text (`scoring.py`, token overlap, no LLM judge):

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
