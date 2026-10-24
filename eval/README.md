# Eval set

`questions.jsonl`, one JSON object per line:

| field | meaning |
|---|---|
| `id` | `q001`, `q002`, … |
| `author` | `seed` (shipped examples), `sreya` (hand-written), `llm` (LLM-drafted, then hand-verified) |
| `question` | what a pharmacist or patient would actually ask |
| `expected_generic_name` | the drug the answer must come from; ingestion guarantees one label for every name listed here |
| `expected_section` | the label section that holds the answer (one of the sections in `labellens/config.py`) |
| `must_mention` | lower-case words a correct answer has to contain; used for the faithfulness check in week 3 |

Target: 100 questions. 60 written by hand, 40 drafted with an LLM and every one checked
against the actual label text before it is committed. Out-of-scope questions
(`"expected_generic_name": null`) are welcome: they test the refusal guardrail.

Run with `make eval`. Results land in `eval/results/latest.json` (git-ignored); the table in the
top-level README is pasted from that file.
