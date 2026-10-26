"""Judged faithfulness: is each cited sentence of an answer supported by the source it cites?

Reads eval/results/answers.json (written by run_answer_eval.py, so no answers are regenerated), splits
every answered question into sentences, and for each sentence that carries [n] citations asks a judge
model whether the cited source text supports it. Sentences with no citation are counted separately,
never judged: an uncited claim is a failure of the citing rule, not of faithfulness.

The judge is a language model, so it is itself evaluated: eval/judge_labels.jsonl holds hand-labelled
(sentence, sources) pairs; --validate reports the judge's agreement with them. A faithfulness number
is only as good as that agreement, which is why both are printed together.

Usage: python eval/run_faithfulness.py [--validate-only]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labellens import config, llm, rag  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import overlap  # noqa: E402

ANSWERS = Path(__file__).parent / "results" / "answers.json"
LABELS = Path(__file__).parent / "judge_labels.jsonl"
RESULTS = Path(__file__).parent / "results" / "faithfulness.json"

JUDGE_PROMPT = """You compare one sentence from an answer with the label text it cites.
First copy the exact words from the source that say the same thing as the sentence (the "quote").
Then decide: supported = true when the quote covers what the sentence says, even if the sentence is shorter, reordered or reworded.
supported = false only when the sentence says something that the source does not say anywhere: a different drug, a different number, a condition or a claim that is absent.
Introductory phrases ("According to the label", "Yes,") carry no claim; ignore them. Citation markers like [1] are not claims.

Example. Source: "Metformin is contraindicated in patients with severe renal impairment (eGFR below 30)."
Sentence: "Metformin must not be used in severe renal impairment."
{"quote": "contraindicated in patients with severe renal impairment", "supported": true}

Example. Source: "Avoid concomitant use of omeprazole with clopidogrel."
Sentence: "Omeprazole must be stopped two days before clopidogrel."
{"quote": "", "supported": false}

Return ONLY JSON: {"quote": "...", "supported": true or false}"""


def sentences(answer: str) -> list[str]:
    body = answer.replace(rag.FOOTER, "").strip()
    parts = re.split(r"(?<=[.!?\]])\s+(?=[A-Z*\-•\d(])|\s+[*•]\s+", " ".join(body.split()))
    parts = [p.strip(" *-•") for p in parts]
    # Intro phrases ("The contraindications are:") carry no claim and are not judged.
    return [p for p in parts if len(p) > 15 and not p.endswith(":")]


def citations(sentence: str) -> list[int]:
    return sorted({int(n) for g in re.findall(r"\[([\d,\s]+)\]", sentence) for n in re.findall(r"\d+", g)})


def judge(sentence: str, sources: list[str]) -> tuple[bool, str, llm.Completion]:
    src = "\n\n".join(f"Source {i + 1}:\n{s}" for i, s in enumerate(sources))
    claim = re.sub(r"\s*\[[\d,\s]+\]", "", sentence).strip()
    completion = llm.chat(JUDGE_PROMPT, f"{src}\n\nSentence to check:\n{claim}", json_mode=True)
    text = re.sub(r"^```(?:json)?|```$", "", completion.text.strip(), flags=re.M).strip()
    data = json.loads(text)
    quote = str(data.get("quote") or "")
    # The judge must point at real source text. A "quote" that is not in the source is the judge
    # copying the sentence it was asked to check, and its verdict is discarded.
    supported = bool(data.get("supported")) and bool(quote) and overlap(quote, " ".join(sources)) >= 0.9
    return supported, quote[:200], completion


def pairs_from_answers() -> list[dict]:
    """Every (question, sentence, cited source texts) triple from the latest answer eval."""
    out = []
    for q in json.load(open(ANSWERS))["questions"]:
        if q["kind"] != "in_scope" or q["refused"]:
            continue
        texts = q.get("source_text") or []
        for s in sentences(q["answer"]):
            cites = [n for n in citations(s) if 1 <= n <= len(texts)]
            out.append({"id": q["id"], "sentence": s, "cites": cites, "sources": [texts[n - 1] for n in cites]})
    return out


def validate(pairs_by_key: dict[tuple[str, str], dict]) -> dict | None:
    if not LABELS.exists():
        return None
    labels = [json.loads(l) for l in LABELS.read_text().splitlines() if l.strip()]
    agree, n, disagreements = 0, 0, []
    for lab in labels:
        p = pairs_by_key.get((lab["id"], lab["sentence"]))
        sources = p["sources"] if p else lab.get("sources", [])
        verdict, reason, _ = judge(lab["sentence"], sources)
        n += 1
        if verdict == lab["supported"]:
            agree += 1
        else:
            disagreements.append({"id": lab["id"], "sentence": lab["sentence"][:100], "label": lab["supported"], "judge": verdict, "reason": reason})
    return {"n": n, "agree": agree, "disagreements": disagreements}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true", help="only score the judge against judge_labels.jsonl")
    args = parser.parse_args()

    pairs = pairs_from_answers()
    by_key = {(p["id"], p["sentence"]): p for p in pairs}

    v = validate(by_key)
    if v:
        print(f"judge vs hand labels: {v['agree']}/{v['n']} agree = {v['agree'] / v['n']:.0%}  (model {config.LLM_MODEL})")
        for d in v["disagreements"]:
            print(f"  disagree {d['id']}: label={d['label']} judge={d['judge']}  {d['sentence']}  | {d['reason']}")
    else:
        print("no judge_labels.jsonl: judge not validated")
    if args.validate_only:
        return

    judged, supported, uncited, latency = 0, 0, 0, 0
    per_sentence = []
    for p in pairs:
        if not p["cites"]:
            uncited += 1
            per_sentence.append({**p, "verdict": None})
            continue
        verdict, reason, completion = judge(p["sentence"], p["sources"])
        judged += 1; supported += int(verdict); latency += completion.latency_ms
        per_sentence.append({**p, "verdict": verdict, "reason": reason})
        if not verdict:
            print(f"UNSUPPORTED {p['id']}: {p['sentence'][:110]}  | {reason}")

    print(f"\nsentences: {len(pairs)}  cited and judged: {judged}  uncited: {uncited}")
    if judged:
        print(f"supported by cited source: {supported}/{judged} = {supported / judged:.0%}   judge latency mean {latency // judged} ms")
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({"model": config.LLM_MODEL, "sentences": len(pairs), "judged": judged, "supported": supported,
                                   "uncited": uncited, "validation": v, "per_sentence": per_sentence}, indent=2))


if __name__ == "__main__":
    main()
