"""End-to-end answer eval: run the full ask path on every question and check the answer itself.

Per in-scope question:
  cited the right drug   at least one cited source is a chunk of the expected drug
  mentions required terms  the answer text contains every `must_mention` term (case-insensitive)
  citations valid        every [n] in the answer points at a source that was actually given
  not refused            the scope check let the question through
Per out-of-scope question:
  refused                the scope check stopped it before the model was called

Also reports latency p50/p95 and token counts, so a model or retrieval change shows its cost.
"""
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labellens import config, rag  # noqa: E402

QUESTIONS = Path(__file__).parent / "questions.jsonl"
RESULTS = Path(__file__).parent / "results" / "answers.json"


def cited_numbers(text: str) -> set[int]:
    return {int(n) for group in re.findall(r"\[([\d,\s]+)\]", text) for n in re.findall(r"\d+", group)}


def check(q: dict, a: rag.Answer) -> dict:
    if not q.get("expected_generic_name"):
        return {"id": q["id"], "kind": "out_of_scope", "ok": a.refused, "refused": a.refused}
    text = a.text.lower()
    want = q["expected_generic_name"].lower()
    right_drug = any(want in (c.generic_name or "").lower() for c in a.citations)
    mentions = [m for m in q["must_mention"] if m.lower() in text]
    cites = cited_numbers(a.text)
    valid_cites = bool(cites) and all(1 <= n <= len(a.citations) for n in cites)
    return {
        "id": q["id"], "kind": "in_scope", "refused": a.refused,
        "right_drug": right_drug, "mentions": len(mentions) == len(q["must_mention"]),
        "valid_citations": valid_cites,
        "ok": not a.refused and right_drug and len(mentions) == len(q["must_mention"]) and valid_cites,
        "missing": [m for m in q["must_mention"] if m not in mentions],
    }


def pct(hits: int, n: int) -> str:
    return f"{hits}/{n} = {hits / n:.0%}" if n else "n/a"


def main() -> None:
    questions = [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]
    rows, latencies, prompt_tokens, completion_tokens = [], [], [], []
    for q in questions:
        a = rag.answer(q["question"])
        r = check(q, a)
        r["answer"] = a.text
        r["sources"] = [(c.generic_name, c.section) for c in a.citations]
        r["source_text"] = [c.content for c in a.citations]  # kept for the faithfulness judge
        rows.append(r)
        if a.completion:
            latencies.append(a.completion.latency_ms)
            prompt_tokens.append(a.completion.prompt_tokens)
            completion_tokens.append(a.completion.completion_tokens)
        if not r["ok"]:
            why = "refused" if r["refused"] and r["kind"] == "in_scope" else ("answered" if r["kind"] == "out_of_scope" else
                  ", ".join(k for k in ("right_drug", "mentions", "valid_citations") if not r[k]))
            print(f"MISS {q['id']}: {why}  missing={r.get('missing', [])}  sources={r['sources'][:3]}")

    in_scope = [r for r in rows if r["kind"] == "in_scope"]
    out = [r for r in rows if r["kind"] == "out_of_scope"]
    summary = {
        "model": config.LLM_MODEL,
        "in_scope": {"n": len(in_scope),
                     "answered_correctly": sum(r["ok"] for r in in_scope),
                     "right_drug_cited": sum(r["right_drug"] for r in in_scope),
                     "mentions_required_terms": sum(r["mentions"] for r in in_scope),
                     "valid_citations": sum(r["valid_citations"] for r in in_scope),
                     "wrongly_refused": sum(r["refused"] for r in in_scope)},
        "out_of_scope": {"n": len(out), "refused": sum(r["ok"] for r in out)},
        "latency_ms": {"p50": int(statistics.median(latencies)) if latencies else None,
                       "p95": int(sorted(latencies)[int(0.95 * (len(latencies) - 1))]) if latencies else None},
        "tokens": {"prompt_mean": int(statistics.mean(prompt_tokens)) if prompt_tokens else None,
                   "completion_mean": int(statistics.mean(completion_tokens)) if completion_tokens else None},
    }
    s, o = summary["in_scope"], summary["out_of_scope"]
    print(f"\nmodel={config.LLM_MODEL}  retrieval={config.RETRIEVAL}")
    print(f"in scope ({s['n']}):")
    print(f"  answered correctly (all checks)  {pct(s['answered_correctly'], s['n'])}")
    print(f"  cited the right drug             {pct(s['right_drug_cited'], s['n'])}")
    print(f"  mentions required terms          {pct(s['mentions_required_terms'], s['n'])}")
    print(f"  all citations valid              {pct(s['valid_citations'], s['n'])}")
    print(f"  wrongly refused                  {s['wrongly_refused']}")
    print(f"out of scope ({o['n']}): refused     {pct(o['refused'], o['n'])}")
    print(f"latency p50/p95: {summary['latency_ms']['p50']}/{summary['latency_ms']['p95']} ms   "
          f"tokens prompt/completion mean: {summary['tokens']['prompt_mean']}/{summary['tokens']['completion_mean']}")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({"summary": summary, "questions": rows}, indent=2))


if __name__ == "__main__":
    main()
