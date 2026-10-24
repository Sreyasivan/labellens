"""Retrieval eval: for each question, is the right drug + section among the top-k chunks?

Runs every retrieval config so the README table compares them on the same questions.
Answer faithfulness (LLM-judged) is the next metric to add on top of this.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labellens import config, rag  # noqa: E402

QUESTIONS = Path(__file__).parent / "questions.jsonl"
RESULTS = Path(__file__).parent / "results" / "retrieval.json"

# name -> (mode, dedupe)
CONFIGS = {
    "vector, naive top-k": ("vector", False),
    "vector + (drug, section) dedupe": ("vector", True),
    "hybrid (vector + FTS) + dedupe": ("hybrid", True),
}


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]


def is_hit(q: dict, chunks: list[rag.Chunk]) -> bool:
    want_drug = (q.get("expected_generic_name") or "").lower()
    want_section = q.get("expected_section")
    return any(
        want_drug in (c.generic_name or "").lower() and (want_section is None or c.section == want_section)
        for c in chunks
    )


def run_config(questions: list[dict], mode: str, dedupe: bool) -> dict:
    per_question, by_section = [], defaultdict(lambda: [0, 0])
    for q in questions:
        chunks = rag.retrieve(q["question"], k=config.TOP_K, mode=mode, dedupe_results=dedupe)
        if q.get("expected_generic_name"):
            hit, section = is_hit(q, chunks), q.get("expected_section") or "any"
        else:  # out-of-scope question: a hit means we would refuse
            hit, section = not rag.in_scope(chunks), "out_of_scope"
        by_section[section][0] += int(hit)
        by_section[section][1] += 1
        per_question.append({"id": q["id"], "hit": hit, "top": [(c.generic_name, c.section, round(c.distance, 3)) for c in chunks[:3]]})
    hits = sum(r["hit"] for r in per_question)
    return {"hits": hits, "n": len(questions), "by_section": dict(by_section), "questions": per_question}


def main() -> None:
    questions = load_questions()
    results = {name: run_config(questions, *cfg) for name, cfg in CONFIGS.items()}

    print(f"{'config':36} recall@{config.TOP_K}")
    for name, r in results.items():
        print(f"{name:36} {r['hits']}/{r['n']} = {r['hits'] / r['n']:.0%}")

    best = results["hybrid (vector + FTS) + dedupe"]
    print("\nby section (hybrid):")
    for section, (h, n) in sorted(best["by_section"].items()):
        print(f"  {section:28} {h}/{n}")
    for r in best["questions"]:
        if not r["hit"]:
            print(f"\nMISS {r['id']}: top-3 = {r['top']}")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({"k": config.TOP_K, "configs": results}, indent=2))


if __name__ == "__main__":
    main()
