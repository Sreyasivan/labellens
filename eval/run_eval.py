"""Retrieval eval: for each question, is the right drug + section among the top-k chunks?

Week 1: recall@k only. Week 3 adds answer faithfulness and extraction F1.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labellens import config, rag  # noqa: E402

QUESTIONS = Path(__file__).parent / "questions.jsonl"
RESULTS = Path(__file__).parent / "results" / "latest.json"


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]


def is_hit(q: dict, chunks: list[rag.Chunk]) -> bool:
    want_drug = (q.get("expected_generic_name") or "").lower()
    want_section = q.get("expected_section")
    for c in chunks:
        drug_ok = want_drug in (c.generic_name or "").lower()
        section_ok = want_section is None or c.section == want_section
        if drug_ok and section_ok:
            return True
    return False


def main() -> None:
    questions = load_questions()
    per_question, by_section = [], defaultdict(lambda: [0, 0])
    for q in questions:
        if not q.get("expected_generic_name"):
            chunks = rag.retrieve(q["question"])
            hit = not rag.in_scope(chunks)  # out-of-scope question: a hit means we refused
            section = "out_of_scope"
        else:
            chunks = rag.retrieve(q["question"], k=config.TOP_K)
            hit = is_hit(q, chunks)
            section = q.get("expected_section") or "any"
        by_section[section][0] += int(hit)
        by_section[section][1] += 1
        per_question.append({"id": q["id"], "hit": hit, "top": [(c.generic_name, c.section, round(c.distance, 3)) for c in chunks[:3]]})

    hits = sum(r["hit"] for r in per_question)
    print(f"recall@{config.TOP_K}: {hits}/{len(questions)} = {hits / len(questions):.2%}\n")
    print(f"{'section':28} {'hits':>5} {'n':>4}")
    for section, (h, n) in sorted(by_section.items()):
        print(f"{section:28} {h:>5} {n:>4}")
    for r in per_question:
        if not r["hit"]:
            print(f"\nMISS {r['id']}: top-3 = {r['top']}")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({"k": config.TOP_K, "recall": hits / len(questions), "questions": per_question}, indent=2))


if __name__ == "__main__":
    main()
