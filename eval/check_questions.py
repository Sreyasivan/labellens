"""Verify the question set against the indexed labels before trusting any eval number.

For every in-scope question: the expected drug must be in the index, its label must have the
expected section, and that section must contain every `must_mention` term. A question that
fails here is wrong about the label, not a retrieval failure, and the eval must not count it.
Exit code 1 on any failure so CI stops before running the eval.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labellens.db import connect  # noqa: E402

QUESTIONS = Path(__file__).parent / "questions.jsonl"


def section_texts(conn, generic: str, section: str) -> list[str]:
    """Every copy of the section across the drug's indexed labels (manufacturers and combination
    products differ in detail; the question is answerable if any indexed label of the drug has it)."""
    rows = conn.execute(
        """
        SELECT raw->%(section)s->>0 FROM labels
        WHERE lower(generic_name) LIKE %(pattern)s AND raw ? %(section)s
        """,
        {"section": section, "pattern": f"%{generic.lower()}%"},
    ).fetchall()
    return [r[0] for r in rows]


def main() -> int:
    questions = [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]
    failures = 0
    with connect() as conn:
        for q in questions:
            if not q.get("expected_generic_name"):
                continue
            sections = q["expected_section"] if isinstance(q["expected_section"], list) else [q["expected_section"]]
            texts = [t for sec in sections for t in section_texts(conn, q["expected_generic_name"], sec)]
            if not texts:
                failures += 1
                print(f"{q['id']}: no label for {q['expected_generic_name']} with section {sections}")
                continue
            if not any(all(m.lower() in t.lower() for m in q["must_mention"]) for t in texts):
                failures += 1
                print(f"{q['id']}: no {q['expected_generic_name']}/{q['expected_section']} text contains all of {q['must_mention']}")
    in_scope = sum(1 for q in questions if q.get("expected_generic_name"))
    print(f"{len(questions)} questions ({in_scope} in scope, {len(questions) - in_scope} out of scope), {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
