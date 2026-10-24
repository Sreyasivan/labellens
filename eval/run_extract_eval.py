"""Extraction eval on real labels.

For N labels that have the chosen section (default: contraindications), extract a list of items and
score it against the section text:
  grounding precision = grounded items / items        (1.0 means nothing was made up)
  sentence coverage   = covered sentences / sentences (1.0 means nothing was left out)

What this does NOT prove: that the list is correctly segmented or clinically complete. openFDA
gives the section as prose, not as a gold list, so there is no gold list to compare against.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labellens import config, extract  # noqa: E402
from labellens.db import connect  # noqa: E402
from scoring import score  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"


def sample_labels(section: str, n: int, seed: int = 7, max_chars: int = 2500) -> list[tuple[str, str, str]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT set_id, generic_name, raw->%(section)s->>0
            FROM labels
            WHERE raw ? %(section)s AND length(raw->%(section)s->>0) BETWEEN 80 AND %(max_chars)s
            ORDER BY set_id
            """,
            {"section": section, "max_chars": max_chars},
        ).fetchall()
    random.Random(seed).shuffle(rows)
    return rows[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--section", default="contraindications", choices=sorted(extract.ITEM_RULES))
    args = parser.parse_args()

    totals = {"items": 0, "grounded": 0, "sentences": 0, "covered": 0, "latency_ms": 0, "failures": 0}
    per_label = []
    for set_id, generic, section in sample_labels(args.section, args.n):
        try:
            result, completion = extract.extract_items(args.section, section)
        except Exception as e:  # bad JSON etc. counts as a failure, never silently skipped
            totals["failures"] += 1
            per_label.append({"set_id": set_id, "generic": generic, "error": str(e)[:120]})
            print(f"FAIL {generic}: {str(e)[:80]}")
            continue
        s = score(result.items, section)
        for key in ("items", "grounded", "sentences", "covered"):
            totals[key] += s[key]
        totals["latency_ms"] += completion.latency_ms
        per_label.append({"set_id": set_id, "generic": generic, "extracted": result.items, **s, "latency_ms": completion.latency_ms})
        flag = "" if not s["ungrounded_items"] else f"  ungrounded: {s['ungrounded_items']}"
        print(f"{(generic or '?')[:28]:28} items={s['items']:2} grounded={s['grounded']:2} sentences={s['sentences']:2} covered={s['covered']:2}{flag}")

    n_ok = len(per_label) - totals["failures"]
    precision = totals["grounded"] / totals["items"] if totals["items"] else 0
    coverage = totals["covered"] / totals["sentences"] if totals["sentences"] else 0
    print(f"\nsection={args.section}  model={config.LLM_MODEL}  labels={n_ok} ok, {totals['failures']} failed")
    print(f"grounding precision: {totals['grounded']}/{totals['items']} = {precision:.0%}")
    print(f"sentence coverage:   {totals['covered']}/{totals['sentences']} = {coverage:.0%}")
    if n_ok:
        print(f"mean latency:        {totals['latency_ms'] // n_ok} ms")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"extraction_{args.section}.json").write_text(json.dumps(
        {"section": args.section, "model": config.LLM_MODEL, "totals": totals, "precision": precision, "coverage": coverage, "labels": per_label}, indent=2))


if __name__ == "__main__":
    main()
