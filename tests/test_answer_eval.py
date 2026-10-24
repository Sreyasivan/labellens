import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from run_answer_eval import check, cited_numbers  # noqa: E402

from labellens import rag  # noqa: E402


def chunk(generic: str) -> rag.Chunk:
    return rag.Chunk(1, "set", "Brand", generic, "contraindications", "text", 0.2)


Q = {"id": "q", "expected_generic_name": "metformin", "expected_section": "contraindications", "must_mention": ["renal", "acidosis"]}


def test_cited_numbers_handles_single_and_grouped_citations():
    assert cited_numbers("Renal [1]. Acidosis [2, 3] and [4].") == {1, 2, 3, 4}
    assert cited_numbers("no citations") == set()


def test_good_answer_passes_every_check():
    a = rag.Answer("q", "Severe renal impairment [1] and metabolic acidosis [1].", [chunk("metformin")])
    r = check(Q, a)
    assert r["ok"] and r["right_drug"] and r["mentions"] and r["valid_citations"]


def test_citation_pointing_past_the_sources_fails():
    a = rag.Answer("q", "Renal impairment and acidosis [3].", [chunk("metformin")])
    assert not check(Q, a)["valid_citations"]


def test_missing_term_and_wrong_drug_are_reported():
    a = rag.Answer("q", "Renal impairment [1].", [chunk("glipizide")])
    r = check(Q, a)
    assert not r["ok"] and r["missing"] == ["acidosis"] and not r["right_drug"]


def test_out_of_scope_passes_only_when_refused():
    oos = {"id": "x", "expected_generic_name": None, "expected_section": None, "must_mention": []}
    assert check(oos, rag.Answer("q", "refusal", refused=True))["ok"]
    assert not check(oos, rag.Answer("q", "Paris [1].", [chunk("metformin")]))["ok"]
