import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from scoring import overlap, score, sentences  # noqa: E402

from labellens import extract  # noqa: E402

SECTION = ("Severe renal impairment (eGFR below 30 mL/min/1.73 m2). Hypersensitivity to metformin. "
           "Acute or chronic metabolic acidosis, including diabetic ketoacidosis, with or without coma.")


def test_parse_accepts_fenced_json():
    out = extract._parse('```json\n{"items": ["a", "b"]}\n```')
    assert out.items == ["a", "b"]


def test_parse_accepts_bare_list():
    assert extract._parse('["x"]').items == ["x"]


def test_sentences_split_on_periods():
    assert len(sentences(SECTION)) == 3


def test_overlap_is_fraction_of_item_words_found():
    assert overlap("hypersensitivity to metformin", SECTION) == 1.0
    assert overlap("pregnancy", SECTION) == 0.0


def test_score_flags_invented_items_and_missed_sentences():
    s = score(["Severe renal impairment eGFR below 30", "Pregnancy and lactation"], SECTION)
    assert s["items"] == 2 and s["grounded"] == 1
    assert s["ungrounded_items"] == ["Pregnancy and lactation"]
    assert s["sentences"] == 3 and s["covered"] == 1


def test_stemming_matches_inflections():
    assert overlap("co-administration with aliskiren", "do not coadminister aliskiren") == 1.0
    assert overlap("concurrent use", "used concurrently") == 1.0


def test_coverage_uses_all_items_together():
    section = "Contraindicated in pregnancy, severe hepatic impairment, and hypersensitivity to estradiol."
    items = ["pregnancy", "severe hepatic impairment", "hypersensitivity to estradiol"]
    assert score(items, section)["covered"] == 1
