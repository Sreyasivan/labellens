import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from run_faithfulness import citations, sentences  # noqa: E402

from labellens import rag  # noqa: E402

ANSWER = ("According to the label, metformin is contraindicated in severe renal impairment [1]. "
          "It must not be used in metabolic acidosis, including diabetic ketoacidosis [1, 4]. "
          "Hypersensitivity is also listed.\n\n" + rag.FOOTER)


def test_sentences_drop_footer_and_keep_citations():
    s = sentences(ANSWER)
    assert len(s) == 3
    assert s[0].endswith("[1].") and rag.FOOTER not in " ".join(s)


def test_citations_parse_single_and_grouped():
    s = sentences(ANSWER)
    assert citations(s[0]) == [1]
    assert citations(s[1]) == [1, 4]
    assert citations(s[2]) == []


def test_bulleted_answers_split_per_bullet():
    s = sentences("The contraindications are: * Severe renal impairment [1]. * Hypersensitivity to metformin [1]. * Metabolic acidosis [1].")
    assert len(s) == 3 and all(citations(x) == [1] for x in s)  # the intro phrase carries no claim
