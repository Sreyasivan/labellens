"""Token-overlap scoring for extraction. No LLM judge, so it is cheap, deterministic and explainable.

grounded(item, section): share of the item's content words that occur in the section.
    High = the item was taken from the text. Low = the model invented or imported it.
covered(sentence, items): share of the sentence's content words that occur anywhere in the
    extracted items taken together. High = that part of the section made it into the extraction.
Words are crudely stemmed (suffix strip, then cut to 8 chars) so "coadminister" and "co-administration",
or "concurrent" and "concurrently", count as the same word.
"""
import re

STOPWORDS = {
    "the", "and", "or", "of", "to", "in", "with", "for", "a", "an", "is", "are", "be", "by", "on",
    "as", "at", "any", "its", "this", "that", "patients", "patient", "use", "used", "see", "who",
    "have", "has", "known", "including", "not", "should", "contraindicated", "contraindication",
}
_SUFFIXES = ("ation", "ations", "ing", "ly", "ed", "es", "s")


def stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)]
            break
    return word[:8]  # "coadminister" / "coadministration" -> "coadmini"


def tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower().replace("-", ""))
    return {stem(w) for w in words if len(w) > 2 and w not in STOPWORDS}


def overlap(part: str, whole: str) -> float:
    """Fraction of `part`'s content words that appear in `whole`. 1.0 if `part` has no content words."""
    p = tokens(part)
    return 1.0 if not p else len(p & tokens(whole)) / len(p)


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.;])\s+(?=[A-Z(])", " ".join(text.split()))
    return [s for s in parts if len(tokens(s)) >= 2]


def score(items: list[str], section: str, ground_t: float = 0.8, cover_t: float = 0.5) -> dict:
    grounded = [overlap(i, section) >= ground_t for i in items]
    sents = sentences(section)
    all_items = " ".join(items)
    covered = [overlap(s, all_items) >= cover_t for s in sents]
    return {
        "items": len(items),
        "grounded": sum(grounded),
        "sentences": len(sents),
        "covered": sum(covered),
        "ungrounded_items": [i for i, g in zip(items, grounded) if not g],
        "uncovered_sentences": [s for s, c in zip(sents, covered) if not c],
    }
