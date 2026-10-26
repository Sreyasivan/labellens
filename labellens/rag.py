"""Retrieve label chunks and answer with citations.

Retrieval, in order:
1. Dense: nearest chunks by cosine distance on bge-small embeddings.
2. Lexical (hybrid mode): Postgres full-text search on the same chunks, any-term match ranked by
   how many terms hit. Catches exact terms that embeddings blur: drug names, "boxed warning", "eGFR".
3. Fuse the two ranked lists with reciprocal rank fusion (no score calibration needed).
4. If the question names a drug in the index, that drug's chunks come first.
5. Keep one chunk per (drug, section). A drug often has 10+ near-identical labels from different
   manufacturers; without this the top-k is one passage repeated.

Guardrails, in order:
1. Scope, before any model call: the closest chunk must be within a distance threshold AND the
   question must name a drug that is in the index. The threshold alone let through general medical
   questions ("what are the symptoms of type 2 diabetes") that happen to sit close to label text.
2. The model may only use the retrieved chunks and must cite them as [1], [2], ...
3. Every answer carries a fixed "not medical advice" footer.
"""
import re
from dataclasses import dataclass, field

from labellens import config, embed, llm
from labellens.db import connect

FOOTER = "This is a summary of FDA label text, not medical advice. Talk to a pharmacist or doctor before acting on it."

SYSTEM_PROMPT = """You answer questions about FDA drug labels.
Rules:
- Use ONLY the numbered sources below. If they do not contain the answer, say "The provided label sections do not answer this."
- End every sentence with the number of the source it came from, in square brackets: "... fetal toxicity [1]."
- The only valid citations are the source numbers shown below ([1], [2], ...). Never cite the label's own section numbers such as 7.3 or 8.1.
- Never recommend a dose or a treatment for a specific person. Quote what the label says instead.
- Be concise: 2-6 sentences."""


@dataclass
class Chunk:
    id: int
    set_id: str
    brand_name: str | None
    generic_name: str | None
    section: str
    content: str
    distance: float  # cosine distance to the question; lower is closer


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Chunk] = field(default_factory=list)
    refused: bool = False
    completion: llm.Completion | None = None


_CANDIDATES = """
    SELECT c.id, c.set_id, l.brand_name, l.generic_name, c.section, c.content,
           c.embedding <=> %(vec)s AS distance
    FROM chunks c JOIN labels l USING (set_id)
    {where}
    ORDER BY {order}
    LIMIT %(n)s
"""


def _dense(conn, vector, n: int) -> list[Chunk]:
    sql = _CANDIDATES.format(where="", order="distance")
    return [Chunk(*r) for r in conn.execute(sql, {"vec": vector, "n": n}).fetchall()]


# Lexical leg. plainto_tsquery ANDs every word, so one word the label never uses ("say", "cover")
# would remove every match. Instead: match on ANY term, rank by how many distinct terms the chunk
# contains, and only then by Postgres' density rank. Postgres FTS has no IDF, so without the
# distinct-term count a long chunk repeating "warning" ten times beats the one that names the drug.
_TERMS = "regexp_split_to_array(regexp_replace(plainto_tsquery('english', %(q)s)::text, '''', '', 'g'), ' & ')"
_ANY_TERM = "replace(plainto_tsquery('english', %(q)s)::text, '&', '|')::tsquery"
_MATCHED = f"(SELECT count(*) FROM unnest({_TERMS}) term WHERE c.tsv @@ term::tsquery)"


def _lexical(conn, vector, question: str, n: int) -> list[Chunk]:
    sql = _CANDIDATES.format(where=f"WHERE c.tsv @@ {_ANY_TERM}",
                             order=f"{_MATCHED} DESC, ts_rank_cd(c.tsv, {_ANY_TERM}) DESC")
    return [Chunk(*r) for r in conn.execute(sql, {"vec": vector, "q": question, "n": n}).fetchall()]


def rrf(*ranked: list[Chunk], k: int = 60) -> list[Chunk]:
    """Reciprocal rank fusion: score = sum over lists of 1 / (k + rank). Robust, no tuning."""
    scores: dict[int, float] = {}
    by_id: dict[int, Chunk] = {}
    for chunks in ranked:
        for rank, c in enumerate(chunks, 1):
            scores[c.id] = scores.get(c.id, 0.0) + 1.0 / (k + rank)
            by_id[c.id] = c
    return [by_id[i] for i in sorted(scores, key=scores.__getitem__, reverse=True)]


# Words in label names that are not the drug: salts, forms, routes. "WARFARIN SODIUM" names warfarin,
# and a question about "sodium levels" must not count as naming it.
_NOT_A_DRUG = {
    "sodium", "potassium", "calcium", "magnesium", "hydrochloride", "hcl", "sulfate", "sulphate", "acetate",
    "citrate", "tartrate", "succinate", "maleate", "mesylate", "fumarate", "phosphate", "besylate", "bromide",
    "chloride", "nitrate", "injection", "tablets", "tablet", "capsules", "capsule", "oral", "solution",
    "suspension", "extended", "release", "delayed", "film", "coated", "topical", "cream", "ointment",
    "ophthalmic", "nasal", "spray", "patch", "transdermal", "with", "and", "plus", "anhydrous", "monohydrate",
    "dihydrate", "hydrate", "kit", "pediatric", "adult", "powder", "concentrate", "human", "recombinant",
}


def drug_words(chunk: Chunk) -> set[str]:
    words = set()
    for name in (chunk.generic_name, chunk.brand_name):
        for w in re.findall(r"[a-z]+", (name or "").lower()):
            if len(w) >= 4 and w not in _NOT_A_DRUG:
                words.add(w)
    return words


def names_drug(question: str, chunk: Chunk) -> bool:
    """Does the question mention this chunk's drug (generic or brand name)?"""
    q = question.lower()
    return any(w in q for w in drug_words(chunk))


def named_first(question: str, chunks: list[Chunk]) -> list[Chunk]:
    """Stable: chunks of a drug the question names come before everything else. Generic passages
    ("the following adverse reactions are discussed elsewhere...") sit close to every question in
    embedding space; this is what stops them crowding out the drug the user actually asked about."""
    return sorted(chunks, key=lambda c: not names_drug(question, c))


def dedupe(chunks: list[Chunk]) -> list[Chunk]:
    """One chunk per (drug, section), keeping the first (best ranked)."""
    seen, out = set(), []
    for c in chunks:
        key = (c.generic_name, c.section)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def retrieve(question: str, k: int = config.TOP_K, mode: str = config.RETRIEVAL,
             dedupe_results: bool = True, drug_first: bool = True) -> list[Chunk]:
    vector = embed.embed_query(question)
    pool = k * config.CANDIDATE_POOL
    # Each leg fetches more rows than the pool and is deduped to one chunk per (drug, section) before
    # fusion. Otherwise a drug with 18 manufacturers' labels fills a leg's pool with one section
    # repeated and its other sections never reach the fusion step.
    fetch = pool * config.LEG_OVERSAMPLE if dedupe_results else pool
    with connect() as conn:
        legs = [_dense(conn, vector, fetch)]
        if mode == "hybrid":
            legs.append(_lexical(conn, vector, question, fetch))
    if dedupe_results:
        legs = [dedupe(leg)[:pool] for leg in legs]
    ranked = rrf(*legs) if len(legs) > 1 else legs[0]
    if drug_first:
        ranked = named_first(question, ranked)
    if dedupe_results:
        ranked = dedupe(ranked)
    return ranked[:k]


def in_scope(chunks: list[Chunk], question: str = "", max_distance: float = config.MAX_DISTANCE,
             require_drug: bool = True) -> bool:
    """Two signals, both required: something in the index is close, and the question names a drug
    that was retrieved. An empty question skips the second check (unit tests, programmatic use)."""
    if not chunks or min(c.distance for c in chunks) > max_distance:
        return False
    if require_drug and question:
        return any(names_drug(question, c) for c in chunks)
    return True


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    sources = "\n\n".join(
        f"[{i}] {c.brand_name or c.generic_name} ({c.generic_name}), section: {c.section}\n{c.content}"
        for i, c in enumerate(chunks, 1)
    )
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def answer(question: str) -> Answer:
    chunks = retrieve(question)
    if not in_scope(chunks, question):
        return Answer(
            question=question,
            text="I can only answer questions about a specific drug whose FDA label is in this index. "
                 "Name the drug (generic or brand) and ask about its label.",
            refused=True,
        )
    completion = llm.chat(SYSTEM_PROMPT, build_prompt(question, chunks))
    return Answer(
        question=question,
        text=f"{completion.text.strip()}\n\n{FOOTER}",
        citations=chunks,
        completion=completion,
    )
