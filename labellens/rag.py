"""Retrieve label chunks and answer with citations.

Retrieval, in order:
1. Dense: nearest chunks by cosine distance on bge-small embeddings.
2. Lexical (hybrid mode): Postgres full-text search on the same chunks. Catches exact terms
   that embeddings blur: drug names, "boxed warning", "eGFR".
3. Fuse the two ranked lists with reciprocal rank fusion (no score calibration needed).
4. Keep one chunk per (drug, section). A drug often has 10+ near-identical labels from different
   manufacturers; without this the top-k is one passage repeated.

Guardrails, in order:
1. If nothing in the index is close to the question, refuse. The LLM is never called.
2. The model may only use the retrieved chunks and must cite them as [1], [2], ...
3. Every answer carries a fixed "not medical advice" footer.
"""
from dataclasses import dataclass, field

from labellens import config, embed, llm
from labellens.db import connect

FOOTER = "This is a summary of FDA label text, not medical advice. Talk to a pharmacist or doctor before acting on it."

SYSTEM_PROMPT = """You answer questions about FDA drug labels.
Rules:
- Use ONLY the numbered sources below. If they do not contain the answer, say "The provided label sections do not answer this."
- Cite every claim with the source number in square brackets, e.g. [2].
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


def _lexical(conn, vector, question: str, n: int) -> list[Chunk]:
    query = "websearch_to_tsquery('english', %(q)s)"
    sql = _CANDIDATES.format(where=f"WHERE c.tsv @@ {query}", order=f"ts_rank_cd(c.tsv, {query}) DESC")
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


def dedupe(chunks: list[Chunk]) -> list[Chunk]:
    """One chunk per (drug, section), keeping the first (best ranked)."""
    seen, out = set(), []
    for c in chunks:
        key = (c.generic_name, c.section)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def retrieve(question: str, k: int = config.TOP_K, mode: str = config.RETRIEVAL, dedupe_results: bool = True) -> list[Chunk]:
    vector = embed.embed_query(question)
    pool = k * config.CANDIDATE_POOL
    with connect() as conn:
        ranked = _dense(conn, vector, pool)
        if mode == "hybrid":
            ranked = rrf(ranked, _lexical(conn, vector, question, pool))
    if dedupe_results:
        ranked = dedupe(ranked)
    return ranked[:k]


def in_scope(chunks: list[Chunk], max_distance: float = config.MAX_DISTANCE) -> bool:
    return bool(chunks) and min(c.distance for c in chunks) <= max_distance


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    sources = "\n\n".join(
        f"[{i}] {c.brand_name or c.generic_name} ({c.generic_name}), section: {c.section}\n{c.content}"
        for i, c in enumerate(chunks, 1)
    )
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def answer(question: str) -> Answer:
    chunks = retrieve(question)
    if not in_scope(chunks):
        return Answer(
            question=question,
            text="I can only answer questions about the FDA drug labels in this index, and nothing in it matches that question.",
            refused=True,
        )
    completion = llm.chat(SYSTEM_PROMPT, build_prompt(question, chunks))
    return Answer(
        question=question,
        text=f"{completion.text.strip()}\n\n{FOOTER}",
        citations=chunks,
        completion=completion,
    )
