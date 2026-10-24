"""Retrieve label chunks and answer with citations.

Guardrails, in order:
1. If nothing in the index is close to the question, refuse (out of scope).
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
    distance: float


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Chunk] = field(default_factory=list)
    refused: bool = False
    completion: llm.Completion | None = None


def retrieve(question: str, k: int = config.TOP_K, dedupe: bool = config.DEDUPE_BY_LABEL) -> list[Chunk]:
    """Nearest chunks to the question.

    Many drugs have a dozen near-identical labels from different manufacturers, so the naive
    top-k is often one passage repeated. With dedupe=True we keep the best chunk per
    (drug, section) from a wider candidate pool, so the k results cover k different sections.
    Measured on the seed eval: naive 6/8, per-label 6/8, per-(drug, section) 7/8.
    """
    vector = embed.embed_query(question)
    nearest = """
        SELECT c.id, c.set_id, l.brand_name, l.generic_name, c.section, c.content,
               c.embedding <=> %s AS distance
        FROM chunks c JOIN labels l USING (set_id)
        ORDER BY distance
        LIMIT %s
    """
    if dedupe:
        sql = f"""
            SELECT * FROM (
                SELECT DISTINCT ON (generic_name, section) * FROM ({nearest}) AS nearest
                ORDER BY generic_name, section, distance
            ) AS best_per_drug_section
            ORDER BY distance
            LIMIT %s
        """
        params = (vector, k * config.DEDUPE_POOL, k)
    else:
        sql, params = nearest, (vector, k)
    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [Chunk(*row) for row in rows]


def in_scope(chunks: list[Chunk], max_distance: float = config.MAX_DISTANCE) -> bool:
    return bool(chunks) and chunks[0].distance <= max_distance


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
