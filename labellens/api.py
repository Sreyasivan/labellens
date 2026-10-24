"""HTTP API and the small web page that uses it. Run: make api  ->  http://localhost:8000

The API calls the same functions as the CLI and the evals, so the numbers in the README describe
exactly what this serves. Interactive docs are generated at /docs.
"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from labellens import config, extract, rag
from labellens.db import connect

sys.path.insert(0, str(config.ROOT / "eval"))
from scoring import score  # noqa: E402

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="LabelLens", version="0.1.0",
              description="Cited answers and typed extraction over FDA drug labels.")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Source(BaseModel):
    n: int
    brand_name: str | None
    generic_name: str | None
    section: str
    distance: float
    content: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[Source]
    model: str | None = None
    latency_ms: int | None = None


class ExtractRequest(BaseModel):
    generic_name: str = Field(min_length=2, max_length=100)
    section: str = Field(default="contraindications")


class ExtractResponse(BaseModel):
    brand_name: str | None
    manufacturer: str | None
    section: str
    section_text: str
    items: list[str]
    ungrounded_items: list[str]
    grounded: int
    sentences: int
    covered: int
    model: str
    latency_ms: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": config.LLM_MODEL, "retrieval": config.RETRIEVAL}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    a = rag.answer(req.question)
    return AskResponse(
        answer=a.text,
        refused=a.refused,
        sources=[Source(n=i, brand_name=c.brand_name, generic_name=c.generic_name, section=c.section,
                        distance=round(c.distance, 3), content=c.content) for i, c in enumerate(a.citations, 1)],
        model=a.completion.model if a.completion else None,
        latency_ms=a.completion.latency_ms if a.completion else None,
    )


def _section_for(generic_name: str, section: str):
    with connect() as conn:
        return conn.execute(
            """
            SELECT brand_name, manufacturer, raw->%(section)s->>0 FROM labels
            WHERE lower(generic_name) LIKE %(pattern)s AND raw ? %(section)s
            ORDER BY length(raw->%(section)s->>0) DESC LIMIT 1
            """,
            {"section": section, "pattern": f"%{generic_name.lower()}%"},
        ).fetchone()


@app.post("/extract", response_model=ExtractResponse)
def extract_section(req: ExtractRequest) -> ExtractResponse:
    if req.section not in extract.ITEM_RULES:
        raise HTTPException(400, f"section must be one of {sorted(extract.ITEM_RULES)}")
    row = _section_for(req.generic_name, req.section)
    if row is None:
        raise HTTPException(404, f"no indexed label for '{req.generic_name}' with a {req.section} section")
    brand, manufacturer, text = row
    items, completion = extract.extract_items(req.section, text)
    s = score(items.items, text)
    return ExtractResponse(
        brand_name=brand, manufacturer=manufacturer, section=req.section, section_text=text,
        items=items.items, ungrounded_items=s["ungrounded_items"],
        grounded=s["grounded"], sentences=s["sentences"], covered=s["covered"],
        model=completion.model, latency_ms=completion.latency_ms,
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
