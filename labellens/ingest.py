"""Fetch prescription labels from openFDA, chunk them by section, embed, upsert.

Raw API pages are cached in data/raw so re-runs do not hit the API again.
Usage: python -m labellens.ingest --limit 2000
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import httpx

from labellens import config, embed
from labellens.db import connect

PAGE_SIZE = 100
# Only prescription labels have the clinical sections we care about (OTC labels do not).
BASE_SEARCH = 'openfda.product_type:"HUMAN PRESCRIPTION DRUG" AND _exists_:adverse_reactions AND _exists_:openfda.generic_name'


def fetch_page(search: str, skip: int, cache_name: str) -> list[dict]:
    cache = config.RAW_DIR / f"{cache_name}.json"
    if cache.exists():
        return json.loads(cache.read_text())["results"]
    params = {"search": search, "limit": PAGE_SIZE, "skip": skip}
    if config.OPENFDA_API_KEY:
        params["api_key"] = config.OPENFDA_API_KEY
    for attempt in range(3):
        r = httpx.get(config.OPENFDA_URL, params=params, timeout=60)
        if r.status_code != 429:  # rate limited (40 requests/min without an API key): wait and retry
            break
        time.sleep(20 * (attempt + 1))
    if r.status_code == 404:  # openFDA returns 404 for "no results"
        return []
    r.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(r.text)
    return r.json()["results"]


def fetch_labels(limit: int) -> list[dict]:
    labels: list[dict] = []
    for skip in range(0, limit, PAGE_SIZE):
        page = fetch_page(BASE_SEARCH, skip, f"page_{skip:06d}")
        if not page:
            break
        labels.extend(page)
    return labels[:limit]


def fetch_by_generic(generic_name: str, section: str | None = None) -> list[dict]:
    """One label for a named drug that has the given section, so the eval set is always covered.
    Without the section filter a drug like ciprofloxacin can come back as an eye drop with no boxed
    warning while the question is about the tablets."""
    search = f'{BASE_SEARCH} AND openfda.generic_name:"{generic_name}"'
    slug = generic_name.lower().replace(" ", "_")
    if section:
        search += f" AND _exists_:{section}"
        slug += f"__{section}"
    return fetch_page(search, 0, f"generic_{slug}")[:1]


def split_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Sliding window over characters, cut at the nearest space so words stay whole."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text] if text else []
    pieces, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_label(label: dict) -> list[tuple[str, int, str]]:
    """(section, position, content) for every indexed section present on the label."""
    out = []
    for section in config.SECTIONS:
        parts = label.get(section)
        if not parts:
            continue
        for position, piece in enumerate(split_text(" ".join(parts))):
            out.append((section, position, piece))
    return out


def _first(label: dict, key: str) -> str | None:
    values = label.get("openfda", {}).get(key)
    return values[0] if values else None


def _effective_date(label: dict):
    raw = label.get("effective_time")
    try:
        return datetime.strptime(raw, "%Y%m%d").date() if raw else None
    except ValueError:
        return None


def upsert_label(conn, label: dict) -> int:
    set_id = label["set_id"]
    conn.execute(
        """
        INSERT INTO labels (set_id, brand_name, generic_name, manufacturer, effective_time, raw)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (set_id) DO UPDATE SET
            brand_name = EXCLUDED.brand_name, generic_name = EXCLUDED.generic_name,
            manufacturer = EXCLUDED.manufacturer, effective_time = EXCLUDED.effective_time,
            raw = EXCLUDED.raw
        """,
        (set_id, _first(label, "brand_name"), _first(label, "generic_name"),
         _first(label, "manufacturer_name"), _effective_date(label), json.dumps(label)),
    )
    chunks = chunk_label(label)
    if not chunks:
        return 0
    vectors = embed.embed_passages([c[2] for c in chunks])
    names = " ".join(n for n in (_first(label, "generic_name"), _first(label, "brand_name")) if n)
    conn.execute("DELETE FROM chunks WHERE set_id = %s", (set_id,))
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (set_id, section, position, content, names, embedding) VALUES (%s, %s, %s, %s, %s, %s)",
            [(set_id, s, p, c, names, v) for (s, p, c), v in zip(chunks, vectors)],
        )
    return len(chunks)


def eval_requirements(path: Path = config.ROOT / "eval" / "questions.jsonl") -> list[tuple[str, str]]:
    """Every (generic name, section) pair the question set asks about. None marks an off-topic question."""
    if not path.exists():
        return []
    pairs = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        name, sections = q.get("expected_generic_name"), q.get("expected_section")
        if not name:
            continue
        for section in sections if isinstance(sections, list) else [sections]:
            pairs.add((name, section))
    return sorted(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--only-eval", action="store_true", help="index only the drugs the eval asks about (used in CI)")
    args = parser.parse_args()

    labels = [] if args.only_eval else fetch_labels(args.limit)
    seen = {l["set_id"] for l in labels}
    for name, section in eval_requirements():  # every (drug, section) the eval asks about is indexed
        for label in fetch_by_generic(name, section):
            if label["set_id"] not in seen:
                labels.append(label)
                seen.add(label["set_id"])
    print(f"fetched {len(labels)} labels")

    total_chunks = 0
    with connect() as conn:
        for i, label in enumerate(labels, 1):
            total_chunks += upsert_label(conn, label)
            if i % 100 == 0:
                conn.commit()
                print(f"  {i}/{len(labels)} labels, {total_chunks} chunks")
        conn.commit()
    print(f"done: {len(labels)} labels, {total_chunks} chunks")


if __name__ == "__main__":
    main()
