"""All settings in one place. Values come from .env (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://labellens:labellens@localhost:5433/labellens")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384  # must match vector(384) in sql/schema.sql

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY", "")
OPENFDA_URL = "https://api.fda.gov/drug/label.json"

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SCHEMA_SQL = ROOT / "sql" / "schema.sql"

# Label sections we index. Everything else (packaging, SPL metadata) is noise for Q&A.
SECTIONS = [
    "boxed_warning",
    "indications_and_usage",
    "dosage_and_administration",
    "contraindications",
    "warnings_and_cautions",
    "adverse_reactions",
    "drug_interactions",
    "use_in_specific_populations",
    "overdosage",
]

# Retrieval guardrail: if the closest chunk is further than this (cosine distance),
# the question is treated as out of scope. Tune with `make eval`.
MAX_DISTANCE = 0.55
TOP_K = 5

# "hybrid" fuses dense + full-text search; "vector" is dense only (kept for the eval comparison).
RETRIEVAL = os.getenv("RETRIEVAL", "hybrid")
# Each retriever returns TOP_K * CANDIDATE_POOL candidates before fusion and (drug, section) dedupe.
CANDIDATE_POOL = 10
