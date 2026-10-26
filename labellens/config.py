"""All settings in one place. Values come from .env (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://labellens:labellens@localhost:5433/labellens")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384  # must match vector(384) in sql/schema.sql

# "openai_compatible" (default: Ollama, or OpenAI/vLLM/Groq via LLM_BASE_URL) or "anthropic" (Claude).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai_compatible")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-5-5" if LLM_PROVIDER == "anthropic" else "llama3.2")
# Answers are 2-6 sentences and extractions are short lists; anything longer is a model looping.
MAX_OUTPUT_TOKENS = 2048

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

# Scope guardrail: if the closest chunk is further than this (cosine distance), the question is
# refused before any model call. On the 58-question set the closest in-scope chunk is never further
# than 0.32 and the closest chunk for an off-topic question is never nearer than 0.39; 0.35 sits in
# that gap. Re-check with `make eval` whenever the question set or the embedding model changes.
MAX_DISTANCE = 0.35
TOP_K = 5

# "hybrid" fuses dense + full-text search; "vector" is dense only (kept for the eval comparison).
RETRIEVAL = os.getenv("RETRIEVAL", "hybrid")
# Each retriever contributes TOP_K * CANDIDATE_POOL distinct (drug, section) candidates to the fusion
# step, fetched from LEG_OVERSAMPLE times as many raw rows so duplicate labels cannot crowd a leg out.
CANDIDATE_POOL = 10
LEG_OVERSAMPLE = 6
