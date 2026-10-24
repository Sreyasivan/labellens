"""Local embeddings with bge-small (384 dims, free, runs on CPU)."""
import numpy as np

from labellens import config

_model = None  # SentenceTransformer, loaded on first use (importing torch is slow)

# bge models score better when short queries carry this prefix; passages get none.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def embed_passages(texts: list[str]) -> np.ndarray:
    return _get_model().encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)


def embed_query(text: str) -> np.ndarray:
    return _get_model().encode(QUERY_PREFIX + text, normalize_embeddings=True)
