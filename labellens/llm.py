"""Thin wrapper over any OpenAI-compatible chat endpoint (Ollama locally, OpenAI in the cloud)."""
import time
from dataclasses import dataclass

from openai import OpenAI

from labellens import config

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    return _client


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    model: str


def chat(system: str, user: str, temperature: float = 0.0) -> Completion:
    start = time.perf_counter()
    resp = _get_client().chat.completions.create(
        model=config.LLM_MODEL,
        temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    usage = resp.usage
    return Completion(
        text=resp.choices[0].message.content or "",
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        latency_ms=int((time.perf_counter() - start) * 1000),
        model=config.LLM_MODEL,
    )
