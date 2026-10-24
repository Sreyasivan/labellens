"""One chat() function over two providers.

  LLM_PROVIDER=openai_compatible  (default) any OpenAI-compatible endpoint: Ollama locally, OpenAI,
                                  vLLM, Groq, ... Selected by LLM_BASE_URL / LLM_API_KEY / LLM_MODEL.
  LLM_PROVIDER=anthropic          Claude through the official anthropic SDK. Key from ANTHROPIC_API_KEY.

Both return the same Completion, so retrieval, extraction and the evals never know which one ran.
"""
import time
from dataclasses import dataclass

from labellens import config

_client = None


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    model: str


def _openai_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    return _client


def _anthropic_client():
    global _client
    if _client is None:
        try:
            import anthropic
        except ImportError as e:  # the SDK is an optional extra so the local default stays small
            raise RuntimeError("LLM_PROVIDER=anthropic needs the anthropic SDK: uv sync --extra anthropic") from e
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    return _client


def _chat_openai(system: str, user: str, temperature: float, json_mode: bool) -> tuple[str, int, int]:
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = _openai_client().chat.completions.create(
        model=config.LLM_MODEL,
        temperature=temperature,
        max_tokens=config.MAX_OUTPUT_TOKENS,  # a 3B model occasionally loops; this ends it
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **extra,
    )
    usage = resp.usage
    return (resp.choices[0].message.content or "",
            usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0)


def _chat_anthropic(system: str, user: str, json_mode: bool) -> tuple[str, int, int]:
    # No temperature: current Claude models reject sampling parameters. JSON comes from the prompt;
    # extract._parse tolerates code fences. fallbacks="default" re-runs a safety-classifier refusal on
    # the model Anthropic recommends, so a false positive on clinical text does not become a failure.
    resp = _anthropic_client().beta.messages.create(
        model=config.LLM_MODEL,
        max_tokens=config.MAX_OUTPUT_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"model refused the request: {getattr(resp.stop_details, 'category', None)}")
    text = "".join(block.text for block in resp.content if block.type == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def chat(system: str, user: str, temperature: float = 0.0, json_mode: bool = False) -> Completion:
    start = time.perf_counter()
    if config.LLM_PROVIDER == "anthropic":
        text, prompt_tokens, completion_tokens = _chat_anthropic(system, user, json_mode)
    else:
        text, prompt_tokens, completion_tokens = _chat_openai(system, user, temperature, json_mode)
    return Completion(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=int((time.perf_counter() - start) * 1000),
        model=config.LLM_MODEL,
    )
