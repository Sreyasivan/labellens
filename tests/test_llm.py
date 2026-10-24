from types import SimpleNamespace

from labellens import config, llm


def test_default_provider_is_local_openai_compatible():
    assert config.LLM_PROVIDER == "openai_compatible"
    assert "11434" in config.LLM_BASE_URL  # Ollama


def test_anthropic_provider_maps_usage_and_text(monkeypatch):
    calls = {}

    class FakeMessages:
        def create(self, **kw):
            calls.update(kw)
            return SimpleNamespace(
                stop_reason="end_turn", stop_details=None,
                content=[SimpleNamespace(type="text", text='{"items": ["a"]}')],
                usage=SimpleNamespace(input_tokens=11, output_tokens=3),
            )

    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(llm, "_anthropic_client", lambda: SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages())))
    out = llm.chat("sys", "user", json_mode=True)
    assert out.text == '{"items": ["a"]}' and (out.prompt_tokens, out.completion_tokens) == (11, 3)
    assert calls["system"] == "sys" and "temperature" not in calls and calls["fallbacks"] == "default"


def test_anthropic_refusal_raises(monkeypatch):
    import pytest

    class FakeMessages:
        def create(self, **kw):
            return SimpleNamespace(stop_reason="refusal", stop_details=SimpleNamespace(category="bio"), content=[], usage=None)

    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(llm, "_anthropic_client", lambda: SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages())))
    with pytest.raises(RuntimeError, match="refused"):
        llm.chat("sys", "user")
