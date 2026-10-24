from labellens import rag
from labellens.llm import Completion


def chunk(distance: float, section: str = "contraindications") -> rag.Chunk:
    return rag.Chunk(1, "set", "Brand", "generic", section, "Do not use with X.", distance)


def test_far_chunks_are_out_of_scope():
    assert not rag.in_scope([chunk(0.9)], max_distance=0.55)


def test_close_chunks_are_in_scope():
    assert rag.in_scope([chunk(0.3)], max_distance=0.55)


def test_no_chunks_is_out_of_scope():
    assert not rag.in_scope([])


def test_refusal_does_not_call_the_llm(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q, k=5: [chunk(0.95)])
    monkeypatch.setattr(rag.llm, "chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called")))
    result = rag.answer("What is the capital of France?")
    assert result.refused and not result.citations


def test_answer_has_citations_and_footer(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q, k=5: [chunk(0.2)])
    monkeypatch.setattr(rag.llm, "chat", lambda *a, **k: Completion("Do not use with X [1].", 10, 5, 1, "test"))
    result = rag.answer("What are the contraindications?")
    assert not result.refused
    assert len(result.citations) == 1
    assert "[1]" in result.text
    assert result.text.endswith(rag.FOOTER)


def test_prompt_numbers_sources():
    prompt = rag.build_prompt("q?", [chunk(0.1), chunk(0.2, "adverse_reactions")])
    assert "[1]" in prompt and "[2]" in prompt
    assert "adverse_reactions" in prompt
