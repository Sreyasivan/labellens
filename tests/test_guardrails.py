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
    result = rag.answer("What are the contraindications of Brand?")
    assert not result.refused
    assert len(result.citations) == 1
    assert "[1]" in result.text
    assert result.text.endswith(rag.FOOTER)


def test_prompt_numbers_sources():
    prompt = rag.build_prompt("q?", [chunk(0.1), chunk(0.2, "adverse_reactions")])
    assert "[1]" in prompt and "[2]" in prompt
    assert "adverse_reactions" in prompt


def test_rrf_rewards_chunks_present_in_both_lists():
    a, b, c = chunk(0.1), chunk(0.2, "adverse_reactions"), chunk(0.3, "overdosage")
    a.id, b.id, c.id = 1, 2, 3
    fused = rag.rrf([a, b, c], [c, a])
    assert fused[0].id == 1          # 1st and 2nd: best combined
    assert fused[1].id == 3          # 3rd and 1st
    assert fused[2].id == 2          # 2nd only


def test_dedupe_keeps_first_per_drug_and_section():
    first, dup, other = chunk(0.1), chunk(0.2), chunk(0.3, "adverse_reactions")
    first.id, dup.id, other.id = 1, 2, 3
    kept = rag.dedupe([first, dup, other])
    assert [k.id for k in kept] == [1, 3]


def test_in_scope_uses_closest_chunk_not_first():
    assert rag.in_scope([chunk(0.9), chunk(0.2)], max_distance=0.55)


def warfarin(distance: float = 0.2) -> rag.Chunk:
    return rag.Chunk(2, "set", "Coumadin", "WARFARIN SODIUM", "boxed_warning", "WARNING: BLEEDING RISK", distance)


def test_names_drug_matches_generic_or_brand_but_not_the_salt():
    assert rag.names_drug("Does warfarin have a boxed warning?", warfarin())
    assert rag.names_drug("What is the boxed warning for Coumadin?", warfarin())
    assert not rag.names_drug("What is a normal sodium level?", warfarin())


def test_close_chunk_is_still_out_of_scope_when_no_drug_is_named():
    assert not rag.in_scope([warfarin(0.2)], "What are the symptoms of type 2 diabetes?")
    assert rag.in_scope([warfarin(0.2)], "Is warfarin safe in pregnancy?")


def test_named_drug_comes_first_and_order_is_otherwise_kept():
    hub = chunk(0.1, "adverse_reactions")
    w1, w2 = warfarin(0.3), warfarin(0.4)
    w2.section = "contraindications"
    assert rag.named_first("boxed warning for warfarin?", [hub, w1, w2]) == [w1, w2, hub]


def test_refusal_when_question_names_no_indexed_drug(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q, k=5: [warfarin(0.2)])
    monkeypatch.setattr(rag.llm, "chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called")))
    assert rag.answer("How is blood pressure measured at home?").refused


def test_eval_hit_accepts_a_list_of_sections():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
    from run_eval import is_hit
    q = {"expected_generic_name": "metformin", "expected_section": ["contraindications", "warnings_and_cautions"]}
    assert is_hit(q, [rag.Chunk(1, "s", "B", "METFORMIN HCL", "warnings_and_cautions", "x", 0.2)])
    assert not is_hit(q, [rag.Chunk(1, "s", "B", "METFORMIN HCL", "adverse_reactions", "x", 0.2)])
