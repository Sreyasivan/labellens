from fastapi.testclient import TestClient

from labellens import api, rag
from labellens.llm import Completion

client = TestClient(api.app)


def chunk(generic="metformin"):
    return rag.Chunk(1, "set", "Glucophage", generic, "contraindications", "Severe renal impairment.", 0.21)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ask_returns_answer_and_numbered_sources(monkeypatch):
    monkeypatch.setattr(api.rag, "answer", lambda q: rag.Answer(q, "Renal [1].", [chunk()], completion=Completion("x", 10, 2, 5, "test")))
    r = client.post("/ask", json={"question": "What are the contraindications of metformin?"})
    body = r.json()
    assert r.status_code == 200 and not body["refused"]
    assert body["sources"][0]["n"] == 1 and body["sources"][0]["generic_name"] == "metformin"
    assert body["model"] == "test"


def test_ask_rejects_empty_question():
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_extract_404_when_drug_not_indexed(monkeypatch):
    monkeypatch.setattr(api, "_section_for", lambda g, s: None)
    assert client.post("/extract", json={"generic_name": "nosuchdrug"}).status_code == 404


def test_extract_rejects_unknown_section():
    assert client.post("/extract", json={"generic_name": "metformin", "section": "packaging"}).status_code == 400


def test_extract_scores_items_against_section(monkeypatch):
    monkeypatch.setattr(api, "_section_for", lambda g, s: ("Glucophage", "Acme", "Severe renal impairment. Hypersensitivity to metformin."))
    monkeypatch.setattr(api.extract, "extract_items",
                        lambda s, t: (api.extract.Items(items=["Severe renal impairment", "Pregnancy"]), Completion("{}", 10, 2, 7, "test")))
    body = client.post("/extract", json={"generic_name": "metformin"}).json()
    assert body["grounded"] == 1 and body["ungrounded_items"] == ["Pregnancy"] and body["sentences"] == 2


def test_index_serves_page():
    r = client.get("/")
    assert r.status_code == 200 and "LabelLens" in r.text
