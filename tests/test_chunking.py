from labellens import config
from labellens.ingest import chunk_label, split_text


def test_short_text_is_one_chunk():
    assert split_text("short section") == ["short section"]


def test_empty_text_gives_no_chunks():
    assert split_text("   ") == []


def test_long_text_is_split_at_word_boundaries_with_overlap():
    text = " ".join(f"word{i}" for i in range(1000))
    pieces = split_text(text, max_chars=300, overlap=50)
    assert len(pieces) > 1
    assert all(len(p) <= 300 for p in pieces)
    assert all(not p.startswith(" ") and not p.endswith(" ") for p in pieces)
    # overlap: the start of each piece appears near the end of the previous one
    for prev, nxt in zip(pieces, pieces[1:]):
        assert nxt.split()[0] in prev


def test_chunk_label_only_indexes_known_sections():
    label = {
        "contraindications": ["Do not use in patients with X."],
        "package_label_principal_display_panel": ["NDC 1234 bottle of 30"],
    }
    chunks = chunk_label(label)
    assert [c[0] for c in chunks] == ["contraindications"]
    assert all(section in config.SECTIONS for section, _, _ in chunks)
