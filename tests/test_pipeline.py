import numpy as np
import pytest
from fakes import FakeModel

from semantic_search import (
    SemanticSearch,
    chunk_text,
    cosine_similarity,
    format_results,
    load_text,
    overlap_size,
    sliding_windows,
)


# --------------------------------------------------------------------------- loading

def test_load_text_splits_on_blank_lines(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("First  paragraph\nstill first.\n\nSecond paragraph.\n\n\nThird.")
    text, paragraphs = load_text(f)
    assert paragraphs == ["First paragraph still first.", "Second paragraph.", "Third."]
    assert text == "First paragraph still first. Second paragraph. Third."


def test_load_text_falls_back_to_lines(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Line one.\nLine two.\nLine three.")
    _, paragraphs = load_text(f)
    assert paragraphs == ["Line one.", "Line two.", "Line three."]


def test_sample_file_has_at_least_ten_paragraphs():
    _, paragraphs = load_text("data/sample_football_clubs.txt")
    assert len(paragraphs) >= 10


# --------------------------------------------------------------------------- chunking

@pytest.mark.parametrize("size, percent, expected", [(500, 15, 75), (500, 10, 50), (500, 20, 100), (800, 15, 120)])
def test_overlap_size(size, percent, expected):
    assert overlap_size(size, percent) == expected


@pytest.mark.parametrize("percent", [5, 9.9, 20.1, 50])
def test_overlap_outside_10_to_20_percent_is_rejected(percent):
    with pytest.raises(ValueError):
        overlap_size(500, percent)


def test_sliding_windows_cover_the_whole_text():
    windows = sliding_windows(length=1000, size=500, step=425)
    assert windows == [(0, 500), (425, 925), (850, 1000)]


def test_chunks_have_fixed_size_except_the_last():
    text = "abcdefghij" * 100                       # 1000 characters
    chunks = chunk_text(text, chunk_size=200, overlap_percent=15)
    assert all(len(c) == 200 for c in chunks[:-1])
    assert 0 < len(chunks[-1]) <= 200


def test_neighbouring_chunks_share_the_overlap():
    text = "".join(chr(65 + i % 26) for i in range(2000))
    chunks = chunk_text(text, chunk_size=500, overlap_percent=20)
    overlap = overlap_size(500, 20)
    for left, right in zip(chunks, chunks[1:]):
        assert left[-overlap:] == right[:overlap]


def test_chunks_reconstruct_the_original_text():
    text = "The quick brown fox jumps over the lazy dog. " * 40
    size, percent = 300, 15
    chunks = chunk_text(text, size, percent)
    overlap = overlap_size(size, percent)
    rebuilt = chunks[0] + "".join(c[overlap:] for c in chunks[1:])
    assert rebuilt == text


def test_short_text_gives_one_chunk():
    assert chunk_text("short text", chunk_size=500, overlap_percent=15) == ["short text"]


def test_unknown_unit_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("text", unit="words")


# --------------------------------------------------------------------------- similarity

def test_cosine_similarity_known_values():
    query = np.array([1.0, 0.0])
    matrix = np.array([[2.0, 0.0], [0.0, 3.0], [-1.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(cosine_similarity(query, matrix), [1.0, 0.0, -1.0, np.sqrt(0.5)])


def test_cosine_similarity_ignores_vector_length():
    query = np.array([1.0, 2.0, 3.0])
    matrix = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    np.testing.assert_allclose(cosine_similarity(query, matrix), [1.0, 1.0])


# --------------------------------------------------------------------------- search (fake model, no download)

def test_search_ranks_the_most_similar_chunk_first():
    chunks = ["zzzz zzzz", "apple apple", "banana"]
    engine = SemanticSearch(chunks, model=FakeModel())
    results = engine.search("apple", top_k=3, min_score=0.0)
    assert results[0].chunk_id == 1
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].score >= results[1].score >= results[2].score


def test_search_returns_nothing_below_min_score():
    engine = SemanticSearch(["zzzz", "qqqq"], model=FakeModel())
    assert engine.search("apple", min_score=0.9) == []


def test_format_results_reports_missing_answers():
    assert "not in the document" in format_results("Anything?", [], min_score=0.6)
