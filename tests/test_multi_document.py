import json

import numpy as np
import pytest
from fakes import FakeModel

from multi_document import (
    CollectionConfig,
    DocumentRouter,
    RouterConfig,
    VectorStore,
    format_answer,
    load_config,
)


def make_store(name, summary, chunks, min_score=0.0):
    config = CollectionConfig(name, f"{name}.txt", f"{name}_summary.txt",
                              chunk_size=100, overlap_percent=10, min_score=min_score)
    return VectorStore(config, summary, chunks, FakeModel().encode(chunks))


@pytest.fixture
def stores():
    return [
        make_store("aaa", "aaaa aaaa", ["aaaa", "aaab", "zzzz"]),
        make_store("bbb", "bbbb bbbb", ["bbbb", "bbbc", "zzzz"]),
    ]


def test_build_uses_the_collection_settings(tmp_path):
    (tmp_path / "doc.txt").write_text("x" * 1000)
    (tmp_path / "doc_summary.txt").write_text("a summary\n")
    config = CollectionConfig("doc", "doc.txt", "doc_summary.txt", chunk_size=200, overlap_percent=20, min_score=0.5)
    store = VectorStore.build(config, FakeModel(), base_dir=tmp_path)
    assert all(len(c) == 200 for c in store.chunks[:-1])
    assert store.vectors.shape == (len(store.chunks), 26)
    assert store.summary == "a summary"


def test_store_search_applies_its_own_min_score():
    store = make_store("aaa", "aaaa", ["aaaa", "zzzz"], min_score=0.9)
    results, best = store.search(FakeModel().encode("aaaa"), top_k=3)
    assert [r.chunk_id for r in results] == [0]          # the "zzzz" chunk is below 0.9
    assert best == pytest.approx(1.0)


def test_store_save_and_load_round_trip(tmp_path, stores):
    path = stores[0].save(tmp_path)
    loaded = VectorStore.load(path)
    assert loaded.config == stores[0].config
    assert loaded.chunks == stores[0].chunks
    assert loaded.summary == stores[0].summary
    np.testing.assert_allclose(loaded.vectors, stores[0].vectors)


def test_router_ranks_collections_by_summary(stores):
    router = DocumentRouter(stores, FakeModel())
    assert [name for name, _ in router.route("bbbb")] == ["bbb", "aaa"]


def test_ask_searches_only_the_selected_collection(stores):
    answer = DocumentRouter(stores, FakeModel()).ask("aaab", top_k=2)
    assert answer.collection == "aaa"
    assert answer.tried == ["aaa"]
    assert [r.text for r in answer.results] == ["aaab", "aaaa"]


def test_ask_falls_back_to_the_next_collection():
    strict = make_store("first", "cccc", ["zzzz"], min_score=0.99)   # wins routing, has no good chunk
    backup = make_store("second", "ccca", ["cccc"], min_score=0.5)
    answer = DocumentRouter([strict, backup], FakeModel(), RouterConfig(max_routes=2)).ask("cccc")
    assert answer.tried == ["first", "second"]
    assert answer.collection == "second"


def test_ask_reports_not_found_below_route_min_score(stores):
    answer = DocumentRouter(stores, FakeModel(), RouterConfig(route_min_score=0.999)).ask("zzzz")
    assert not answer.found and answer.tried == []
    assert "not in the documents provided" in format_answer(answer)


def test_load_config_reads_the_project_file():
    model_name, router, collections = load_config()
    assert model_name
    assert router.max_routes >= 1
    assert {c.name for c in collections} == {"football_clubs", "solar_system", "coffee"}


def test_every_collection_has_its_files_and_ten_paragraphs():
    from semantic_search import load_text
    _, _, collections = load_config()
    for c in collections:
        _, paragraphs = load_text(c.document)
        assert len(paragraphs) >= 10, c.name
        assert open(c.summary).read().strip(), c.name


def test_eval_questions_cover_every_collection():
    with open("data/eval_questions.json") as f:
        questions = json.load(f)
    _, _, collections = load_config()
    assert set(questions["collections"]) == {c.name for c in collections}
