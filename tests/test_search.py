import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from search_agent.app import create_app
from search_agent.catalog import JsonCatalog
from search_agent.config import BGE_ID, QWEN_ID, ModelConfig, Settings
from search_agent.domain import SearchRequest
from search_agent.extraction import eligible, extract
from search_agent.models import BGEReranker, ModelError, QwenEmbedding, normalize
from search_agent.retrieval import EmbeddingCache, SemanticSearch, fuse
from search_agent.service import SearchAgent


@pytest.fixture
def catalog():
    return JsonCatalog(Path("data/sample_catalog.json"))


class FixtureEmbedding:
    """Contract fixture only; never shipped as a selectable inference adapter."""

    identity: ClassVar[dict] = {"model": QWEN_ID, "revision": "fixture-1", "dimensions": 32}

    def __init__(self):
        self.calls = 0

    def documents(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), 32))
        out[:, 0] = 1
        return out

    def query(self, text):
        out = np.zeros(32)
        out[0] = 1
        return out


class FixtureReranker:
    def score(self, query, passages):
        return [float(i) for i in range(len(passages))]


def real_fixture(tmp_path, catalog, reranker=None):
    return SearchAgent(
        Settings(mode="real"),
        catalog,
        SemanticSearch(FixtureEmbedding(), EmbeddingCache(tmp_path)),
        reranker or FixtureReranker(),
    )


@pytest.mark.parametrize(
    "query,age",
    [
        ("5살 아이가 볼 공룡 영상", 5),
        ("만 5세에게 공룡", 5),
        ("다섯 살 공룡", 5),
        ("3살에게 토리", 3),
        ("공룡", None),
        ("0세", 0),
    ],
)
def test_extract(query, age, catalog):
    conditions = extract(query, catalog.all())
    assert conditions.age == age
    if "공룡" in query:
        assert conditions.topics == ["공룡"]


@pytest.mark.parametrize("query", ["19살", "5살 8살", "-1살", "5.5세"])
def test_invalid_age(query, catalog):
    with pytest.raises(ValueError):
        extract(query, catalog.all())


def test_characters_and_alias(catalog):
    c = extract("토리 캐릭터 티라노", catalog.all())
    assert c.characters == ["토리"] and c.topics == ["공룡"]
    assert extract("없는친구 캐릭터", catalog.all()).characters == ["없는친구"]


def test_hard_age_and_topic_filter(catalog):
    c = extract("5살 공룡", catalog.all())
    ids = [x.id for x in catalog.all() if eligible(x, c)]
    assert ids == ["sample-001", "sample-002", "sample-009"]


def test_fusion_dedup_and_stability():
    result = fuse([["b", "b", "a"], ["a", "c"]], 10)
    assert len(result) == 3 and result[0][0] == "a"
    assert dict(result)["b"] == pytest.approx(1 / 61)
    assert fuse([["b"], ["a"]], 1)[0][0] == "a"


def test_real_contract_order_topn_single_and_repeat(tmp_path, catalog):
    agent = real_fixture(tmp_path, catalog)
    req = SearchRequest(query="5살 공룡", top_k=3, top_n=2)
    a = agent.search(req)
    b = agent.search(req)
    assert a.results == b.results
    assert len(a.results) == 2 and a.candidate_count == 3
    assert a.results[0].reranker_score > a.results[1].reranker_score
    assert len({r.content.id for r in a.results}) == 2
    assert all(5 >= r.content.min_age and 5 <= r.content.max_age for r in a.results)
    assert all(r.retrieval_sources == ["keyword", "semantic"] for r in a.results)
    one = agent.search(SearchRequest(query="5살 토리 공룡", top_k=1, top_n=1))
    assert len(one.results) == 1 and one.results[0].reranker_score == 0.0


@pytest.mark.parametrize("query", ["18살 공룡", "없는친구 캐릭터"])
def test_empty_skips_models(query, catalog):
    agent = SearchAgent(Settings(mode="real"), catalog)
    response = agent.search(SearchRequest(query=query))
    assert response.results == [] and not response.model_inference_performed


def test_mock_explicit_no_model_scores(catalog):
    response = SearchAgent(Settings(mode="mock"), catalog).search(
        SearchRequest(query="토리 캐릭터")
    )
    assert response.mode == "mock" and not response.model_inference_performed
    assert response.results
    assert all(r.reranker_score is None and r.fusion_score is None for r in response.results)


@pytest.mark.parametrize(
    "payload",
    [
        {"query": " "},
        {"query": "x" * 501},
        {"query": "x", "top_n": 0},
        {"query": "x", "top_k": 101},
        {"query": "x", "top_n": True},
        {"query": "x", "top_n": "2"},
        {"query": 3},
        {"query": "x", "mode": "mock"},
    ],
)
def test_request_validation(payload):
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(payload)


def test_settings_and_topn_boundary(catalog):
    with pytest.raises(ValidationError):
        Settings(candidate_top_k=2, result_top_n=3)
    with pytest.raises(ValidationError):
        Settings(dimensions=2561)
    agent = SearchAgent(Settings(mode="mock"), catalog)
    with pytest.raises(ValueError):
        agent.search(SearchRequest(query="공룡", top_k=1))
    assert len(agent.search(SearchRequest(query="공룡", top_k=100, top_n=100)).results) == 4


def test_cache_invalidation_and_corruption(tmp_path, catalog):
    cache = EmbeddingCache(tmp_path)
    adapter = FixtureEmbedding()
    items = catalog.all()[:2]
    first = cache.documents(adapter, items)
    assert np.allclose(first, cache.documents(adapter, items))
    assert adapter.calls == 1
    initial_key = cache.key(adapter.identity, items)
    for field, value in [
        ("model", "different"),
        ("revision", "fixture-2"),
        ("dimensions", 64),
        ("max_length", 128),
    ]:
        assert cache.key(dict(adapter.identity, **{field: value}), items) != initial_key
    changed = [items[0].model_copy(update={"title": "새 제목"}), items[1]]
    assert cache.key(adapter.identity, changed) != initial_key
    (tmp_path / (initial_key + ".npy")).write_bytes(b"broken")
    cache.documents(adapter, items)
    assert adapter.calls == 2


def test_normalization_and_dimension_validation():
    assert np.allclose(normalize([[3, 4]], 1, 2), [[0.6, 0.8]])
    for value in [[[0, 0]], [[np.nan, 1]], [[1, 2, 3]]]:
        with pytest.raises(ValueError):
            normalize(value, 1, 2)


def test_qwen_query_only_prompt_and_model_ids(monkeypatch):
    adapter = QwenEmbedding(ModelConfig(), "/tmp/unused")
    calls = []

    def encode(texts):
        calls.append(texts)
        return np.ones((len(texts), 2560))

    monkeypatch.setattr(adapter, "_encode", encode)
    adapter.query("공룡")
    adapter.documents(["문서 공룡"])
    assert calls[0][0].startswith("Instruct: ") and calls[0][0].endswith("\nQuery: 공룡")
    assert calls[1] == ["문서 공룡"]
    assert adapter.model_id == QWEN_ID and adapter.dimensions == 2560
    assert BGEReranker.model_id == BGE_ID


@pytest.mark.parametrize("stage", ["embedding", "reranker"])
def test_independent_failure_no_fallback(tmp_path, catalog, stage):
    agent = real_fixture(tmp_path, catalog)
    if stage == "embedding":

        class BrokenEmbedding(FixtureEmbedding):
            def query(self, text):
                raise RuntimeError("embedding failed")

        agent.semantic.adapter = BrokenEmbedding()
    else:

        class BrokenReranker:
            def score(self, query, passages):
                raise RuntimeError("reranker failed")

        agent.reranker = BrokenReranker()
    with pytest.raises(ModelError) as error:
        agent.search(SearchRequest(query="공룡"))
    assert error.value.stage == stage


@pytest.mark.parametrize("scores", [[1.0], [float("nan")] * 4])
def test_reject_bad_reranker_contract(tmp_path, catalog, scores):
    class InvalidReranker:
        def score(self, query, passages):
            return scores

    with pytest.raises(ModelError):
        real_fixture(tmp_path, catalog, InvalidReranker()).search(SearchRequest(query="공룡"))


def test_catalog_duplicate_handling(tmp_path, catalog):
    path = tmp_path / "catalog.json"
    row = catalog.all()[0].model_dump()
    path.write_text(json.dumps([row, row]))
    assert len(JsonCatalog(path).all()) == 1
    path.write_text(json.dumps([row, dict(row, title="conflict")]))
    with pytest.raises(ValueError):
        JsonCatalog(path).all()


def test_api_mock_ui_and_errors(catalog):
    client = TestClient(create_app(Settings(mode="mock")))
    assert client.get("/").status_code == 200
    assert "모의 모드" in client.get("/").text
    assert client.get("/health").json()["model_readiness"] == "not_checked"
    for query in ["5살 아이가 볼 공룡 영상", "토리 캐릭터 영상", "18살 공룡 영상"]:
        response = client.post("/api/search", json={"query": query})
        assert response.status_code == 200
        assert response.json()["mode"] == "mock"
    assert client.post("/api/search", json={"query": ""}).status_code == 422
    assert client.post("/api/search", json={"query": "공룡", "top_k": 1}).status_code == 422
    broken = TestClient(create_app(Settings(), SearchAgent(Settings(), catalog)))
    failure = broken.post("/api/search", json={"query": "공룡"})
    assert failure.status_code == 503
    assert failure.json()["detail"]["code"] == "MODEL_UNAVAILABLE"


def test_reranker_ties_sort_by_stable_id(tmp_path, catalog):
    class TiedReranker:
        def score(self, query, passages):
            return [1.0] * len(passages)

    response = real_fixture(tmp_path, catalog, TiedReranker()).search(
        SearchRequest(query="공룡", top_k=4, top_n=4)
    )
    assert [r.content.id for r in response.results] == [
        "sample-001",
        "sample-002",
        "sample-003",
        "sample-009",
    ]
