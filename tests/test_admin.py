"""Admin HTTP contracts using synthetic vectors, never actual Qwen/BGE inference."""

import copy
import json
import socket
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from search_agent.app import create_app
from search_agent.catalog import JsonCatalog
from search_agent.config import QWEN_ID, ModelConfig, Settings
from search_agent.media.config import OpenSearchSettings
from search_agent.media.domain import MediaEntry
from search_agent.media.ingest import media_lock
from search_agent.media.opensearch import OpenSearchStore, index_mapping
from search_agent.media.service import MediaSearchAgent
from search_agent.models import ModelError
from search_agent.service import SearchAgent

API_KEY = "admin-test-only-key-with-at-least-32-characters"
AUTH = {"Authorization": f"Bearer {API_KEY}"}
UPSTREAM_SECRET = "sensitive-upstream-diagnostic-must-not-escape"
IDENTITY = {
    "model": QWEN_ID,
    "dimensions": 2560,
    "revision": "synthetic-test-only",
    "resolved_revision": "synthetic-test-only",
}


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_ROOT", str(tmp_path / "media"))


@pytest.fixture
def entry():
    return {
        "source_id": "admin-fixture:1",
        "canonical_url": "https://example.org/works/admin-fixture-1",
        "original_url": "https://example.org/watch/admin-fixture-1",
        "title": "관리자 테스트 공룡 영상",
        "description": "합성 메타데이터. 실제 콘텐츠나 모델의 품질을 검증하지 않습니다.",
        "tags": ["공룡"],
        "author": "Test fixture",
        "license": "CC0-1.0",
        "license_url": "https://example.org/license",
        "attribution": "Synthetic fixture",
        "rights_verified": True,
        "metadata_provenance": {
            field: {"method": "editorial", "note": "Synthetic test metadata"}
            for field in ["title", "description", "tags"]
        },
    }


def ingest_payload(entry, **options):
    return {"manifest": {"version": 1, "entries": [entry]}, **options}


@pytest.fixture
def sample_client():
    config = Settings(_env_file=None, mode="mock", admin_api_key=API_KEY)
    with TestClient(create_app(config)) as client:
        yield client


class SyntheticEmbedding:
    """A deterministic shape fixture; this is not an inference implementation."""

    model = None
    model_id = QWEN_ID
    resolved_revision = None
    config = ModelConfig()

    def __init__(self):
        self.identity_reads = 0
        self.document_calls = 0
        self.query_calls = 0

    @property
    def identity(self):
        self.identity_reads += 1
        return dict(IDENTITY)

    def documents(self, texts):
        self.document_calls += 1
        vectors = np.zeros((len(texts), 2560), dtype=np.float32)
        vectors[:, 0] = 1
        return vectors

    def query(self, text):
        self.query_calls += 1
        return self.documents([text])[0]


class SyntheticReranker:
    model = None
    resolved_revision = None
    config = ModelConfig()

    def __init__(self):
        self.calls = 0

    def score(self, query, passages):
        self.calls += 1
        return [float(i) for i in range(len(passages))]


class FakeOpenSearch:
    """In-process HTTP fixture for the actual OpenSearchStore serializer and parser."""

    index = "admin-fixture-v1"

    def __init__(self):
        self.mapping = None
        self.docs = {}
        self.requests = []
        self.unavailable = False
        self.fail_upsert_ids = set()
        self.corrupt_hit_identity = False
        self.refresh_response = {"_shards": {"successful": 1, "failed": 0}}

    def handle(self, request):
        method, path = request.method, request.url.path
        body = json.loads(request.content) if request.content else None
        self.requests.append((method, path, body))
        if self.unavailable:
            raise httpx.ConnectError(UPSTREAM_SECRET, request=request)
        if method == "GET" and path == "/":
            return httpx.Response(200, json={"version": {"number": "2.19.3"}})
        if method == "GET" and path == f"/{self.index}/_mapping":
            return httpx.Response(
                200 if self.mapping is not None else 404,
                json=self.mapping or {"error": "index_not_found_exception"},
            )
        if method == "PUT" and path == f"/{self.index}":
            self.mapping = {self.index: {"mappings": body["mappings"]}}
            return httpx.Response(200, json={"acknowledged": True})
        if path.startswith("/_cluster/health"):
            return httpx.Response(200, json={"status": "green", "timed_out": False})
        if path == f"/{self.index}/_count":
            return httpx.Response(200, json={"count": len(self.docs)})
        if method == "POST" and path == f"/{self.index}/_refresh":
            return httpx.Response(200, json=self.refresh_response)
        if method == "PUT" and path.startswith(f"/{self.index}/_doc/"):
            content_id = path.rsplit("/", 1)[-1]
            if content_id in self.fail_upsert_ids:
                return httpx.Response(503, json={"error": UPSTREAM_SECRET})
            existed = content_id in self.docs
            self.docs[content_id] = body
            return httpx.Response(200, json={"result": "updated" if existed else "created"})
        if method == "POST" and path == f"/{self.index}/_search":
            if "aggs" in body:
                return httpx.Response(
                    200,
                    json={
                        "aggregations": {
                            field: {
                                "sum_other_doc_count": 0,
                                "buckets": [
                                    {"key": value}
                                    for value in sorted(
                                        {
                                            v
                                            for doc in self.docs.values()
                                            for v in doc.get(field, [])
                                        }
                                    )
                                ],
                            }
                            for field in body["aggs"]
                        }
                    },
                )
            documents = sorted(self.docs.items())
            multi_match = body.get("query", {}).get("multi_match")
            if multi_match:
                query = multi_match["query"].casefold()
                documents = [
                    (key, doc)
                    for key, doc in documents
                    if query in json.dumps(doc, ensure_ascii=False).casefold()
                ]
            offset, size = body.get("from", 0), body.get("size", 10)
            hits = [
                {
                    "_id": "wrong-content-id" if self.corrupt_hit_identity else key,
                    "_source": copy.deepcopy(doc),
                }
                for key, doc in documents[offset : offset + size]
            ]
            return httpx.Response(
                200,
                json={
                    "timed_out": False,
                    "_shards": {"failed": 0},
                    "hits": {"total": {"value": len(documents), "relation": "eq"}, "hits": hits},
                },
            )
        raise AssertionError(f"Unexpected OpenSearch request: {method} {path}")


@dataclass
class MediaHarness:
    client: TestClient
    upstream: FakeOpenSearch
    agent: MediaSearchAgent
    embedding: SyntheticEmbedding
    reranker: SyntheticReranker
    root: Path


@pytest.fixture
def media_harness(tmp_path):
    upstream = FakeOpenSearch()
    upstream_client = httpx.Client(
        base_url="https://opensearch.example.invalid",
        transport=httpx.MockTransport(upstream.handle),
    )
    store = OpenSearchStore(
        OpenSearchSettings(
            _env_file=None,
            url="https://opensearch.example.invalid",
            index=upstream.index,
            username="fixture-user",
            password=UPSTREAM_SECRET,
        ),
        upstream_client,
    )
    config = Settings(_env_file=None, mode="real", backend="opensearch", admin_api_key=API_KEY)
    embedding, reranker = SyntheticEmbedding(), SyntheticReranker()
    agent = MediaSearchAgent(config, store, embedding, reranker)
    with TestClient(create_app(config, agent)) as client:
        yield MediaHarness(client, upstream, agent, embedding, reranker, tmp_path / "media")
    store.close()


@pytest.mark.parametrize("key", ["", "short", "x" * 31, "x" * 32 + " ", "한" * 32])
def test_admin_rejects_weak_or_invalid_configured_keys(key):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, admin_api_key=key)


def test_admin_disabled_by_default_preserves_public_search():
    config = Settings(_env_file=None, mode="mock", admin_api_key=None)
    client = TestClient(create_app(config))
    assert client.get("/health").status_code == 200
    assert client.post("/api/search", json={"query": "5살 공룡"}).status_code == 200
    response = client.get("/api/admin/status", headers=AUTH)
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "ADMIN_DISABLED"
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("GET", "/api/admin/status", None),
        ("GET", "/api/admin/contents", None),
        ("POST", "/api/admin/search", {"query": "공룡"}),
        ("POST", "/api/admin/ingest", {}),
        ("POST", "/api/admin/index/ensure", {"confirmed": True}),
        ("POST", "/api/admin/index/refresh", {"confirmed": True}),
    ],
)
def test_every_admin_endpoint_requires_auth(sample_client, method, path, payload):
    for headers in [{}, {"Authorization": "Bearer wrong-key"}, {"Authorization": "Basic abc"}]:
        response = sample_client.request(method, path, headers=headers, json=payload)
        assert response.status_code == 401
        assert response.json()["detail"]["code"] == "ADMIN_UNAUTHORIZED"
        assert response.headers["cache-control"] == "no-store"


def test_auth_precedes_json_validation(sample_client):
    response = sample_client.post(
        "/api/admin/ingest", content=b"{not-json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 401


def test_status_does_not_load_real_models_or_expose_secrets(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Status must not load a model")

    monkeypatch.setattr("search_agent.models.TransformerAdapter._load", forbidden)
    config = Settings(_env_file=None, mode="real", admin_api_key=API_KEY)
    client = TestClient(create_app(config))
    response = client.get("/api/admin/status", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["backend"] == "sample"
    assert API_KEY not in response.text
    assert config.admin_api_key.get_secret_value() not in repr(config)


def test_sample_browse_pagination_and_search_match_existing_api(sample_client):
    first = sample_client.get("/api/admin/contents", params={"limit": 2}, headers=AUTH).json()
    next_page = sample_client.get(
        "/api/admin/contents", params={"limit": 2, "offset": 2}, headers=AUTH
    ).json()
    assert first["total"] == next_page["total"] == 12
    assert [item["content_id"] for item in first["items"]] == ["sample-001", "sample-002"]
    assert [item["content_id"] for item in next_page["items"]] == ["sample-003", "sample-004"]
    filtered = sample_client.get("/api/admin/contents", params={"q": "공룡"}, headers=AUTH).json()
    assert filtered["total"] == 4
    assert all("공룡" in item["tags"] for item in filtered["items"])
    by_id = sample_client.get(
        "/api/admin/contents", params={"q": "SAMPLE-001"}, headers=AUTH
    ).json()
    assert by_id["total"] == 1 and by_id["items"][0]["content_id"] == "sample-001"
    payload = {"query": "5살 공룡", "top_k": 5, "top_n": 3}
    public = sample_client.post("/api/search", json=payload).json()
    admin_response = sample_client.post("/api/admin/search", json=payload, headers=AUTH)
    assert admin_response.status_code == 200
    admin = admin_response.json()
    assert admin["results"] == public["results"]
    assert admin["conditions"] == public["conditions"]
    assert not admin["model_inference_performed"]
    assert admin["mode"] == "mock"


@pytest.mark.parametrize(
    "params",
    [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"offset": 9999, "limit": 2}],
)
def test_browse_rejects_unbounded_pagination(sample_client, params):
    assert sample_client.get("/api/admin/contents", params=params, headers=AUTH).status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"query": " "},
        {"query": "공룡", "mode": "mock"},
        {"query": "공룡", "top_n": True},
        {"query": "공룡", "top_k": 1, "top_n": 2},
    ],
)
def test_search_reuses_strict_request_validation(sample_client, payload):
    assert sample_client.post("/api/admin/search", headers=AUTH, json=payload).status_code == 422


def test_preview_has_no_models_network_downloads_or_disk_writes(
    sample_client, entry, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Preview must not access models, network, or media bytes")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr("search_agent.models.TransformerAdapter._load", forbidden)
    monkeypatch.setattr("search_agent.media.ingest.LocalMediaStore", forbidden)
    monkeypatch.setattr("search_agent.media.download.URLPolicy.check", forbidden)
    response = sample_client.post("/api/admin/ingest", headers=AUTH, json=ingest_payload(entry))
    assert response.status_code == 200
    preview = response.json()
    assert preview["dry_run"] is True
    assert preview["network_accessed"] is False
    assert preview["media_download_performed"] is False
    assert preview["ingest_mode"] == "metadata_only"
    assert preview["entries"][0]["content_id"] == MediaEntry.model_validate(entry).content_id
    assert not (tmp_path / "media").exists()


@pytest.mark.parametrize("extra", [{"download_media": True}, {"manifest_path": "/etc/passwd"}])
def test_ingest_rejects_downloads_and_server_file_paths(sample_client, entry, extra):
    payload = ingest_payload(entry, **extra)
    response = sample_client.post("/api/admin/ingest", json=payload, headers=AUTH)
    assert response.status_code == 422
    forbidden_entry = dict(entry, local_path="/etc/passwd")
    response = sample_client.post(
        "/api/admin/ingest", json=ingest_payload(forbidden_entry), headers=AUTH
    )
    assert response.status_code == 422


def test_ingest_batch_limit_rejects_before_side_effects(entry, tmp_path):
    config = Settings(_env_file=None, mode="mock", admin_api_key=API_KEY, admin_max_batch_size=1)
    client = TestClient(create_app(config))
    other = dict(entry, canonical_url="https://example.org/works/another")
    response = client.post(
        "/api/admin/ingest",
        headers=AUTH,
        json={"manifest": {"version": 1, "entries": [entry, other]}},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "BATCH_TOO_LARGE"
    assert not (tmp_path / "media").exists()


@pytest.mark.parametrize("chunked", [False, True])
def test_request_body_limit_including_chunked_json(sample_client, chunked):
    oversized = b" " * 2_000_001
    content = iter([oversized[:1_000_000], oversized[1_000_000:]]) if chunked else oversized
    response = sample_client.post(
        "/api/admin/ingest",
        content=content,
        headers={**AUTH, "Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "REQUEST_TOO_LARGE"


def test_validation_errors_do_not_echo_untrusted_input(sample_client):
    response = sample_client.post(
        "/api/admin/search", json={"query": "공룡", "unexpected": UPSTREAM_SECRET}, headers=AUTH
    )
    assert response.status_code == 422
    assert UPSTREAM_SECRET not in response.text


def test_execute_confirmation_and_sample_backend_guard(sample_client, entry):
    unconfirmed = sample_client.post(
        "/api/admin/ingest", json=ingest_payload(entry, dry_run=False), headers=AUTH
    )
    assert unconfirmed.status_code == 409
    assert unconfirmed.json()["detail"]["code"] == "CONFIRMATION_REQUIRED"
    confirmed = sample_client.post(
        "/api/admin/ingest",
        json=ingest_payload(entry, dry_run=False, confirmed=True),
        headers=AUTH,
    )
    assert confirmed.status_code == 409
    assert confirmed.json()["detail"]["code"] == "BACKEND_UNSUPPORTED"


def test_opensearch_status_and_preview_do_not_resolve_model_identity(media_harness, entry):
    harness = media_harness
    status = harness.client.get("/api/admin/status", headers=AUTH)
    assert status.status_code == 200
    assert status.json()["backend"] == "opensearch"
    assert status.json()["index"]["state"] == "missing"
    assert status.json()["index"]["exists"] is False
    assert status.json()["content_count"] is None
    assert API_KEY not in status.text and UPSTREAM_SECRET not in status.text
    prior_requests = len(harness.upstream.requests)
    preview = harness.client.post("/api/admin/ingest", headers=AUTH, json=ingest_payload(entry))
    assert preview.status_code == 200
    assert len(harness.upstream.requests) == prior_requests
    assert harness.embedding.identity_reads == harness.embedding.document_calls == 0
    assert harness.reranker.calls == 0
    assert harness.upstream.mapping is None and not harness.upstream.docs


def test_existing_index_status_does_not_claim_inference_readiness(media_harness, entry):
    harness = media_harness
    harness.upstream.mapping = {
        harness.upstream.index: {"mappings": index_mapping(IDENTITY)["mappings"]}
    }
    harness.upstream.docs["fixture"] = entry
    response = harness.client.get("/api/admin/status", headers=AUTH)
    assert response.status_code == 200
    status = response.json()
    assert status["status"] == "ok" and status["content_count"] == 1
    assert status["index"]["exists"] is True and status["index"]["health"] == "green"
    assert status["index"]["embedding_identity"] == IDENTITY
    assert status["index"]["identity_status"] == "not_checked"
    assert all(model["inference_readiness"] == "not_checked" for model in status["models"].values())
    assert harness.embedding.identity_reads == harness.embedding.document_calls == 0


@pytest.mark.parametrize("operation", ["ensure", "refresh"])
def test_index_operations_require_confirmation(media_harness, operation):
    harness = media_harness
    response = harness.client.post(f"/api/admin/index/{operation}", json={}, headers=AUTH)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "CONFIRMATION_REQUIRED"
    assert not harness.upstream.requests and harness.embedding.identity_reads == 0


@pytest.mark.parametrize("confirmed", ["true", "false", 1])
def test_confirmation_must_be_a_json_boolean(media_harness, confirmed):
    response = media_harness.client.post(
        "/api/admin/index/ensure", json={"confirmed": confirmed}, headers=AUTH
    )
    assert response.status_code == 422
    assert not media_harness.upstream.requests


def test_index_ensure_is_additive_and_refresh_is_explicit(media_harness):
    harness = media_harness
    for _ in range(2):
        response = harness.client.post(
            "/api/admin/index/ensure", json={"confirmed": True}, headers=AUTH
        )
        assert response.status_code == 200
    creates = [
        request
        for request in harness.upstream.requests
        if request[0:2] == ("PUT", f"/{harness.upstream.index}")
    ]
    assert len(creates) == 1
    assert not any("_refresh" in path for _, path, _ in harness.upstream.requests)
    refreshed = harness.client.post(
        "/api/admin/index/refresh", json={"confirmed": True}, headers=AUTH
    )
    assert refreshed.status_code == 200
    assert sum(path.endswith("/_refresh") for _, path, _ in harness.upstream.requests) == 1
    assert not any(method == "DELETE" for method, _, _ in harness.upstream.requests)


@pytest.mark.parametrize(
    "upstream_response",
    [{}, {"_shards": {"failed": 0}}, {"_shards": {"successful": 0, "failed": 0}}],
)
def test_refresh_rejects_missing_or_unsuccessful_shard_results(media_harness, upstream_response):
    harness = media_harness
    harness.upstream.refresh_response = upstream_response
    response = harness.client.post(
        "/api/admin/index/refresh", json={"confirmed": True}, headers=AUTH
    )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "OPENSEARCH_UNAVAILABLE"
    assert not harness.agent.lock.locked()


def test_live_metadata_ingest_persists_once_and_browse_hides_internal_fields(
    media_harness, entry, monkeypatch
):
    harness = media_harness

    def forbidden(*args, **kwargs):
        raise AssertionError("Admin ingest must never download media")

    monkeypatch.setattr("search_agent.media.ingest.LocalMediaStore", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    payload = ingest_payload(entry, dry_run=False, confirmed=True)
    for _ in range(2):
        response = harness.client.post("/api/admin/ingest", json=payload, headers=AUTH)
        assert response.status_code == 200
        result = response.json()
        assert result["indexed"] == 1
        assert result["status"] == "completed"
        assert result["media_download_performed"] is False
    content_id = MediaEntry.model_validate(entry).content_id
    assert list(harness.upstream.docs) == [content_id]
    stored = harness.upstream.docs[content_id]
    assert stored["source_id"] == entry["source_id"]
    assert stored["title"] == entry["title"]
    assert len(stored["embedding"]) == 2560
    assert np.linalg.norm(stored["embedding"]) == pytest.approx(1)
    assert "local_path" not in stored and "checksum_sha256" not in stored
    assert harness.embedding.document_calls == 1  # The second write reuses a verified cache.
    receipt = json.loads((harness.root / f"{content_id}.metadata-receipt.json").read_text())
    assert receipt["status"] == "indexed"
    # Old indexed records may contain a local file path; never return it through admin browse.
    stored["local_path"] = "/private/media/internal.webm"
    response = harness.client.get("/api/admin/contents", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["total"] == 1
    item = response.json()["items"][0]
    assert item["content_id"] == content_id and item["title"] == entry["title"]
    assert not {"embedding", "local_path", "search_text"} & item.keys()
    assert "/private/media" not in response.text


def test_partial_ingest_reports_per_item_failures_without_secrets(media_harness, entry):
    harness = media_harness
    other = dict(
        entry, canonical_url="https://example.org/works/other", source_id="admin-fixture:2"
    )
    harness.upstream.fail_upsert_ids.add(MediaEntry.model_validate(entry).content_id)
    response = harness.client.post(
        "/api/admin/ingest",
        headers=AUTH,
        json={
            "manifest": {"version": 1, "entries": [entry, other]},
            "dry_run": False,
            "confirmed": True,
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "partial_failure" and result["indexed"] == 1
    assert [item["status"] for item in result["entries"]] == ["failed", "indexed"]
    assert result["entries"][0]["stage"] == "index_upsert"
    assert UPSTREAM_SECRET not in response.text and API_KEY not in response.text


def test_admin_search_uses_existing_media_adapters_without_writes(media_harness, entry):
    harness = media_harness
    inserted = harness.client.post(
        "/api/admin/ingest",
        headers=AUTH,
        json=ingest_payload(entry, dry_run=False, confirmed=True),
    )
    assert inserted.status_code == 200
    harness.upstream.requests.clear()
    response = harness.client.post(
        "/api/admin/search", headers=AUTH, json={"query": "공룡", "top_k": 3, "top_n": 1}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["backend"] == "opensearch" and result["candidate_count"] == 1
    assert result["results"][0]["content"]["title"] == entry["title"]
    assert result["results"][0]["retrieval_sources"] == ["keyword", "semantic"]
    assert harness.embedding.query_calls == 1 and harness.reranker.calls == 1
    assert all(method in {"GET", "POST"} for method, _, _ in harness.upstream.requests)
    assert not any(path.endswith("/_refresh") for _, path, _ in harness.upstream.requests)


def test_opensearch_unavailable_is_degraded_and_safely_reported(media_harness):
    harness = media_harness
    harness.upstream.unavailable = True
    status = harness.client.get("/api/admin/status", headers=AUTH)
    assert status.status_code == 200
    assert status.json()["status"] == "degraded"
    assert status.json()["errors"]
    listing = harness.client.get("/api/admin/contents", headers=AUTH)
    assert listing.status_code == 503
    assert listing.json()["detail"]["code"] == "OPENSEARCH_UNAVAILABLE"
    assert UPSTREAM_SECRET not in listing.text + status.text
    assert harness.embedding.identity_reads == 0


def test_busy_guard_prevents_search_and_writes_while_model_in_use(media_harness, entry):
    harness = media_harness
    harness.agent.lock.acquire()
    try:
        requests = [
            ("/api/admin/search", {"query": "공룡"}),
            ("/api/admin/index/ensure", {"confirmed": True}),
            ("/api/admin/index/refresh", {"confirmed": True}),
            ("/api/admin/ingest", ingest_payload(entry, dry_run=False, confirmed=True)),
        ]
        for path, payload in requests:
            response = harness.client.post(path, headers=AUTH, json=payload)
            assert response.status_code == 409
            assert response.json()["detail"]["code"] == "ADMIN_BUSY"
        assert not harness.upstream.requests
        assert harness.embedding.identity_reads == 0
    finally:
        harness.agent.lock.release()
    # A rejected request must not leave the lock stuck for later work.
    assert (
        harness.client.post(
            "/api/admin/index/ensure", headers=AUTH, json={"confirmed": True}
        ).status_code
        == 200
    )


def test_live_ingest_respects_existing_cli_file_lock(media_harness, entry):
    harness = media_harness
    with media_lock(harness.root):
        preview = harness.client.post("/api/admin/ingest", headers=AUTH, json=ingest_payload(entry))
        assert preview.status_code == 200
        response = harness.client.post(
            "/api/admin/ingest",
            headers=AUTH,
            json=ingest_payload(entry, dry_run=False, confirmed=True),
        )
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "ADMIN_BUSY"
        assert not harness.upstream.requests and not harness.agent.lock.locked()


def test_model_failure_is_sanitized_and_releases_lock(monkeypatch):
    config = Settings(_env_file=None, mode="mock", admin_api_key=API_KEY)
    agent = SearchAgent(config, JsonCatalog(Path("data/sample_catalog.json")))
    original = agent._search

    def failed(request):
        raise ModelError("embedding", UPSTREAM_SECRET)

    monkeypatch.setattr(agent, "_search", failed)
    client = TestClient(create_app(config, agent))
    response = client.post("/api/admin/search", headers=AUTH, json={"query": "공룡"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "MODEL_UNAVAILABLE"
    assert response.json()["detail"]["stage"] == "embedding"
    assert UPSTREAM_SECRET not in response.text
    assert not agent.lock.locked()
    monkeypatch.setattr(agent, "_search", original)
    assert client.post("/api/admin/search", headers=AUTH, json={"query": "공룡"}).status_code == 200


def test_browse_rejects_inconsistent_content_identity(media_harness, entry):
    harness = media_harness
    harness.upstream.mapping = {
        harness.upstream.index: {"mappings": index_mapping(IDENTITY)["mappings"]}
    }
    content_id = MediaEntry.model_validate(entry).content_id
    harness.upstream.docs[content_id] = dict(
        entry, content_id=content_id, retrieved_at="2026-10-07T00:00:00Z"
    )
    harness.upstream.corrupt_hit_identity = True
    response = harness.client.get("/api/admin/contents", headers=AUTH)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "OPENSEARCH_UNAVAILABLE"
