import hashlib
import io
import json
import socket
from pathlib import Path
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
from pydantic import ValidationError

from search_agent.config import QWEN_ID, Settings
from search_agent.domain import Conditions, SearchRequest
from search_agent.media.config import MediaSettings, OpenSearchSettings
from search_agent.media.domain import MediaEntry, MediaManifest, MediaRecord
from search_agent.media.download import (
    DownloadError,
    LocalMediaStore,
    PinnedHTTPSTransport,
    UnsafeDownload,
    URLPolicy,
    valid_magic,
)
from search_agent.media.ingest import Ingestor, adapt_commons_source, load_manifest
from search_agent.media.opensearch import (
    OpenSearchError,
    OpenSearchStore,
    bm25_query,
    hard_filter,
    index_mapping,
    vector_query,
)
from search_agent.media.service import MediaSearchAgent

FIXTURE_BYTES = b"\x1a\x45\xdf\xa3" + b"fictional webm-like fixture, not playable media"


@pytest.fixture
def entry():
    provenance = {
        key: {"method": "editorial", "note": "Synthetic test metadata, not real media"}
        for key in ["title", "description", "tags", "language", "duration_seconds"]
    }
    return MediaEntry(
        canonical_url="https://example.org/media/test",
        original_url="https://media.example.org/test.webm",
        title="테스트 펭귄",
        description="테스트 전용 자료",
        tags=["동물", "펭귄"],
        author="Test fixture",
        license="CC0-1.0",
        license_url="https://example.org/license",
        attribution="Synthetic test fixture",
        media_format="webm",
        rights_verified=True,
        metadata_provenance=provenance,
    )


@pytest.fixture
def record(entry):
    return MediaRecord(
        **entry.model_dump(),
        local_path=entry.content_id + ".webm",
        checksum_sha256=hashlib.sha256(FIXTURE_BYTES).hexdigest(),
        size_bytes=len(FIXTURE_BYTES),
        retrieved_at="2026-10-03T00:00:00Z",
    )


class Response:
    def __init__(self, body=FIXTURE_BYTES, status=200, headers=None):
        self.status = status
        self.headers = headers if headers is not None else {"Content-Type": "video/webm"}
        self.stream = io.BytesIO(body)
        self.closed = False

    def getheader(self, key, default=None):
        return self.headers.get(key, default)

    def read(self, size):
        return self.stream.read(size)

    def close(self):
        self.closed = True

    def set_timeout(self, seconds):
        assert seconds > 0


def storage(tmp_path, responses, **kwargs):
    transport = Mock()
    transport.open.side_effect = responses
    settings = MediaSettings(root=tmp_path, allowed_domains=["media.example.org"], **kwargs)
    return LocalMediaStore(settings, transport)


@pytest.mark.parametrize(
    "url",
    [
        "http://media.example.org/a",
        "https://media.example.org.evil.test/a",
        "https://user:password@media.example.org/a",
        "https://media.example.org:8443/a",
        "file:///etc/passwd",
        "https://media.example.org/a#x",
        "https://media.example.org/\nx",
    ],
)
def test_url_policy_rejects(url):
    with pytest.raises((ValueError, UnsafeDownload)):
        URLPolicy(["media.example.org"]).check(url)


@pytest.mark.parametrize(
    "address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "fc00::1", "0.0.0.0"]
)
def test_dns_ssrf(monkeypatch, address):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [(2, 1, 6, "", (address, 443))])
    with pytest.raises(UnsafeDownload):
        URLPolicy(["media.example.org"]).resolve("https://media.example.org/a")


def test_public_dns_and_proxy_fail_closed(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [(2, 1, 6, "", ("1.1.1.1", 443))])
    policy = URLPolicy(["media.example.org"])
    assert policy.resolve("https://media.example.org/a") == ("media.example.org", "1.1.1.1")
    monkeypatch.setattr("search_agent.media.download.getproxies", lambda: {"https": "http://proxy"})
    with pytest.raises(UnsafeDownload, match="proxy"):
        PinnedHTTPSTransport().open("https://media.example.org/a", policy, 1)


def test_atomic_download_checksum_and_reuse(entry, tmp_path):
    expected = hashlib.sha256(FIXTURE_BYTES).hexdigest()
    entry = entry.model_copy(update={"expected_sha256": expected})
    store = storage(tmp_path, [Response()])
    first = store.download(entry)
    second = store.download(entry)
    assert first.checksum_sha256 == expected and second.reused
    assert store.transport.open.call_count == 1
    assert not list(tmp_path.glob(".download-*"))
    assert (tmp_path / first.local_path).read_bytes() == FIXTURE_BYTES


@pytest.mark.parametrize(
    "response,updates",
    [
        (Response(headers={"Content-Type": "text/html"}), {}),
        (Response(body=b"<html>error</html>"), {}),
        (Response(headers={"Content-Type": "video/webm", "Content-Length": "99999999"}), {}),
        (Response(headers={"Content-Type": "video/webm", "Content-Encoding": "gzip"}), {}),
        (Response(), {"expected_sha256": "0" * 64}),
    ],
)
def test_download_rejections_cleanup(entry, tmp_path, response, updates):
    store = storage(tmp_path, [response], retries=0)
    with pytest.raises(DownloadError):
        store.download(entry.model_copy(update=updates))
    assert not list(tmp_path.iterdir())


def test_stream_limit_redirect_and_symlink(entry, tmp_path):
    store = storage(tmp_path, [Response()], max_bytes=10)
    with pytest.raises(UnsafeDownload):
        store.download(entry)
    redirect = Response(status=302, headers={"Location": "https://localhost/private"})
    store = storage(tmp_path, [redirect])
    with pytest.raises(UnsafeDownload):
        store.download(entry)
    assert store.transport.open.call_count == 1 and redirect.closed
    store.target(entry).symlink_to(tmp_path / "outside")
    with pytest.raises(UnsafeDownload):
        store.download(entry)


def test_redirect_success_and_retry(entry, tmp_path):
    redirect = Response(status=302, headers={"Location": "/new.webm"})
    store = storage(tmp_path, [OSError("temporary"), redirect, Response()], retries=1)
    assert store.download(entry).size_bytes == len(FIXTURE_BYTES)
    assert store.transport.open.call_count == 3
    assert store.transport.open.call_args.args[0] == "https://media.example.org/new.webm"


def test_unknown_age_subtitle_validation_and_manifest_dedup(entry):
    assert entry.min_age is None and entry.max_age is None and entry.language == "und"
    assert entry.subtitle_text is None
    assert len(MediaManifest(version=1, entries=[entry, entry]).entries) == 1
    for updates in [
        {"min_age": 5},
        {"min_age": 3, "max_age": 6},
        {"subtitle_text": "invented"},
        {"local_path": "../../bad"},
        {"media_format": "../../webm"},
        {"rights_verified": False},
    ]:
        with pytest.raises(ValidationError):
            MediaEntry.model_validate(entry.model_dump() | updates)
    with pytest.raises(ValidationError):
        MediaManifest(version=1, entries=[entry, entry.model_copy(update={"title": "conflict"})])


def test_source_manifest_preservation():
    source = json.loads(Path("data/source_manifest.json").read_text())
    manifest = adapt_commons_source(source)
    assert len(manifest.entries) == 8
    assert all(e.min_age is None and e.subtitle_text is None for e in manifest.entries)
    assert manifest.entries[0].language == "ja"
    assert manifest.entries[1].language == "und"
    assert manifest.entries[3].language == "zxx"
    assert all(
        e.metadata_provenance["title"].method == "assistant_translation" for e in manifest.entries
    )
    assert all(e.metadata_license == "CC-BY-SA-4.0" for e in manifest.entries)
    assert manifest.entries[-1].license == "PD-USGov-NASA"
    assert (
        manifest.entries[3].expected_sha256
        == "9c175ea0a1f25cea60a1d88c04cbbdf2f0294c29cf91738062a3c7bd86e84545"
    )
    assert len(load_manifest(Path("data/media_manifest.json")).entries) == 8


IDENTITY = {
    "model": QWEN_ID,
    "revision": "test-fixture",
    "resolved_revision": "test-fixture-commit",
    "dimensions": 2560,
}


class FixtureEmbedding:
    @property
    def identity(self):
        return dict(IDENTITY)

    def documents(self, texts):
        return np.ones((len(texts), 2560))

    def query(self, text):
        return np.ones(2560)


def test_mapping_and_query_contract():
    mapping = index_mapping(IDENTITY)
    vector = mapping["mappings"]["properties"]["embedding"]
    assert vector["dimension"] == 2560 and vector["method"]["engine"] == "faiss"
    assert vector["method"]["space_type"] == "cosinesimil"
    assert mapping["mappings"]["properties"]["metadata_provenance"]["enabled"] is False
    conditions = Conditions(age=5, topics=["동물"], characters=["test"])
    bm25 = bm25_query("펭귄", conditions, 10)
    knn = vector_query(np.ones(2560), conditions, 10)
    assert bm25["query"]["bool"]["filter"][0] == knn["query"]["knn"]["embedding"]["filter"]
    assert hard_filter(conditions)["bool"]["filter"][:2] == [
        {"range": {"min_age": {"lte": 5}}},
        {"range": {"max_age": {"gte": 5}}},
    ]
    assert len(knn["query"]["knn"]["embedding"]["vector"]) == 2560
    with pytest.raises(ValueError):
        index_mapping(IDENTITY | {"dimensions": 32})


def test_opensearch_secure_defaults_and_errors(record):
    with pytest.raises(ValidationError):
        OpenSearchSettings(url="http://remote.example.org:9200")
    with pytest.raises(ValidationError):
        OpenSearchSettings(url="https://user:pass@example.org")
    with pytest.raises(ValidationError):
        Settings(backend="opensearch", mode="mock")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"result": "updated"})

    client = httpx.Client(base_url="https://localhost:9200", transport=httpx.MockTransport(handler))
    store = OpenSearchStore(OpenSearchSettings(), client)
    store.upsert(record, np.ones(2560))
    store.upsert(record, np.ones(2560))
    assert seen[0].url == seen[1].url
    body = json.loads(seen[0].content)
    assert body["content_id"] == record.content_id and body["min_age"] is None
    assert body["metadata_provenance"] and body["checksum_sha256"]
    assert np.linalg.norm(body["embedding"]) == pytest.approx(1)
    with pytest.raises(OpenSearchError):
        store.validate_mapping(
            {store.index: index_mapping(IDENTITY)}, IDENTITY | {"resolved_revision": "other"}
        )


def test_partial_failure_then_recover(entry, tmp_path):
    store = storage(tmp_path, [Response()])
    index = Mock()
    index.upsert.side_effect = [OpenSearchError("temporary failure"), {"result": "created"}]
    ingestor = Ingestor(store.settings, store, FixtureEmbedding(), index)
    manifest = MediaManifest(version=1, entries=[entry, entry])
    assert ingestor.run(manifest)["dry_run"]
    assert not list(tmp_path.iterdir())
    first = ingestor.run(manifest, dry_run=False)
    assert first["status"] == "partial_failure" and first["entries"][0]["stage"] == "index_upsert"
    second = ingestor.run(manifest, dry_run=False)
    assert second["indexed"] == 1 and second["entries"][0]["download"]["reused"]
    assert store.transport.open.call_count == 1
    assert index.upsert.call_count == 2


def test_download_only_no_model(entry, tmp_path):
    store = storage(tmp_path, [Response()])
    ingestor = Ingestor(store.settings, store)
    result = ingestor.run(
        MediaManifest(version=1, entries=[entry]), dry_run=False, download_only=True
    )
    assert result["indexed"] == 0 and result["downloaded_only"] == 1


def test_media_search_reranks_and_empty_without_models(record):
    store = Mock()
    store.vocabulary.return_value = (["펭귄"], [])
    store.count.return_value = 0
    embedding, reranker = Mock(), Mock()
    service = MediaSearchAgent(Settings(backend="opensearch"), store, embedding, reranker)
    empty = service.search(SearchRequest(query="5살 펭귄"))
    assert not empty.results and not embedding.query.called and not reranker.score.called
    store.count.return_value = 1
    store.search.return_value = [record]
    embedding.query.return_value = np.ones(2560)
    reranker.score.return_value = [2.0]
    found = service.search(SearchRequest(query="펭귄", top_k=1, top_n=1))
    assert found.results[0].content_id == record.content_id
    assert found.results[0].evidence[0] == "연령등급 미상"
    assert found.results[0].retrieval_sources == ["keyword", "semantic"]
    assert found.results[0].reranker_score == 2.0
    reranker.score.return_value = [float("nan")]
    with pytest.raises(Exception, match="reranker"):
        service.search(SearchRequest(query="펭귄"))


@pytest.mark.parametrize(
    "fmt,content",
    [
        ("webm", b"\x1aE\xdf\xa3"),
        ("ogv", b"OggS"),
        ("mp4", b"\x00\x00\x00\x20ftyp"),
        ("wav", b"RIFFxxxxWAVE"),
        ("mp3", b"ID3"),
        ("flac", b"fLaC"),
    ],
)
def test_media_magic(fmt, content):
    assert valid_magic(fmt, content)
    assert not valid_magic(fmt, b"<html>Forbidden</html>")


def test_local_import_requires_checksum_and_never_downloads(entry, tmp_path):
    from search_agent.media.local_import import import_local

    source = tmp_path / "provided.webm"
    source.write_bytes(FIXTURE_BYTES)
    settings = MediaSettings(root=tmp_path / "media")
    with pytest.raises(ValueError, match="checksum"):
        import_local(entry, source, settings)
    verified = entry.model_copy(
        update={"expected_sha256": hashlib.sha256(FIXTURE_BYTES).hexdigest()}
    )
    receipt = import_local(verified, source, settings)
    assert receipt["url_download_performed"] is False
    assert receipt["acquisition"] == "explicit_local_file"
    assert (settings.root / receipt["download"]["local_path"]).read_bytes() == FIXTURE_BYTES


def test_redirect_loop_and_no_retry_on_access_denied(entry, tmp_path):
    redirect = lambda: Response(status=302, headers={"Location": "/again.webm"})
    store = storage(tmp_path, [redirect(), redirect()], max_redirects=1)
    with pytest.raises(UnsafeDownload):
        store.download(entry)
    assert store.transport.open.call_count == 2
    store = storage(tmp_path, [Response(status=403)])
    with pytest.raises(UnsafeDownload):
        store.download(entry)
    assert store.transport.open.call_count == 1


def test_download_deadline_and_dns_timeout(entry, tmp_path, monkeypatch):
    import time

    store = storage(tmp_path, [Response()])
    with pytest.raises(DownloadError, match="deadline"):
        store._attempt(entry, store.target(entry), deadline=time.monotonic() - 1)
    assert not store.transport.open.called
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: time.sleep(0.1))
    with pytest.raises(DownloadError, match="DNS deadline"):
        URLPolicy(["media.example.org"]).resolve(entry.original_url, timeout=0.01)


def test_batch_budget_partial_failure(entry, tmp_path):
    second = entry.model_copy(update={"canonical_url": "https://example.org/second"})
    store = storage(tmp_path, [Response()], total_max_bytes=len(FIXTURE_BYTES))
    result = Ingestor(store.settings, store).run(
        MediaManifest(version=1, entries=[entry, second]), dry_run=False, download_only=True
    )
    assert result["status"] == "partial_failure" and result["downloaded_only"] == 1
    assert result["entries"][1]["error_type"] == "ValueError"
    assert store.transport.open.call_count == 1


def test_media_api_backend_and_no_auto_fallback():
    from fastapi.testclient import TestClient

    from search_agent.app import create_app

    config = Settings(backend="opensearch")
    agent = Mock()
    agent.search.side_effect = OpenSearchError("Index unavailable")
    client = TestClient(create_app(config, agent))
    health = client.get("/health").json()
    assert health["backend"] == "opensearch" and health["sample"] is False
    response = client.post("/api/search", json={"query": "펭귄"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "OPENSEARCH_UNAVAILABLE"


def test_pinned_transport_uses_validated_ip_and_original_tls_name(monkeypatch):
    import http.client
    import ssl

    monkeypatch.setattr("search_agent.media.download.getproxies", dict)
    resolver = Mock(return_value=[(2, 1, 6, "", ("1.1.1.1", 443))])
    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    raw = Mock()
    connect = Mock(return_value=raw)
    monkeypatch.setattr(socket, "create_connection", connect)
    tls = Mock()
    context = Mock()
    context.wrap_socket.return_value = tls
    monkeypatch.setattr(ssl, "create_default_context", lambda: context)
    connection = Mock()
    connection.getresponse.return_value = Mock(status=200)
    monkeypatch.setattr(http.client, "HTTPSConnection", lambda *a, **kw: connection)
    response = PinnedHTTPSTransport().open(
        "https://media.example.org/test.webm", URLPolicy(["media.example.org"]), 2
    )
    response.close()
    assert connect.call_args.args[0] == ("1.1.1.1", 443)
    assert context.wrap_socket.call_args.kwargs["server_hostname"] == "media.example.org"
    assert resolver.call_count == 1
    tls.do_handshake.assert_called_once()
