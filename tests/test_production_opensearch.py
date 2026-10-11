"""Fault-injected HTTP tests. Vectors and model provenance are synthetic fixtures."""

import json
from unittest.mock import Mock

import httpx
import numpy as np
import pytest

from search_agent.config import QWEN_ID, Settings
from search_agent.domain import Conditions
from search_agent.media.config import MediaSettings, OpenSearchSettings
from search_agent.media.domain import MediaEntry, MediaManifest, MediaRecord
from search_agent.media.ingest import Ingestor
from search_agent.media.opensearch import OpenSearchError, OpenSearchStore, index_mapping
from search_agent.models import normalize

IDENTITY = {"model": QWEN_ID, "dimensions": 2560, "resolved_revision": "synthetic-fixture"}


@pytest.fixture
def records():
    return [
        MediaRecord(
            canonical_url=f"https://example.org/{i}",
            original_url="https://example.org/media",
            title=f"Synthetic {i}",
            description="Test metadata only",
            author="Fixture",
            license="CC0-1.0",
            license_url="https://example.org/license",
            attribution="Fixture",
            rights_verified=True,
            metadata_provenance={
                f: {"method": "editorial", "note": "Fixture"}
                for f in ("title", "description", "tags")
            },
            retrieved_at="2026-10-11T00:00:00Z",
        )
        for i in range(3)
    ]


def store_for(handler, **kwargs):
    return OpenSearchStore(
        OpenSearchSettings(_env_file=None, retry_base_seconds=0, **kwargs),
        httpx.Client(base_url="https://localhost:9200", transport=httpx.MockTransport(handler)),
    )


def test_bulk_retries_only_failed_transient_items(records):
    calls = []

    def handler(request):
        assert request.headers["content-type"] == "application/x-ndjson"
        assert request.content.endswith(b"\n")
        lines = [json.loads(line) for line in request.content.splitlines()]
        ids = [line["index"]["_id"] for line in lines[::2]]
        calls.append(ids)
        statuses = [201, 429, 400] if len(calls) == 1 else [200]
        return httpx.Response(
            200,
            json={
                "errors": len(calls) == 1,
                "items": [
                    {
                        "index": {
                            "_id": key,
                            "status": status,
                            **(
                                {"error": {"reason": "secret source text"}} if status >= 400 else {}
                            ),
                        }
                    }
                    for key, status in zip(ids, statuses, strict=True)
                ],
            },
        )

    store = store_for(handler)
    result = store.bulk_upsert([(r, np.ones(2560)) for r in records])
    assert calls == [[r.content_id for r in records], [records[1].content_id]]
    assert [r["status"] for r in result] == ["indexed", "indexed", "failed"]
    assert [r["attempts"] for r in result] == [1, 2, 1]
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize("failure", ["timeout", "malformed", "429", "503", "401", "413"])
def test_bulk_bounded_request_failures(records, failure):
    calls = []

    def handler(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("secret", request=request)
        if failure == "malformed":
            return httpx.Response(200, json={"items": []})
        return httpx.Response(int(failure), json={"error": "secret"})

    result = store_for(handler, bulk_retries=2).bulk_upsert([(records[0], np.ones(2560))])
    assert result[0]["status"] == "failed"
    assert len(calls) == (1 if failure in {"401", "413"} else 3)
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize("vector", [np.ones(32), np.zeros(2560), np.full(2560, np.nan)])
def test_bad_vectors_never_reach_network(records, vector):
    handler = Mock(side_effect=AssertionError("No network allowed"))
    with pytest.raises(ValueError):
        store_for(handler).bulk_upsert([(records[0], vector)])
    handler.assert_not_called()


def test_large_finite_vectors_normalize_safely():
    vector = normalize([np.full(2560, 1e30)], 1, 2560)
    assert np.linalg.norm(vector) == pytest.approx(1)


def test_checkpoint_resume_bound_to_metadata_model_and_index_uuid(records, tmp_path):
    state = {"uuid": "one", "writes": 0}
    store = store_for(lambda r: httpx.Response(500))
    store.ensure_index = Mock()
    store.checkpoint_target = lambda: {"uuid": state["uuid"], "index": store.index}

    def bulk(items):
        state["writes"] += len(items)
        return [{"status": "indexed", "content_id": r.content_id} for r, _ in items]

    store.bulk_upsert = bulk
    embedding = Mock(identity=IDENTITY)
    embedding.documents.side_effect = lambda texts: np.ones((len(texts), 2560))
    entry = MediaEntry.model_validate(
        records[0].model_dump(
            exclude={"retrieved_at", "checksum_sha256", "local_path", "size_bytes"}
        )
    )
    manifest = MediaManifest(version=1, entries=[entry])
    ingest = Ingestor(MediaSettings(root=tmp_path), embedding=embedding, index=store)
    assert ingest.run(manifest, dry_run=False, resume=True)["resumed"] == 0
    assert ingest.run(manifest, dry_run=False, resume=True)["resumed"] == 1
    assert state["writes"] == 1
    entry.title = "Changed metadata"
    assert ingest.run(manifest, dry_run=False, resume=True)["resumed"] == 0
    state["uuid"] = "recreated"
    assert ingest.run(manifest, dry_run=False, resume=True)["resumed"] == 0
    embedding.identity = IDENTITY | {"resolved_revision": "another-fixture"}
    assert ingest.run(manifest, dry_run=False, resume=True)["resumed"] == 0
    assert state["writes"] == 4


def test_failed_checkpoint_is_retried(records, tmp_path):
    store = store_for(lambda r: httpx.Response(500))
    store.ensure_index = Mock()
    store.checkpoint_target = lambda: {"uuid": "one"}
    store.bulk_upsert = Mock(side_effect=[[{"status": "failed"}], [{"status": "indexed"}]])
    embedding = Mock(identity=IDENTITY)
    embedding.documents.return_value = np.ones((1, 2560))
    entry = MediaEntry.model_validate(
        records[0].model_dump(
            exclude={"retrieved_at", "checksum_sha256", "local_path", "size_bytes"}
        )
    )
    manifest = MediaManifest(version=1, entries=[entry])
    ingestor = Ingestor(MediaSettings(root=tmp_path), embedding=embedding, index=store)
    assert ingestor.run(manifest, dry_run=False, resume=True)["status"] == "partial_failure"
    assert ingestor.run(manifest, dry_run=False, resume=True)["indexed"] == 1
    assert store.bulk_upsert.call_count == 2


@pytest.mark.parametrize("bad", [{"mode": "mock"}, {}, {"embedding": {"revision": "main"}}])
def test_production_requires_real_pinned_models(bad):
    with pytest.raises(ValueError):
        Settings(_env_file=None, environment="production", **bad)


def test_production_security_config():
    with pytest.raises(ValueError):
        OpenSearchSettings(_env_file=None, production=True)
    with pytest.raises(ValueError):
        OpenSearchSettings(
            _env_file=None,
            production=True,
            username="fixture",
            password="fixture",
            url="http://localhost:9200",
            allow_http_local=True,
            replicas=1,
        )
    config = OpenSearchSettings(
        _env_file=None,
        production=True,
        username="fixture",
        password="fixture",
        replicas=1,
        url="https://localhost:9200",
        allow_http_local=False,
    )
    assert "fixture" not in repr(config.password)


def test_alias_mapping_and_snapshot():
    mapping = {"kids-media-v1-build1": index_mapping(IDENTITY)}
    store = store_for(
        lambda r: httpx.Response(
            200,
            json=(
                {"kids-media-v1-build1": {"aliases": {"kids-media-read": {}}}}
                if r.url.path.startswith("/_alias/")
                else mapping
            ),
        ),
        index="kids-media-read",
    )
    store.verify_identity(IDENTITY)
    assert store.snapshot().index == "kids-media-v1-build1"
    with pytest.raises(OpenSearchError):
        store.validate_mapping(mapping | {"other": index_mapping(IDENTITY)}, IDENTITY)
    mapping["kids-media-v1-build1"]["mappings"]["properties"]["tags"] = {"type": "text"}
    with pytest.raises(OpenSearchError):
        store.validate_mapping(mapping, IDENTITY)


@pytest.mark.parametrize("operation", ["count", "vocabulary"])
def test_partial_responses_rejected(operation):
    store = store_for(lambda r: httpx.Response(200, json={"_shards": {"failed": 1}, "count": 0}))
    with pytest.raises(OpenSearchError):
        store.count(Conditions()) if operation == "count" else store.vocabulary()


@pytest.mark.parametrize("failure", ["stale", "count", "health", "identity", "ack"])
def test_publication_failures_do_not_switch_alias(failure):
    calls = []

    def handler(request):
        calls.append(request)
        path = request.url.path
        if path == "/_alias/kids-media-read":
            return httpx.Response(
                200,
                json={
                    "old" if failure != "stale" else "unexpected": {
                        "aliases": {"kids-media-read": {}}
                    }
                },
            )
        if path.endswith("/_mapping"):
            identity = (
                IDENTITY if failure != "identity" else IDENTITY | {"resolved_revision": "bad"}
            )
            return httpx.Response(200, json={"kids-media-v1-build1": index_mapping(identity)})
        if path.startswith("/_cluster/health"):
            return httpx.Response(
                200, json={"status": "yellow" if failure == "health" else "green"}
            )
        if path.endswith("/_block/write"):
            return httpx.Response(
                200,
                json={
                    "acknowledged": failure != "ack",
                    "shards_acknowledged": True,
                    "indices": [{"name": "kids-media-v1-build1", "blocked": True}],
                },
            )
        if path.endswith("/_refresh"):
            return httpx.Response(200, json={"_shards": {"successful": 1, "failed": 0}})
        if path.endswith("/_count"):
            return httpx.Response(200, json={"count": 0})
        raise AssertionError(path)

    with pytest.raises(OpenSearchError):
        store_for(handler, index="kids-media-v1-build1").publish(
            "kids-media-read", IDENTITY, expected_current="old", expected_count=1
        )
    assert all(r.url.path != "/_aliases" and r.method != "DELETE" for r in calls)


def test_bulk_limits_fail_before_network(records):
    handler = Mock(side_effect=AssertionError("No network allowed"))
    with pytest.raises(ValueError, match="item limit"):
        store_for(handler, bulk_size=1).bulk_upsert([(r, np.ones(2560)) for r in records])
    large = records[0].model_copy(update={"description": "x" * 120_000})
    with pytest.raises(ValueError, match="byte limit"):
        store_for(handler, bulk_max_bytes=100_000).bulk_upsert([(large, np.ones(2560))])
    handler.assert_not_called()


@pytest.mark.parametrize("corrupt", ["broken json", "[]", '{"status":"failed"}'])
def test_corrupt_checkpoint_never_skips_write(records, tmp_path, corrupt):
    store = store_for(lambda r: httpx.Response(500))
    store.ensure_index = Mock()
    store.checkpoint_target = lambda: {"uuid": "one"}
    store.bulk_upsert = Mock(return_value=[{"status": "indexed"}])
    embedding = Mock(identity=IDENTITY)
    embedding.documents.return_value = np.ones((1, 2560))
    entry = MediaEntry.model_validate(
        records[0].model_dump(
            exclude={"retrieved_at", "checksum_sha256", "local_path", "size_bytes"}
        )
    )
    manifest = MediaManifest(version=1, entries=[entry])
    ingestor = Ingestor(MediaSettings(root=tmp_path), embedding=embedding, index=store)
    ingestor.run(manifest, dry_run=False)
    checkpoint = next((tmp_path / "checkpoints").glob("*/*.json"))
    checkpoint.write_text(corrupt)
    assert ingestor.run(manifest, dry_run=False, resume=True)["indexed"] == 1
    assert store.bulk_upsert.call_count == 2


def test_production_never_creates_a_read_alias_as_index():
    handler = Mock(side_effect=AssertionError("No network allowed"))
    store = store_for(
        handler,
        production=True,
        username="fixture",
        password="fixture",
        replicas=1,
        url="https://localhost:9200",
        allow_http_local=False,
        index="kids-media-read",
    )
    with pytest.raises(OpenSearchError, match="versioned concrete"):
        store.ensure_index(IDENTITY)
    handler.assert_not_called()


def test_pinning_rejects_filtered_alias_instead_of_bypassing_filter():
    def handler(request):
        body = (
            {
                "kids-media-v1-build1": {
                    "aliases": {"kids-media-read": {"filter": {"term": {"tenant": "a"}}}}
                }
            }
            if request.url.path.startswith("/_alias/")
            else {"kids-media-v1-build1": index_mapping(IDENTITY)}
        )
        return httpx.Response(200, json=body)

    with pytest.raises(OpenSearchError, match="Filtered/routed"):
        store_for(handler, index="kids-media-read").snapshot()


@pytest.mark.parametrize(
    "block",
    [
        {"acknowledged": True},
        {"acknowledged": True, "shards_acknowledged": False},
        {
            "acknowledged": True,
            "shards_acknowledged": False,
            "indices": [{"name": "kids-media-v1-build1", "blocked": True}],
        },
        {"acknowledged": True, "shards_acknowledged": True, "indices": []},
        {
            "acknowledged": True,
            "shards_acknowledged": True,
            "indices": [{"name": "kids-media-v1-build1", "blocked": False}],
        },
        {
            "acknowledged": True,
            "shards_acknowledged": True,
            "indices": [{"name": "wrong-index", "blocked": True}],
        },
        {
            "acknowledged": True,
            "shards_acknowledged": True,
            "indices": [{"name": "kids-media-v1-build1", "blocked": True, "exception": "secret"}],
        },
    ],
)
def test_incomplete_write_block_never_publishes(block):
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path.startswith("/_alias/"):
            return httpx.Response(404, json={})
        if request.url.path.endswith("/_mapping"):
            return httpx.Response(200, json={"kids-media-v1-build1": index_mapping(IDENTITY)})
        if request.url.path.startswith("/_cluster/health"):
            return httpx.Response(200, json={"status": "green"})
        if request.url.path.endswith("/_block/write"):
            return httpx.Response(200, json=block)
        raise AssertionError("No further request allowed after incomplete block")

    with pytest.raises(OpenSearchError, match="not fully acknowledged"):
        store_for(handler, index="kids-media-v1-build1").publish(
            "kids-media-read", IDENTITY, expected_current=None, expected_count=1
        )
    assert len(calls) == 4


@pytest.mark.parametrize("option", ["filter", "routing", "search_routing", "index_routing"])
def test_publication_does_not_strip_alias_restrictions(option):
    def handler(request):
        assert request.method == "GET" and request.url.path == "/_alias/kids-media-read"
        return httpx.Response(200, json={"old": {"aliases": {"kids-media-read": {option: {}}}}})

    with pytest.raises(OpenSearchError, match="Filtered/routed"):
        store_for(handler, index="kids-media-v1-build1").publish(
            "kids-media-read", IDENTITY, expected_current="old", expected_count=1
        )


def test_crash_after_remote_write_cannot_resume_older_metadata(records, tmp_path):
    store = store_for(lambda r: httpx.Response(500))
    store.ensure_index = Mock()
    store.checkpoint_target = lambda: {"uuid": "one"}
    committed = []

    def bulk(items):
        committed.extend(r.title for r, _ in items)
        if len(committed) == 2:
            raise KeyboardInterrupt("Simulated crash after remote commit")
        return [{"status": "indexed"}]

    store.bulk_upsert = bulk
    embedding = Mock(identity=IDENTITY)
    embedding.documents.return_value = np.ones((1, 2560))
    entry = MediaEntry.model_validate(
        records[0].model_dump(
            exclude={"retrieved_at", "checksum_sha256", "local_path", "size_bytes"}
        )
    )
    manifest = MediaManifest(version=1, entries=[entry])
    ingestor = Ingestor(MediaSettings(root=tmp_path), embedding=embedding, index=store)
    original = entry.title
    ingestor.run(manifest, dry_run=False, resume=True)
    entry.title = "Newer metadata"
    with pytest.raises(KeyboardInterrupt):
        ingestor.run(manifest, dry_run=False, resume=True)
    checkpoint = next((tmp_path / "checkpoints").glob("*/*.json"))
    assert json.loads(checkpoint.read_text())["status"] == "pending"
    entry.title = original
    result = ingestor.run(manifest, dry_run=False, resume=True)
    assert result["resumed"] == 0 and result["indexed"] == 1
    assert committed == [original, "Newer metadata", original]


def test_failed_pending_checkpoint_prevents_remote_write(records, tmp_path, monkeypatch):
    store = store_for(lambda r: httpx.Response(500))
    store.ensure_index = Mock()
    store.checkpoint_target = lambda: {"uuid": "one"}
    store.bulk_upsert = Mock()
    embedding = Mock(identity=IDENTITY)
    entry = MediaEntry.model_validate(
        records[0].model_dump(
            exclude={"retrieved_at", "checksum_sha256", "local_path", "size_bytes"}
        )
    )
    monkeypatch.setattr("search_agent.media.batch.atomic_json", Mock(side_effect=OSError("full")))
    result = Ingestor(MediaSettings(root=tmp_path), embedding=embedding, index=store).run(
        MediaManifest(version=1, entries=[entry]), dry_run=False
    )
    assert result["status"] == "partial_failure"
    assert result["entries"][0]["receipt_write_failed"] is True
    store.bulk_upsert.assert_not_called()
    embedding.documents.assert_not_called()


def test_production_switches_never_silently_weaken_each_other(monkeypatch):
    from search_agent.app import create_app

    for key, value in {
        "OPENSEARCH_PRODUCTION": "true",
        "OPENSEARCH_URL": "https://localhost:9200",
        "OPENSEARCH_ALLOW_HTTP_LOCAL": "false",
        "OPENSEARCH_USERNAME": "fixture",
        "OPENSEARCH_PASSWORD": "fixture",
        "OPENSEARCH_REPLICAS": "1",
    }.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="SEARCH_ENVIRONMENT"):
        create_app(Settings(_env_file=None, backend="opensearch"))
    pinned = {"revision": "a" * 40}
    settings = Settings(_env_file=None, environment="production", embedding=pinned, reranker=pinned)
    monkeypatch.setenv("OPENSEARCH_PRODUCTION", "false")
    assert OpenSearchSettings.for_search(settings).production is True
    monkeypatch.setenv("OPENSEARCH_REPLICAS", "0")
    with pytest.raises(ValueError):
        OpenSearchSettings.for_search(settings)


def test_configuration_validation_does_not_print_secret_values():
    secret = "sensitive-fixture-secret"
    with pytest.raises(ValueError) as error:
        OpenSearchSettings(_env_file=None, url="http://remote.invalid", password=secret)
    assert secret not in str(error.value) and secret not in repr(error.value)
    with pytest.raises(ValueError) as error:
        Settings(_env_file=None, admin_api_key=secret)
    assert secret not in str(error.value) and secret not in repr(error.value)


def test_malformed_upstream_record_is_sanitized_at_public_api(records):
    from fastapi.testclient import TestClient

    from search_agent.app import create_app

    secret = "upstream-secret-sentinel"
    source = records[0].model_dump() | {
        "content_id": records[0].content_id,
        "title": {"secret": secret},
    }
    store = store_for(
        lambda r: httpx.Response(
            200, json={"hits": {"hits": [{"_id": records[0].content_id, "_source": source}]}}
        )
    )
    agent = Mock()
    agent.search.side_effect = lambda request: store.search({})
    with TestClient(create_app(Settings(mode="mock"), agent)) as client:
        response = client.post("/api/search", json={"query": "fixture"})
    assert response.status_code == 503 and secret not in response.text
    assert response.json()["detail"]["code"] == "OPENSEARCH_UNAVAILABLE"


@pytest.mark.parametrize("path", ["/api/search", "/api/admin/search"])
def test_model_failure_logs_do_not_include_upstream_secrets(caplog, path):
    from fastapi.testclient import TestClient

    from search_agent.app import create_app
    from search_agent.models import ModelError

    agent = Mock()
    agent.search.side_effect = ModelError("embedding", "sensitive-fixture-secret")
    agent._search.side_effect = ModelError("embedding", "sensitive-fixture-secret")
    key = "ci-only-admin-browser-key-never-use-in-production"
    with TestClient(create_app(Settings(mode="mock", admin_api_key=key), agent)) as client:
        response = client.post(
            path, json={"query": "fixture"}, headers={"Authorization": f"Bearer {key}"}
        )
    assert response.status_code == 503
    assert "sensitive-fixture-secret" not in response.text + caplog.text


def test_ensure_index_rejects_alias_before_additive_mapping_write():
    mapping = index_mapping(IDENTITY)
    del mapping["mappings"]["properties"]["source_id"]

    def handler(request):
        assert request.method == "GET"
        if request.url.path == "/":
            return httpx.Response(200, json={"version": {"number": "2.19.3"}})
        return httpx.Response(200, json={"actual-v1-build": mapping})

    with pytest.raises(OpenSearchError, match="concrete index"):
        store_for(handler, index="alias-v1-build").ensure_index(IDENTITY)


@pytest.mark.parametrize("setting", [False, "false", None, True, "true"])
def test_exact_completed_block_noop_requires_fresh_concrete_write_block(setting):
    published = []

    def handler(request):
        path = request.url.path
        if path == "/_alias/kids-media-read":
            return (
                httpx.Response(200, json={"kids-media-v1-build1": {}})
                if published
                else httpx.Response(404, json={})
            )
        if path.endswith("/_mapping"):
            return httpx.Response(200, json={"kids-media-v1-build1": index_mapping(IDENTITY)})
        if path.startswith("/_cluster/health"):
            return httpx.Response(200, json={"status": "green"})
        if path.endswith("/_block/write"):
            return httpx.Response(
                200, json={"acknowledged": True, "shards_acknowledged": False, "indices": []}
            )
        if path.endswith("/_settings"):
            return httpx.Response(
                200,
                json={
                    "kids-media-v1-build1": {
                        "settings": {
                            "index": {"blocks": {"write": setting} if setting is not None else {}}
                        }
                    }
                },
            )
        if path.endswith("/_refresh"):
            return httpx.Response(200, json={"_shards": {"successful": 1, "failed": 0}})
        if path.endswith("/_count"):
            return httpx.Response(200, json={"count": 1})
        if path == "/_aliases":
            published.append(json.loads(request.content))
            return httpx.Response(200, json={"acknowledged": True})
        raise AssertionError(path)

    store = store_for(handler, index="kids-media-v1-build1")
    if setting in (True, "true"):
        assert (
            store.publish("kids-media-read", IDENTITY, expected_current=None, expected_count=1)[
                "status"
            ]
            == "published"
        )
    else:
        with pytest.raises(OpenSearchError, match="not fully acknowledged"):
            store.publish("kids-media-read", IDENTITY, expected_current=None, expected_count=1)
        assert not published
