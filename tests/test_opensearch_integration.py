"""Opt-in live OpenSearch test. Synthetic vectors/metadata ONLY; no Qwen/BGE inference."""

import os
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from search_agent.app import create_app
from search_agent.config import QWEN_ID, Settings
from search_agent.domain import Conditions
from search_agent.media.config import MediaSettings, OpenSearchSettings
from search_agent.media.domain import MediaEntry, MediaManifest, MediaRecord
from search_agent.media.ingest import Ingestor
from search_agent.media.opensearch import OpenSearchError, OpenSearchStore, bm25_query, vector_query
from search_agent.media.service import MediaSearchAgent

pytestmark = pytest.mark.skipif(
    os.getenv("OPENSEARCH_TEST") != "1", reason="Set OPENSEARCH_TEST=1 for a live server"
)


def test_live_2560_mapping_filters_upsert_and_identity(tmp_path, monkeypatch):
    config = OpenSearchSettings().model_copy(
        update={"index": "search-agent-test-v1-" + uuid.uuid4().hex}
    )
    store = OpenSearchStore(config)
    identity = {
        "model": QWEN_ID,
        "dimensions": 2560,
        "revision": "fixture-NOT-real-inference",
        "resolved_revision": "fixture-NOT-real-inference",
    }
    vector = np.zeros(2560)
    vector[0] = 1
    provenance = {
        field: {"method": "editorial", "note": "Integration test fixture, not real media"}
        for field in ["title", "description", "tags", "language", "duration_seconds"]
    }
    ids = []
    next_store = OpenSearchStore(config.model_copy(update={"index": config.index + "-next"}))
    alias = config.index + "-read"
    try:
        store.ensure_index(identity)
        store.ensure_index(identity)
        for name, ages, tags in [
            ("known-age", (3, 6), ["펭귄"]),
            ("older", (8, 12), ["펭귄"]),
            ("unknown", (None, None), ["펭귄"]),
            ("other", (3, 6), ["우주"]),
        ]:
            record = MediaRecord(
                canonical_url=f"https://example.org/integration/{name}",
                original_url="https://example.org/synthetic.webm",
                title=f"펭귄 fixture {name}",
                description="Synthetic test metadata",
                tags=tags,
                language="und",
                media_format="webm",
                min_age=ages[0],
                max_age=ages[1],
                age_rating_source="https://example.org/fixture-rating" if ages[0] else None,
                author="Test",
                license="CC0-1.0",
                license_url="https://example.org/test-license",
                attribution="Synthetic integration fixture",
                rights_verified=True,
                metadata_provenance=provenance,
                source_id=f"fixture:{name}",
                retrieved_at="2026-10-03T00:00:00Z",
            )
            ids.append(record.content_id)
            store.upsert(record, vector)
            store.upsert(record, vector)

        # Exercise the default metadata-only CLI service against the live server, too.
        class FixtureEmbedding:
            @property
            def identity(self):
                return identity

            def documents(self, texts):
                return np.tile(vector, (len(texts), 1))

            def query(self, text):
                return vector

        class ForbiddenMediaStorage:
            def download(self, *args, **kwargs):
                raise AssertionError("No media download allowed")

        entry = MediaEntry.model_validate(
            record.model_dump(
                exclude={"retrieved_at", "local_path", "size_bytes", "checksum_sha256"}
            )
        )
        ingestor = Ingestor(
            MediaSettings(root=tmp_path), ForbiddenMediaStorage(), FixtureEmbedding(), store
        )
        for _ in range(2):
            result = ingestor.run(MediaManifest(version=1, entries=[entry]), dry_run=False)
            assert result["indexed"] == 1 and result["media_download_performed"] is False
        assert store.count(Conditions()) == 4  # Repeated PUT does not create duplicates.
        store.verify_identity(identity)
        assert "펭귄" in store.vocabulary()[0]
        c = Conditions(age=5, topics=["펭귄"])
        assert store.count(c) == 1
        assert [r.content_id for r in store.search(bm25_query("펭귄", c, 10))] == [ids[0]]
        assert [r.content_id for r in store.search(vector_query(vector, c, 10))] == [ids[0]]
        assert store.count(Conditions(age=18)) == 0
        assert not store.search(vector_query(vector, Conditions(age=18), 1))
        all_penguins = store.search(vector_query(vector, Conditions(topics=["펭귄"]), 10))
        assert len(all_penguins) == 3
        assert all(
            r.local_path is None and r.checksum_sha256 is None and r.size_bytes is None
            for r in all_penguins
        )
        assert {r.source_id for r in all_penguins} == {
            "fixture:known-age",
            "fixture:older",
            "fixture:unknown",
        }
        assert any(r.min_age is None for r in all_penguins)

        # Exercise the admin HTTP boundary against this actual temporary index.
        # Embedding/reranking are still synthetic fixtures, never Qwen/BGE inference.
        class FixtureReranker:
            def score(self, query, passages):
                return [0.5] * len(passages)

        monkeypatch.setenv("MEDIA_ROOT", str(tmp_path / "admin-media"))
        admin_key = "integration-only-admin-key-never-use-in-production"
        settings = Settings(mode="real", backend="opensearch", admin_api_key=admin_key)
        agent = MediaSearchAgent(settings, store, FixtureEmbedding(), FixtureReranker())
        with TestClient(create_app(settings, agent)) as client:
            assert client.get("/api/admin/status").status_code == 401
            client.headers["Authorization"] = f"Bearer {admin_key}"
            status = client.get("/api/admin/status")
            assert status.status_code == 200
            assert status.json()["content_count"] == 4
            assert status.json()["index"]["state"] == "available"
            assert status.json()["models"]["embedding"]["state"] == "adapter_unverified"
            page = client.get("/api/admin/contents", params={"limit": 2})
            assert page.status_code == 200
            assert page.json()["total"] == 4 and len(page.json()["items"]) == 2
            assert page.json()["has_more"] is True
            assert all("local_path" not in item for item in page.json()["items"])
            preview = client.post(
                "/api/admin/ingest",
                json={"manifest": {"version": 1, "entries": [entry.model_dump()]}},
            )
            assert preview.status_code == 200 and preview.json()["dry_run"] is True
            ingested = client.post(
                "/api/admin/ingest",
                json={
                    "manifest": {"version": 1, "entries": [entry.model_dump()]},
                    "dry_run": False,
                    "confirmed": True,
                },
            )
            assert ingested.status_code == 200 and ingested.json()["indexed"] == 1
            assert ingested.json()["media_download_performed"] is False
            assert (
                client.post("/api/admin/index/ensure", json={"confirmed": True}).status_code == 200
            )
            assert (
                client.post("/api/admin/index/refresh", json={"confirmed": True}).status_code == 200
            )
            found = client.post("/api/admin/search", json={"query": "5살 펭귄 영상"})
            assert found.status_code == 200
            assert [item["content_id"] for item in found.json()["results"]] == [ids[0]]
            assert store.count(Conditions()) == 4
        # Bulk checkpoints and alias promotion/rollback against a real engine.
        resumed = ingestor.run(
            MediaManifest(version=1, entries=[entry]), dry_run=False, resume=True
        )
        assert resumed["resumed"] == 1
        next_store.ensure_index(identity)
        outcomes = next_store.bulk_upsert([(record, vector)])
        assert outcomes[0]["status"] == "indexed"
        assert next_store.count(Conditions()) == 1
        first = store.publish(alias, identity, expected_current=None, expected_count=4)
        assert first["previous_index"] is None
        reader = OpenSearchStore(config.model_copy(update={"index": alias}))
        try:
            reader.verify_identity(identity)
            assert reader.count(Conditions()) == 4
            pinned = reader.snapshot()
            second = next_store.publish(
                alias, identity, expected_current=store.index, expected_count=1
            )
            assert second["previous_index"] == store.index
            assert reader.count(Conditions()) == 1
            assert pinned.count(Conditions()) == 4
            with pytest.raises(OpenSearchError):
                store.publish(alias, identity, expected_current=store.index, expected_count=4)
            # Rollback uses the same validation and retains both immutable versions.
            store.publish(alias, identity, expected_current=next_store.index, expected_count=4)
            assert reader.count(Conditions()) == 4
            assert next_store.count(Conditions()) == 1
            assert store.bulk_upsert([(record, vector)])[0]["status"] == "failed"
            with TestClient(
                create_app(
                    settings,
                    MediaSearchAgent(settings, reader, FixtureEmbedding(), FixtureReranker()),
                )
            ) as client:
                client.headers["Authorization"] = f"Bearer {admin_key}"
                assert client.get("/api/admin/status").json()["index"]["state"] == "available"
                found = client.post("/api/search", json={"query": "5살 펭귄 영상"})
                assert found.status_code == 200
                assert [item["content_id"] for item in found.json()["results"]] == [ids[0]]
        finally:
            reader.close()
    finally:
        # Only unique test-owned indices created by this test are removed.
        next_store.request("DELETE", f"/{next_store.index}", allow_not_found=True)
        next_store.close()
        store.request("DELETE", f"/{config.index}", allow_not_found=True)
        store.close()


def test_live_stale_alias_swap_after_preflight_is_atomic():
    """Another publisher wins after our GET: must_exist must abort the whole stale POST."""
    store = OpenSearchStore(OpenSearchSettings())
    prefix = "search-agent-race-test-" + uuid.uuid4().hex
    old, winner, stale = [prefix + suffix for suffix in ("-old", "-winner", "-stale")]
    alias = prefix + "-read"
    created = []
    try:
        for name in (old, winner, stale):
            store.request("PUT", f"/{name}", body={"settings": {"number_of_replicas": 0}})
            created.append(name)
        store.request(
            "POST",
            "/_aliases",
            body={"actions": [{"add": {"index": old, "alias": alias, "is_write_index": False}}]},
        )
        assert set(store.request("GET", f"/_alias/{alias}")) == {old}
        # Competing serialized update reaches the server between preflight and stale POST.
        store.request(
            "POST",
            "/_aliases",
            body={
                "actions": [
                    {"remove": {"index": old, "alias": alias, "must_exist": True}},
                    {"add": {"index": winner, "alias": alias, "is_write_index": False}},
                ]
            },
        )
        with pytest.raises(OpenSearchError):
            store.request(
                "POST",
                "/_aliases",
                body={
                    "actions": [
                        {"remove": {"index": old, "alias": alias, "must_exist": True}},
                        {"add": {"index": stale, "alias": alias, "is_write_index": False}},
                    ]
                },
            )
        assert set(store.request("GET", f"/_alias/{alias}")) == {winner}
    finally:
        for name in created:
            store.request("DELETE", f"/{name}", allow_not_found=True)
        store.close()
