"""Opt-in live OpenSearch test. Synthetic vectors/metadata ONLY; no Qwen/BGE inference."""

import os
import uuid

import numpy as np
import pytest

from search_agent.config import QWEN_ID
from search_agent.domain import Conditions
from search_agent.media.config import OpenSearchSettings
from search_agent.media.domain import MediaRecord
from search_agent.media.opensearch import OpenSearchStore, bm25_query, vector_query

pytestmark = pytest.mark.skipif(
    os.getenv("OPENSEARCH_TEST") != "1", reason="Set OPENSEARCH_TEST=1 for a live server"
)


def test_live_2560_mapping_filters_upsert_and_identity():
    config = OpenSearchSettings().model_copy(
        update={"index": "search-agent-test-" + uuid.uuid4().hex}
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
                checksum_sha256="a" * 64,
                local_path="synthetic-not-a-real-file.webm",
                size_bytes=1,
                retrieved_at="2026-10-03T00:00:00Z",
            )
            ids.append(record.content_id)
            store.upsert(record, vector)
            store.upsert(record, vector)
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
        assert any(r.min_age is None for r in all_penguins)
    finally:
        store.request("DELETE", f"/{config.index}", allow_not_found=True)
        store.close()
