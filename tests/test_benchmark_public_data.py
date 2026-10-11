"""Synthetic tests for benchmark arithmetic and public metadata licensing boundaries."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from search_agent.benchmark import quality, run_queries, summarize
from search_agent.media.config import MediaSettings
from search_agent.media.download import LocalMediaStore
from search_agent.media.ingest import Ingestor
from search_agent.media.local_import import import_local
from search_agent.media.wikidata import convert, sparql


def binding(qid="Q123", title="테스트 영화"):
    return {
        "item": {"value": "http://www.wikidata.org/entity/" + qid},
        "itemLabel": {"value": title, "xml:lang": "ko"},
    }


def test_public_metadata_is_deduplicated_and_never_claims_film_rights(tmp_path):
    manifest = convert([binding(), binding()], "2026-10-11T00:00:00Z", 100)
    assert len(manifest.entries) == 1
    entry = manifest.entries[0]
    assert entry.source_id == "wikidata:Q123"
    assert entry.rights_scope == "metadata" and entry.metadata_license == "CC0-1.0"
    assert entry.original_url == entry.canonical_url
    assert entry.min_age is None and entry.media_format is None
    assert entry.source_metadata["media_rights_verified"] is False
    assert "줄거리 미수집" in entry.description
    assert Ingestor(MediaSettings(root=tmp_path)).run(manifest)["media_download_performed"] is False
    with pytest.raises(ValueError, match="does not authorize"):
        Ingestor(MediaSettings(root=tmp_path)).run(manifest, download_only=True)
    with pytest.raises(ValueError, match="does not authorize"):
        LocalMediaStore(MediaSettings(root=tmp_path)).download(entry)
    with pytest.raises(ValueError, match="does not authorize"):
        import_local(entry, tmp_path / "absent.mp4", MediaSettings(root=tmp_path))


@pytest.mark.parametrize("limit", [0, 501, -1])
def test_public_collection_limit(limit):
    with pytest.raises(ValueError):
        sparql(limit)


def test_public_collection_rejects_external_entity_url():
    data = binding()
    data["item"]["value"] = "https://evil.invalid/Q123"
    with pytest.raises(ValueError):
        convert([data], "2026-10-11T00:00:00Z", 100)


def test_benchmark_statistics_include_errors_without_secret_text():
    agent = Mock()
    agent.search.side_effect = [
        SimpleNamespace(
            model_inference_performed=True, results=[SimpleNamespace(content_id="doc")]
        ),
        RuntimeError("secret"),
    ]
    result = run_queries(agent, [{"id": "a", "query": "fixture"}], concurrency=1, repeats=2)
    assert result["requests"] == 2 and result["failure_rate"] == 0.5
    assert result["successful_requests_per_second"] > 0
    assert result["model_inference_requests"] == 1
    assert "secret" not in str(result)
    assert set(result["latency_ms_successes"]) == {"p50", "p95", "p99"}
    assert result["resources"]["peak_rss_bytes"] > 0
    assert summarize([], 0)["latency_ms_successes"]["p95"] is None


def test_quality_requires_reviewed_version_bound_judgments():
    samples = [{"query_id": "q1", "ok": True, "ids": ["good", "bad"]}]
    assert quality(samples, None, "hash", "uuid", 2)["status"] == "not_evaluated"
    qrels = {
        "reviewed": True,
        "queries_sha256": "hash",
        "index_uuid": "uuid",
        "source": "synthetic test only",
        "judgments": {"q1": {"good": 3, "bad": 0}},
    }
    assert quality(samples, qrels, "hash", "uuid", 2)["mean_ndcg"] == 1
    with pytest.raises(ValueError):
        quality(samples, qrels, "wrong", "uuid", 2)
    with pytest.raises(ValueError):
        quality(samples, qrels | {"reviewed": False}, "hash", "uuid", 2)
