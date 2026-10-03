"""Metadata-only ingestion never invokes media networking or requires media bytes."""

import json
import socket
import sys
from pathlib import Path
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
from pydantic import ValidationError

from search_agent.config import QWEN_ID
from search_agent.media import cli
from search_agent.media.config import MediaSettings, OpenSearchSettings
from search_agent.media.domain import MediaEntry, MediaManifest, MediaRecord
from search_agent.media.ingest import Ingestor, load_manifest
from search_agent.media.opensearch import OpenSearchError, OpenSearchStore, index_mapping


@pytest.fixture
def metadata():
    return MediaEntry(
        source_id="fixture:metadata",
        canonical_url="https://example.org/source",
        original_url="https://example.org/watch?id=123",
        title="메타데이터 테스트",
        description="실제 미디어 바이트가 없는 테스트 자료",
        tags=["동물"],
        author="Fixture",
        license="CC0-1.0",
        license_url="https://example.org/license",
        attribution="Fixture metadata only",
        rights_verified=True,
        metadata_provenance={
            field: {"method": "editorial", "note": "Synthetic fixture"}
            for field in ["title", "description", "tags"]
        },
    )


class Embedding:
    @property
    def identity(self):
        return {
            "model": QWEN_ID,
            "dimensions": 2560,
            "revision": "test-fixture",
            "resolved_revision": "test-fixture",
        }

    def documents(self, texts):
        return np.ones((len(texts), 2560))

    def query(self, text):
        return np.ones(2560)


def test_default_metadata_only_no_media_network_and_preserves_files(
    metadata, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Metadata-only must not touch media networking/storage")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr("search_agent.media.ingest.LocalMediaStore", forbidden)
    existing = tmp_path / "already-present.webm"
    existing.write_bytes(b"existing file must remain unchanged")
    old_receipt = tmp_path / f"{metadata.content_id}.receipt.json"
    old_receipt.write_text('{"status":"downloaded"}')
    index = Mock()
    ingestor = Ingestor(
        MediaSettings(root=tmp_path, allowed_domains=[]), embedding=Embedding(), index=index
    )
    manifest = MediaManifest(version=1, entries=[metadata, metadata])
    assert ingestor.run(manifest)["media_download_performed"] is False
    first = ingestor.run(manifest, dry_run=False)
    second = ingestor.run(manifest, dry_run=False)
    assert first["indexed"] == second["indexed"] == 1
    assert first["bytes_accounted"] == second["bytes_accounted"] == 0
    record = index.upsert.call_args.args[0]
    assert record.source_id == "fixture:metadata" and record.content_id == metadata.content_id
    assert (
        record.local_path is None and record.checksum_sha256 is None and record.size_bytes is None
    )
    assert record.media_format is None and record.min_age is None and record.subtitle_text is None
    assert record.license == metadata.license and record.attribution == metadata.attribution
    assert record.metadata_provenance == metadata.metadata_provenance
    assert existing.read_bytes() == b"existing file must remain unchanged"
    assert old_receipt.read_text() == '{"status":"downloaded"}'
    assert index.upsert.call_count == 2


def test_metadata_retry_and_optional_fields(metadata, tmp_path):
    record = MediaRecord(**metadata.model_dump(), retrieved_at="2026-10-03T00:00:00Z")
    assert (
        record.model_dump(exclude_none=True)
        .keys()
        .isdisjoint({"local_path", "checksum_sha256", "size_bytes"})
    )
    index = Mock()
    index.upsert.side_effect = [OpenSearchError("offline"), {}]
    ingestor = Ingestor(MediaSettings(root=tmp_path), embedding=Embedding(), index=index)
    manifest = MediaManifest(version=1, entries=[metadata])
    failed = ingestor.run(manifest, dry_run=False)
    assert failed["status"] == "partial_failure" and failed["entries"][0]["stage"] == "index_upsert"
    assert ingestor.run(manifest, dry_run=False)["indexed"] == 1


@pytest.mark.parametrize(
    "field",
    ["title", "description", "canonical_url", "original_url", "license", "author", "attribution"],
)
def test_required_metadata_validation(metadata, field):
    raw = metadata.model_dump()
    del raw[field]
    with pytest.raises(ValidationError):
        MediaEntry.model_validate(raw)


def test_existing_manifest_source_id_fallback(metadata):
    raw = metadata.model_dump()
    del raw["source_id"]
    assert MediaEntry.model_validate(raw).source_id == metadata.canonical_url
    raw["source_metadata"] = {"source_id": "commons:existing"}
    assert MediaEntry.model_validate(raw).source_id == "commons:existing"
    with pytest.raises(ValidationError):
        MediaEntry.model_validate(raw | {"source_id": ""})


def test_metadata_upsert_omits_bytes_and_adds_mapping(metadata):
    identity = Embedding().identity
    old_mapping = index_mapping(identity)
    del old_mapping["mappings"]["properties"]["source_id"]
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path == "/":
            return httpx.Response(200, json={"version": {"number": "2.19.3"}})
        if request.method == "GET":
            return httpx.Response(200, json={"kids-media-v1": old_mapping})
        return httpx.Response(200, json={"result": "updated"})

    store = OpenSearchStore(
        OpenSearchSettings(),
        httpx.Client(base_url="https://localhost:9200", transport=httpx.MockTransport(handler)),
    )
    store.ensure_index(identity)
    patch = next(r for r in seen if r.method == "PUT")
    assert json.loads(patch.content) == {"properties": {"source_id": {"type": "keyword"}}}
    record = MediaRecord(**metadata.model_dump(), retrieved_at="2026-10-03T00:00:00Z")
    store.upsert(record, np.ones(2560))
    store.upsert(record, np.ones(2560))
    assert seen[-1].url == seen[-2].url
    body = json.loads(seen[-1].content)
    assert body["source_id"] == "fixture:metadata"
    assert not {"local_path", "checksum_sha256", "size_bytes"} & body.keys()


def test_cli_execute_default_metadata_only(metadata, tmp_path, monkeypatch, capsys):
    path = tmp_path / "manifest.json"
    path.write_text(MediaManifest(version=1, entries=[metadata]).model_dump_json())
    index = Mock()
    monkeypatch.setattr(cli, "QwenEmbedding", lambda *args: Embedding())
    monkeypatch.setattr(cli, "OpenSearchStore", lambda *args: index)
    monkeypatch.setattr(cli, "MediaSettings", lambda: MediaSettings(root=tmp_path / "state"))
    monkeypatch.setattr(sys, "argv", ["cli", "ingest", str(path), "--execute"])
    assert cli.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ingest_mode"] == "metadata_only" and not result["media_download_performed"]
    assert result["indexed"] == 1
    index.close.assert_called_once()


def test_metadata_manifest_has_no_media_byte_requirements():
    manifest = load_manifest(Path("data/metadata_manifest.json"))
    assert len(manifest.entries) == 8
    assert all(e.source_id.startswith("commons:") for e in manifest.entries)
    assert all(
        e.expected_sha256 is None and e.min_age is None and e.subtitle_text is None
        for e in manifest.entries
    )
