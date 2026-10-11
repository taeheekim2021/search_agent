"""Metadata-only bulk ingestion with durable, target-bound per-entry checkpoints."""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from .domain import MediaManifest, MediaRecord
from .ingest import atomic_json


def ingest_bulk(ingestor, manifest: MediaManifest, *, resume: bool) -> dict:
    store, embedding = ingestor.index, ingestor.embedding
    identity = embedding.identity
    store.ensure_index(identity)
    target = store.checkpoint_target()
    scope = {"target": target, "embedding_identity": identity, "schema": "media-v1"}
    namespace = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
    root = ingestor.settings.root / "checkpoints" / namespace
    results = []
    pending: list[tuple[MediaRecord, Any, dict, Any]] = []

    def save(receipt, path):
        try:
            # Preserve the existing per-content status receipt consumed by local operations.
            atomic_json(
                ingestor.settings.root / f"{receipt['content_id']}.metadata-receipt.json", receipt
            )
            atomic_json(path, receipt)
        except (OSError, ValueError):
            receipt.update(status="failed", stage="checkpoint", receipt_write_failed=True)
        results.append(receipt)

    def flush():
        if not pending:
            return
        try:
            outcomes = store.bulk_upsert([(r, v) for r, v, _, _ in pending])
        except Exception as exc:  # noqa: BLE001 — sanitized per-entry failure report
            outcomes = [{"status": "failed", "error_type": type(exc).__name__}] * len(pending)
        for (_, _, receipt, path), outcome in zip(pending, outcomes, strict=True):
            receipt.update(outcome)
            receipt["stage"] = "complete" if receipt["status"] == "indexed" else "index_upsert"
            save(receipt, path)
        pending.clear()

    for entry in manifest.entries:
        fingerprint = hashlib.sha256(entry.model_dump_json().encode()).hexdigest()
        path = root / f"{entry.content_id}.json"
        receipt = dict(
            scope,
            content_id=entry.content_id,
            source_id=entry.source_id,
            fingerprint=fingerprint,
            ingest_mode="metadata_only",
            media_download_performed=False,
            status="pending",
            stage="embedding",
        )
        try:
            if path.is_symlink():
                raise ValueError("Checkpoint symlink rejected")
            if resume and path.exists():
                try:
                    previous = json.loads(path.read_text("utf-8"))
                except (OSError, ValueError):
                    previous = {}  # Corrupt checkpoint must never suppress a write.
                if not isinstance(previous, dict):
                    previous = {}
                if (
                    previous.get("status") == "indexed"
                    and previous.get("fingerprint") == fingerprint
                    and all(previous.get(k) == v for k, v in scope.items())
                ):
                    results.append(dict(previous, resumed=True))
                    continue
            # Invalidate any old success before a new remote write can take effect.
            # A crash after remote commit must not allow an older metadata revision to skip.
            atomic_json(path, receipt)
            vector = ingestor.cache.documents(embedding, [entry])[0]
            record = MediaRecord(**entry.model_dump(), retrieved_at=datetime.now(UTC).isoformat())
            pending.append((record, vector, receipt, path))
            if len(pending) >= store.settings.bulk_size:
                flush()
        except Exception as exc:  # noqa: BLE001 — sanitized per-entry failure report
            receipt.update(status="failed", error_type=type(exc).__name__)
            save(receipt, path)
    flush()
    positions = {entry.content_id: i for i, entry in enumerate(manifest.entries)}
    results.sort(key=lambda receipt: positions[receipt["content_id"]])
    return {
        "dry_run": False,
        "ingest_mode": "metadata_only",
        "media_download_performed": False,
        "bytes_accounted": 0,
        "downloaded_only": 0,
        "status": "partial_failure"
        if any(r["status"] == "failed" for r in results)
        else "completed",
        "indexed": sum(r["status"] == "indexed" for r in results),
        "resumed": sum(bool(r.get("resumed")) for r in results),
        "checkpoint_directory": str(root),
        "entries": results,
    }
