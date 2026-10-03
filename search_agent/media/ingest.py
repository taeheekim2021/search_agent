import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from search_agent.models import EmbeddingAdapter, ModelError
from search_agent.retrieval import EmbeddingCache

from .config import MediaSettings
from .domain import MediaEntry, MediaManifest, MediaRecord, Provenance
from .download import DownloadError, LocalMediaStore, MediaStorage, URLPolicy


class IndexWriter(Protocol):
    def ensure_index(self, identity: dict): ...
    def upsert(self, record: MediaRecord, vector): ...


def load_manifest(path: Path) -> MediaManifest:
    if path.stat().st_size > 2_000_000:
        raise ValueError("Manifest exceeds 2 MB limit")
    return MediaManifest.model_validate_json(path.read_text("utf-8"))


def atomic_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Receipt symlink rejected")
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False, encoding="utf-8"
    ) as f:
        tmp = Path(f.name)
        try:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


class Ingestor:
    def __init__(
        self,
        settings: MediaSettings,
        storage: MediaStorage | None = None,
        embedding: EmbeddingAdapter | None = None,
        index: IndexWriter | None = None,
        cache: EmbeddingCache | None = None,
    ):
        self.settings = settings
        self.storage = storage or LocalMediaStore(settings)
        self.embedding, self.index = embedding, index
        self.cache = cache or EmbeddingCache(settings.root / ".vectors")

    def run(
        self, manifest: MediaManifest, *, dry_run: bool = True, download_only: bool = False
    ) -> dict:
        policy = URLPolicy(self.settings.allowed_domains)
        for entry in manifest.entries:
            policy.check(entry.original_url)
        if dry_run:
            return {
                "dry_run": True,
                "network_accessed": False,
                "status": "planned",
                "max_total_bytes": self.settings.total_max_bytes,
                "entries": [
                    {
                        "content_id": e.content_id,
                        "title": e.title,
                        "license": e.license,
                        "age_known": e.min_age is not None,
                        "original_hostname": policy.check(e.original_url),
                        "dns_check": "deferred_until_download",
                    }
                    for e in manifest.entries
                ],
            }
        if not download_only and (self.embedding is None or self.index is None):
            raise ValueError("Indexing requires real embedding and OpenSearch adapters")
        with media_lock(self.settings.root):
            return self._run(manifest, download_only=download_only)

    def _run(self, manifest: MediaManifest, *, download_only: bool) -> dict:
        results, used = [], 0
        ready, setup_error = False, None
        for entry in manifest.entries:
            receipt_path = self.settings.root / f"{entry.content_id}.receipt.json"
            receipt: dict = {
                "content_id": entry.content_id,
                "status": "pending",
                "stage": "download",
            }
            try:
                prior: dict = {}
                if receipt_path.is_symlink():
                    raise ValueError("Receipt symlink rejected")
                if receipt_path.exists():
                    prior = json.loads(receipt_path.read_text()).get("download", {})
                remaining = self.settings.total_max_bytes - used
                if remaining <= 0:
                    raise ValueError("Batch media byte budget exhausted")
                if isinstance(self.storage, LocalMediaStore):
                    bounded = self.settings.model_copy(
                        update={"max_bytes": min(self.settings.max_bytes, remaining)}
                    )
                    downloaded = LocalMediaStore(bounded, self.storage.transport).download(
                        entry, prior
                    )
                else:
                    downloaded = self.storage.download(entry, prior)
                used += downloaded.size_bytes
                if used > self.settings.total_max_bytes:
                    raise ValueError("Storage exceeded batch budget")
                receipt["download"] = dict(asdict(downloaded), original_url=entry.original_url)
                receipt.update(status="downloaded", stage="download")
                atomic_json(receipt_path, receipt)
                if download_only:
                    results.append(receipt)
                    continue
                assert self.embedding is not None and self.index is not None
                receipt["stage"] = "embedding"
                if setup_error is not None:
                    raise setup_error
                if not ready:
                    try:
                        identity = self.embedding.identity
                        receipt["stage"] = "index_setup"
                        self.index.ensure_index(identity)
                        ready = True
                    except Exception as exc:
                        setup_error = exc
                        raise
                receipt["stage"] = "embedding"
                vector = self.cache.documents(self.embedding, [entry])[0]
                record = MediaRecord(
                    **entry.model_dump(),
                    local_path=downloaded.local_path,
                    checksum_sha256=downloaded.checksum_sha256,
                    size_bytes=downloaded.size_bytes,
                    retrieved_at=datetime.now(UTC).isoformat(),
                )
                receipt["stage"] = "index_upsert"
                self.index.upsert(record, vector)
                receipt.update(status="indexed", stage="complete")
                atomic_json(receipt_path, receipt)
            except Exception as exc:  # noqa: BLE001 — per-item failure receipt is the CLI contract
                receipt.update(status="failed", error_type=type(exc).__name__)
                if isinstance(exc, DownloadError):
                    receipt["message"] = str(exc)
                if isinstance(exc, ModelError):
                    receipt["stage"] = exc.stage
                # Preserve successful download receipt for a retry after indexing failure.
                try:
                    atomic_json(receipt_path, receipt)
                except (ValueError, OSError):
                    receipt["receipt_write_failed"] = True
            results.append(receipt)
        return {
            "dry_run": False,
            "status": "partial_failure"
            if any(r["status"] == "failed" for r in results)
            else "completed",
            "indexed": sum(r["status"] == "indexed" for r in results),
            "downloaded_only": sum(r["status"] == "downloaded" for r in results),
            "bytes_accounted": used,
            "entries": results,
        }


def adapt_commons_source(source: dict) -> MediaManifest:
    entries = []
    for asset in source["assets"]:
        url = asset["canonical_url"]
        verification = asset["verification"]
        if not (
            verification.get("source_page_live_browser")
            and verification.get("direct_url_observed_in_original_file_link")
        ):
            raise ValueError("Source must have verified source page and original-file link")
        provenance = {
            field: {
                "method": "assistant_translation",
                "source_url": url,
                "note": asset["korean_metadata_origin"],
            }
            for field in ("title", "description")
        }
        provenance.update(
            {
                "tags": {"method": "editorial", "source_url": url, "note": asset["tags_origin"]},
                "language": {
                    "method": "source" if asset["media_language"] else "unknown",
                    "source_url": url,
                    "note": asset["media_language_status"],
                },
                "duration_seconds": {
                    "method": "source",
                    "source_url": url,
                    "note": "Source file-page duration; not independently measured here.",
                },
            }
        )
        entries.append(
            MediaEntry(
                canonical_url=url,
                original_url=asset["media_url"],
                title=asset["title_ko"],
                description=asset["description_ko"],
                tags=asset["tags_ko"],
                language=asset["media_language"] or "und",
                media_format="webm",
                duration_seconds=asset["duration_seconds"],
                author=asset["author"],
                license=asset["license_id"],
                license_url=asset["license_url"],
                attribution=asset["attribution"],
                rights_verified=True,
                metadata_provenance={
                    k: Provenance.model_validate(v) for k, v in provenance.items()
                },
                expected_sha256=asset.get("sha256"),
                license_notes=asset["license_notes"],
                metadata_license=source["manifest_license"],
                source_metadata={
                    "original_title": asset["original_title"],
                    "description_original": asset["description_original"],
                    "source_id": asset["source_id"],
                    "verification": verification,
                    "source_categories": asset["source_categories"],
                    "license_evidence_url": asset["license_evidence_url"],
                    "required_source_credit": asset.get("required_source_credit"),
                    "child_suitability": asset["child_suitability"],
                    "metadata_license_url": source["manifest_license_url"],
                },
            )
        )
    return MediaManifest(version=1, entries=entries)


@contextmanager
def media_lock(root: Path):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / ".ingest.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another ingestion is running for this MEDIA_ROOT") from exc
        yield
