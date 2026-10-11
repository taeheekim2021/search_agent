"""Import an explicitly supplied, checksummed local media file; never claim URL download."""

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from .config import MediaSettings
from .domain import MediaEntry
from .download import LocalMediaStore, UnsafeDownload, valid_magic
from .ingest import atomic_json, media_lock


def _import_local(
    entry: MediaEntry, source: Path, settings: MediaSettings, *, probe: bool = False
) -> dict:
    if not entry.expected_sha256:
        raise ValueError("Local import requires the verified source checksum in the manifest")
    if source.is_symlink() or not source.is_file():
        raise ValueError("Local import requires a regular non-symlink file")
    if source.stat().st_size > min(settings.max_bytes, settings.total_max_bytes):
        raise ValueError("Local media exceeds configured byte limit")
    target = LocalMediaStore(settings).target(entry)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest, size = hashlib.sha256(), 0
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".import-", delete=False) as f:
        temp = Path(f.name)
        try:
            with source.open("rb") as original:
                prefix = original.read(64)
                if not valid_magic(entry.media_format, prefix):
                    raise UnsafeDownload("Imported media signature mismatch")
                original.seek(0)
                while chunk := original.read(65536):
                    size += len(chunk)
                    if size > min(settings.max_bytes, settings.total_max_bytes):
                        raise ValueError("Local media exceeds configured byte limit")
                    digest.update(chunk)
                    f.write(chunk)
            if digest.hexdigest() != entry.expected_sha256:
                raise ValueError("Local media checksum mismatch")
            f.flush()
            os.fsync(f.fileno())
            probe_result = None
            if probe:
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-protocol_whitelist",
                        "file,pipe",
                        "-show_entries",
                        "format=duration:stream=codec_type,codec_name,width,height",
                        "-of",
                        "json",
                        str(temp),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                probe_result = json.loads(result.stdout)
                if not probe_result.get("streams"):
                    raise ValueError("ffprobe found no media streams")
            if LocalMediaStore(settings).target(entry) != target:
                raise UnsafeDownload("Storage destination changed")
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
    receipt = {
        "content_id": entry.content_id,
        "status": "downloaded",
        "stage": "local_import",
        "acquisition": "explicit_local_file",
        "url_download_performed": False,
        "probe": probe_result,
        "download": {
            "local_path": target.name,
            "checksum_sha256": digest.hexdigest(),
            "size_bytes": size,
            "reused": False,
            "original_url": entry.original_url,
        },
    }
    atomic_json(settings.root / f"{entry.content_id}.receipt.json", receipt)
    return receipt


def import_local(
    entry: MediaEntry, source: Path, settings: MediaSettings, *, probe: bool = False
) -> dict:
    if entry.rights_scope != "media":
        raise ValueError("Metadata licensing does not authorize media acquisition")
    with media_lock(settings.root):
        return _import_local(entry, source, settings, probe=probe)
