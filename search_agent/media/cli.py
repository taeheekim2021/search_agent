"""python -m search_agent.media.cli --help"""

import argparse
import json
from pathlib import Path

from search_agent.config import Settings
from search_agent.models import QwenEmbedding
from search_agent.retrieval import EmbeddingCache

from .config import MediaSettings, OpenSearchSettings
from .ingest import Ingestor, adapt_commons_source, load_manifest
from .local_import import import_local
from .opensearch import OpenSearchStore


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate/index licensed media metadata without downloading videos. Default: dry-run."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser(
        "adapt-commons", help="Convert reviewed source manifest to ingestion schema"
    )
    convert.add_argument("source", type=Path)
    convert.add_argument("output", type=Path)
    local = commands.add_parser(
        "import-local", help="Import one already-authorized file using its verified checksum"
    )
    local.add_argument("manifest", type=Path)
    local.add_argument("file", type=Path)
    local.add_argument(
        "--probe", action="store_true", help="Validate streams with installed ffprobe"
    )
    ingest = commands.add_parser("ingest")
    ingest.add_argument("manifest", type=Path)
    ingest.add_argument("--limit", type=int, default=None)
    mode = ingest.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Run real Qwen and upsert metadata in OpenSearch; no media download",
    )
    mode.add_argument(
        "--download-only", action="store_true", help="Download only; no model/OpenSearch calls"
    )
    ingest.add_argument(
        "--download-media",
        action="store_true",
        help="Optional legacy download+index path; requires --execute",
    )
    ingest.add_argument(
        "--resume",
        action="store_true",
        help="Skip matching successful checkpoints for this concrete index UUID",
    )
    publish = commands.add_parser(
        "publish", help="Publish or roll back a versioned index read alias"
    )
    publish.add_argument("alias")
    publish.add_argument(
        "--expected-current",
        required=True,
        help="Current concrete index, or NONE for first publication",
    )
    publish.add_argument("--expected-count", type=int, required=True)
    publish.add_argument("--execute", action="store_true", help="Apply the reviewed alias switch")
    args = parser.parse_args()
    if args.command == "ingest" and args.download_media and not args.execute:
        parser.error("--download-media requires --execute")
    store = None
    try:
        if args.command == "adapt-commons":
            manifest = adapt_commons_source(json.loads(args.source.read_text("utf-8")))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            print(json.dumps({"status": "adapted", "entries": len(manifest.entries)}))
            return 0
        if args.command == "publish":
            models = Settings()
            os_config = OpenSearchSettings.for_search(models)
            if not args.execute:
                print(
                    json.dumps(
                        {
                            "status": "planned",
                            "index": os_config.index,
                            "alias": args.alias,
                            "expected_current": args.expected_current,
                            "expected_count": args.expected_count,
                            "network_accessed": False,
                        }
                    )
                )
                return 0
            if models.mode != "real" or models.dimensions != 2560:
                raise ValueError("Publication requires real mode and 2560 dimensions")
            publisher_embedding = QwenEmbedding(models.embedding, str(models.hf_cache_dir), 2560)
            store = OpenSearchStore(os_config)
            result = store.publish(
                args.alias,
                publisher_embedding.identity,
                expected_current=None if args.expected_current == "NONE" else args.expected_current,
                expected_count=args.expected_count,
            )
            print(json.dumps(result))
            return 0
        manifest = load_manifest(args.manifest)
        if args.command == "import-local":
            if len(manifest.entries) != 1:
                raise ValueError("import-local requires a one-entry manifest")
            print(
                json.dumps(
                    import_local(manifest.entries[0], args.file, MediaSettings(), probe=args.probe),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be >= 1")
            manifest.entries = manifest.entries[: args.limit]
        media = MediaSettings()
        embedding = None
        cache = None
        if args.execute:
            models = Settings()
            if models.mode != "real" or models.dimensions != 2560:
                raise ValueError("Media index requires real mode and 2560 dimensions")
            embedding = QwenEmbedding(models.embedding, str(models.hf_cache_dir), 2560)
            os_config = OpenSearchSettings.for_search(models)
            store = OpenSearchStore(os_config)
            cache = EmbeddingCache(media.root / ".vectors")
        result = Ingestor(media, embedding=embedding, index=store, cache=cache).run(
            manifest,
            dry_run=not (args.execute or args.download_only),
            download_only=args.download_only,
            download_media=args.download_media,
            resume=args.resume,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result["status"] == "partial_failure" else 0
    except Exception as exc:  # noqa: BLE001 — sanitize dependency errors at CLI boundary
        # Avoid exposing credentials or signed URL query strings from dependency errors.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 2
    finally:
        if store:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
