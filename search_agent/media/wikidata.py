"""Collect at most 500 CC0 structured metadata records. Never fetch linked media or synopses."""

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx

from search_agent.media.domain import MediaEntry, MediaManifest, Provenance
from search_agent.media.ingest import atomic_json

ENDPOINT = "https://query.wikidata.org/sparql"
LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
USER_AGENT = "search-agent-metadata-starter/1.0 (https://github.com/taeheekim2021/search_agent)"


def sparql(limit: int) -> str:
    if not 1 <= limit <= 500:
        raise ValueError("Limit must be 1..500")
    return f"""SELECT ?item ?itemLabel WHERE {{
  {{ SELECT DISTINCT ?item WHERE {{ ?item wdt:P31 wd:Q202866 . }} ORDER BY ?item LIMIT {limit} }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "ko,en". }}
}} ORDER BY ?item"""


def convert(bindings: list[dict], retrieved_at: str, limit: int) -> MediaManifest:
    entries = []
    seen = set()
    for binding in bindings:
        value = binding["item"]["value"]
        match = re.fullmatch(r"https?://www\.wikidata\.org/entity/(Q[1-9][0-9]*)", value)
        if not match:
            raise ValueError("Unexpected Wikidata entity URL")
        qid = match[1]
        if qid in seen:
            continue
        seen.add(qid)
        title = binding["itemLabel"]["value"]
        url = "https://www.wikidata.org/wiki/" + qid
        entries.append(
            MediaEntry(
                source_id="wikidata:" + qid,
                canonical_url=url,
                original_url=url,
                title=title,
                description=f"{title}: Wikidata 애니메이션 영화 메타데이터 ({qid}). 영상·줄거리 미수집.",
                tags=["애니메이션", "영화"],
                author="Wikidata contributors (metadata only)",
                license="CC0-1.0",
                license_url=LICENSE_URL,
                metadata_license="CC0-1.0",
                attribution=f"Wikidata {qid}; structured metadata only, CC0.",
                rights_verified=True,
                rights_scope="metadata",
                license_notes="CC0 applies only to structured metadata, not the film/video/images. Media rights unverified.",
                metadata_provenance={
                    "title": Provenance(
                        method="source",
                        source_url=url,
                        note="Wikidata ko label, English/QID fallback.",
                    ),
                    "description": Provenance(
                        method="editorial",
                        source_url=url,
                        note="Generated factual template; no copied film synopsis.",
                    ),
                    "tags": Provenance(
                        method="source", source_url=url, note="P31 animated film (Q202866)."
                    ),
                },
                source_metadata={
                    "wikidata_id": qid,
                    "metadata_retrieved_at": retrieved_at,
                    "label_language": binding["itemLabel"].get("xml:lang", "und"),
                    "license_scope": "metadata_only",
                    "media_rights_verified": False,
                },
            )
        )
        if len(entries) >= limit:
            break
    return MediaManifest(version=1, entries=entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--execute", action="store_true", help="Issue one bounded public SPARQL request"
    )
    args = parser.parse_args()
    try:
        query = sparql(args.limit)
        if not args.execute:
            print(
                json.dumps(
                    {
                        "status": "planned",
                        "endpoint": ENDPOINT,
                        "limit": args.limit,
                        "query": query,
                        "media_downloaded": False,
                    }
                )
            )
            return 0
        with (
            httpx.Client(
                timeout=60, follow_redirects=False, headers={"User-Agent": USER_AGENT}
            ) as client,
            client.stream(
                "GET",
                ENDPOINT,
                params={"query": query, "format": "json"},
                headers={"Accept": "application/sparql-results+json"},
            ) as response,
        ):
            response.raise_for_status()
            chunks, size = [], 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > 2_000_000:
                    raise ValueError("Public response exceeded 2 MB")
                chunks.append(chunk)
        raw = b"".join(chunks)
        fetched = datetime.now(UTC).isoformat()
        manifest = convert(json.loads(raw)["results"]["bindings"], fetched, args.limit)
        atomic_json(args.output, manifest.model_dump())
        atomic_json(
            args.output.with_suffix(".provenance.json"),
            {
                "endpoint": ENDPOINT,
                "query": query,
                "retrieved_at": fetched,
                "response_sha256": hashlib.sha256(raw).hexdigest(),
                "manifest_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "records": len(manifest.entries),
                "license": "CC0-1.0",
                "scope": "structured_metadata_only",
                "license_evidence": "https://www.wikidata.org/wiki/Wikidata:Licensing",
                "representative_of_future_catalog": False,
            },
        )
        print(
            json.dumps(
                {"status": "collected", "records": len(manifest.entries), "media_downloaded": False}
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 — sanitized external boundary, no automatic retry storm
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
