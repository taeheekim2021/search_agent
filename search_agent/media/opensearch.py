"""OpenSearch 2.19+ Faiss/HNSW cosine vectors with identical hard filters on both branches."""

import json
import random
import re
import ssl
import time
from typing import Any

import httpx

from search_agent.config import QWEN_ID
from search_agent.domain import Conditions
from search_agent.models import normalize

from .config import OpenSearchSettings
from .domain import MediaRecord

DIMENSIONS = 2560


class OpenSearchError(RuntimeError):
    pass


def index_mapping(identity: dict) -> dict:
    if identity.get("model") != QWEN_ID or identity.get("dimensions") != DIMENSIONS:
        raise ValueError("Media index requires Qwen/Qwen3-Embedding-4B at 2560 dimensions")
    if not identity.get("resolved_revision"):
        raise ValueError("Immutable embedding revision is required")
    keyword_fields = [
        "content_id",
        "source_id",
        "rights_scope",
        "canonical_url",
        "original_url",
        "tags",
        "characters",
        "language",
        "media_format",
        "author",
        "license",
        "license_url",
        "checksum_sha256",
        "expected_sha256",
        "local_path",
        "age_rating_source",
        "subtitle_source_url",
        "subtitle_license",
        "metadata_license",
    ]
    props: dict[str, Any] = {key: {"type": "keyword"} for key in keyword_fields}
    props.update(
        {
            key: {"type": "text"}
            for key in [
                "title",
                "description",
                "subtitle_text",
                "attribution",
                "search_text",
                "license_notes",
            ]
        }
    )
    props.update({key: {"type": "integer"} for key in ["min_age", "max_age"]})
    props.update(
        {
            "duration_seconds": {"type": "float"},
            "size_bytes": {"type": "long"},
            "retrieved_at": {"type": "date"},
            "rights_verified": {"type": "boolean"},
            "metadata_provenance": {"type": "object", "enabled": False},
            "source_metadata": {"type": "object", "enabled": False},
            "embedding": {
                "type": "knn_vector",
                "dimension": DIMENSIONS,
                "method": {
                    "name": "hnsw",
                    "engine": "faiss",
                    "space_type": "cosinesimil",
                    "parameters": {"ef_construction": 128, "m": 16},
                },
            },
        }
    )
    return {
        "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {
            "dynamic": "strict",
            "_meta": {"schema": "media-v1", "embedding_identity": identity},
            "properties": props,
        },
    }


def hard_filter(conditions: Conditions) -> dict:
    filters: list[dict] = []
    if conditions.age is not None:
        # Missing/null ratings never satisfy these range queries.
        filters += [
            {"range": {"min_age": {"lte": conditions.age}}},
            {"range": {"max_age": {"gte": conditions.age}}},
        ]
    filters += [{"term": {"tags": t}} for t in conditions.topics]
    filters += [{"term": {"characters": c}} for c in conditions.characters]
    return {"bool": {"filter": filters}} if filters else {"match_all": {}}


def bm25_query(query: str, conditions: Conditions, k: int) -> dict:
    return {
        "size": k,
        "_source": {"excludes": ["embedding", "search_text"]},
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["title^3", "description^2", "search_text", "subtitle_text"],
                        }
                    }
                ],
                "filter": [hard_filter(conditions)],
            }
        },
        "sort": [{"_score": "desc"}, {"content_id": "asc"}],
    }


def vector_query(vector, conditions: Conditions, k: int) -> dict:
    q = normalize([vector], 1, DIMENSIONS)[0]
    return {
        "size": k,
        "_source": {"excludes": ["embedding", "search_text"]},
        "query": {
            "knn": {"embedding": {"vector": q.tolist(), "k": k, "filter": hard_filter(conditions)}}
        },
        "sort": [{"_score": "desc"}, {"content_id": "asc"}],
    }


class OpenSearchStore:
    def __init__(self, settings: OpenSearchSettings, client: httpx.Client | None = None):
        self.settings = settings
        self.index = settings.index
        auth = (
            (settings.username, settings.password.get_secret_value())
            if settings.username and settings.password
            else None
        )
        context = ssl.create_default_context(
            cafile=str(settings.ca_certs) if settings.ca_certs else None
        )
        self.client = client or httpx.Client(
            base_url=settings.url.rstrip("/"),
            verify=context,
            auth=auth,
            timeout=settings.timeout_seconds,
            follow_redirects=False,
        )

    def close(self):
        self.client.close()

    def request(self, method: str, path: str, *, body=None, allow_not_found=False) -> dict | None:
        try:
            response = self.client.request(method, path, json=body)
            if response.status_code == 404 and allow_not_found:
                return None
            response.raise_for_status()
            if response.status_code >= 300:
                raise OpenSearchError("OpenSearch redirects are not allowed")
            result = response.json()
            if not isinstance(result, dict):
                raise TypeError("Expected OpenSearch response object")
            return result
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            raise OpenSearchError(
                f"OpenSearch request failed ({status or type(exc).__name__})"
            ) from exc

    def ensure_index(self, identity: dict):
        if self.settings.production and not re.fullmatch(
            r"[a-z0-9][a-z0-9_-]*-v1-[a-z0-9][a-z0-9_-]*", self.index
        ):
            raise OpenSearchError("Production writes require a versioned concrete index")
        expected = index_mapping(identity)
        expected["settings"]["index"].update(
            number_of_shards=self.settings.shards, number_of_replicas=self.settings.replicas
        )
        info = self.request("GET", "/")
        try:
            version = tuple(int(x) for x in info["version"]["number"].split(".")[:2])  # type: ignore[index]
        except (KeyError, ValueError, TypeError) as exc:
            raise OpenSearchError("Cannot determine OpenSearch version") from exc
        if version < (2, 19):
            raise OpenSearchError("Faiss cosine media index requires OpenSearch >= 2.19")
        current = self.request("GET", f"/{self.index}/_mapping", allow_not_found=True)
        if current is None:
            created = self.request("PUT", f"/{self.index}", body=expected)
            if not created or created.get("acknowledged") is not True:
                raise OpenSearchError("Index creation not acknowledged; inspect before retry")
        else:
            if set(current) != {self.index}:
                raise OpenSearchError("Index setup requires a concrete index, not an alias")
            self.validate_mapping(current, identity)
            properties = next(iter(current.values()))["mappings"]["properties"]
            missing = {
                field: {"type": "keyword"}
                for field in ("source_id", "rights_scope")
                if field not in properties
            }
            if missing:
                # Additive fields keep older media-v1 indices readable.
                self.request("PUT", f"/{self.index}/_mapping", body={"properties": missing})

    def validate_mapping(self, mapping: dict, identity: dict):
        try:
            if len(mapping) != 1:
                raise OpenSearchError("Search requires exactly one concrete index")
            actual = next(iter(mapping.values()))["mappings"]
            vector = actual["properties"]["embedding"]
            if actual["properties"].get("source_id", {"type": "keyword"}).get("type") != "keyword":
                raise OpenSearchError("Incompatible source_id mapping")
            if actual["_meta"] != index_mapping(identity)["mappings"]["_meta"]:
                raise OpenSearchError(
                    "Embedding identity/schema mismatch; create a new index and reingest"
                )
            expected = index_mapping(identity)["mappings"]
            if actual.get("dynamic") != "strict":
                raise OpenSearchError("Strict schema required")
            for field, spec in expected["properties"].items():
                if field in ("source_id", "rights_scope") and field not in actual["properties"]:
                    continue  # Existing v1 indexes receive this additive field in ensure_index.
                if actual["properties"].get(field) != spec:
                    raise OpenSearchError("Incompatible media field mapping")
            if (
                vector["dimension"] != DIMENSIONS
                or vector["method"]["engine"] != "faiss"
                or vector["method"]["space_type"] != "cosinesimil"
            ):
                raise OpenSearchError("Incompatible media vector mapping")
        except (KeyError, TypeError) as exc:
            raise OpenSearchError("Missing media index mapping metadata") from exc

    def verify_identity(self, identity: dict):
        mapping = self.request("GET", f"/{self.index}/_mapping")
        if mapping is None:
            raise OpenSearchError("Media index does not exist")
        self.validate_mapping(mapping, identity)

    def upsert(self, record: MediaRecord, vector):
        v = normalize([vector], 1, DIMENSIONS)[0]
        doc = record.model_dump(exclude_none=True)
        doc["content_id"] = record.content_id
        doc["search_text"] = record.passage()
        doc["embedding"] = v.tolist()
        # Stable _id replaces the whole document; no duplicate versions in search results.
        return self.request(
            "PUT", f"/{self.index}/_doc/{record.content_id}?refresh=wait_for", body=doc
        )

    def vocabulary(self) -> tuple[list[str], list[str]]:
        response = self.request(
            "POST",
            f"/{self.index}/_search",
            body={
                "size": 0,
                "aggs": {
                    key: {"terms": {"field": key, "size": 1000}} for key in ["tags", "characters"]
                },
            },
        )
        if not response:
            raise OpenSearchError("Missing vocabulary response")
        self.require_complete(response)
        aggregations = response["aggregations"]
        if any(aggregations[key].get("sum_other_doc_count", 0) for key in aggregations):
            raise OpenSearchError(
                "Vocabulary exceeds 1000 terms; configure a full vocabulary adapter"
            )
        return tuple(
            [str(bucket["key"]) for bucket in aggregations[key]["buckets"]]
            for key in ["tags", "characters"]
        )  # type: ignore[return-value]

    def count(self, conditions: Conditions) -> int:
        response = self.request(
            "POST", f"/{self.index}/_count", body={"query": hard_filter(conditions)}
        )
        if response is None:
            raise OpenSearchError("Missing count response")
        self.require_complete(response)
        count = response.get("count")
        if type(count) is not int or count < 0:
            raise OpenSearchError("Malformed count response")
        return count

    def search(self, body: dict) -> list[MediaRecord]:
        response = self.request("POST", f"/{self.index}/_search", body=body)
        if response is None:
            raise OpenSearchError("Missing search response")
        if response.get("timed_out") or response.get("_shards", {}).get("failed", 0):
            raise OpenSearchError("Partial/timed-out OpenSearch response rejected")
        records = []
        try:
            for hit in response["hits"]["hits"]:
                raw = dict(hit["_source"])
                content_id = raw.pop("content_id")
                raw.pop("embedding", None)
                raw.pop("search_text", None)
                record = MediaRecord.model_validate(raw)
                if record.content_id != content_id or hit["_id"] != content_id:
                    raise OpenSearchError("Search hit has inconsistent content identity")
                records.append(record)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise OpenSearchError("Malformed search response") from exc
        return records

    @staticmethod
    def require_complete(response: dict):
        if response.get("timed_out") or response.get("_shards", {}).get("failed", 0):
            raise OpenSearchError("Partial/timed-out OpenSearch response rejected")

    def snapshot(self):
        """Resolve once so an alias switch cannot mix retrieval branches/revisions."""
        mapping = self.request("GET", f"/{self.index}/_mapping")
        if not mapping or len(mapping) != 1:
            raise OpenSearchError("Search requires exactly one concrete index")
        name = next(iter(mapping))
        if name != self.index:
            aliases = self.request("GET", f"/_alias/{self.index}")
            self.validate_read_alias(aliases, self.index, name)
        return OpenSearchStore(self.settings.model_copy(update={"index": name}), self.client)

    @staticmethod
    def validate_read_alias(aliases: dict | None, alias: str, target: str):
        try:
            if not aliases or set(aliases) != {target}:
                raise OpenSearchError("Alias changed or has multiple targets; retry operation")
            options = aliases[target]["aliases"][alias]
            if not isinstance(options, dict):
                raise TypeError("Malformed alias options")
            if any(
                key in options for key in ("filter", "routing", "search_routing", "index_routing")
            ):
                raise OpenSearchError("Filtered/routed aliases are not supported")
        except (KeyError, TypeError) as exc:
            raise OpenSearchError("Malformed or changed alias; retry operation") from exc

    def checkpoint_target(self) -> dict:
        info = self.request("GET", f"/{self.index}/_settings")
        if not info or set(info) != {self.index}:
            raise OpenSearchError("Bulk ingestion requires a concrete index, not an alias")
        settings = info[self.index]["settings"]["index"]
        if self.settings.production and int(settings.get("number_of_replicas", 0)) < 1:
            raise OpenSearchError("Production target requires at least one actual replica")
        if settings.get("blocks", {}).get("write") in (True, "true"):
            raise OpenSearchError("Published index is write blocked; create a new version")
        return {
            "endpoint": self.settings.url.rstrip("/"),
            "index": self.index,
            "uuid": settings["uuid"],
        }

    def bulk_upsert(self, items: list[tuple[MediaRecord, Any]]) -> list[dict]:
        """Bounded idempotent replacement writes; retry only transient/ambiguous failures.

        Reports omit server error reasons, which can contain source data or secrets.
        """
        if len(items) > self.settings.bulk_size:
            raise ValueError("Bulk item limit exceeded")
        payloads = []
        for record, vector in items:
            doc = record.model_dump(exclude_none=True)
            doc.update(
                content_id=record.content_id,
                search_text=record.passage(),
                embedding=normalize([vector], 1, DIMENSIONS)[0].tolist(),
            )
            payloads.append(
                json.dumps({"index": {"_id": record.content_id}})
                + "\n"
                + json.dumps(doc, ensure_ascii=False, allow_nan=False)
                + "\n"
            )
        if len("".join(payloads).encode()) > self.settings.bulk_max_bytes:
            raise ValueError("Bulk byte limit exceeded; reduce OPENSEARCH_BULK_SIZE")
        pending = list(range(len(items)))
        results: list[dict] = [{} for _ in items]
        transient = {429, 502, 503, 504}
        for attempt in range(self.settings.bulk_retries + 1):
            if not pending:
                break
            statuses = []
            try:
                response = self.client.post(
                    f"/{self.index}/_bulk?refresh=wait_for&wait_for_active_shards=all",
                    content="".join(payloads[i] for i in pending).encode(),
                    headers={"Content-Type": "application/x-ndjson"},
                )
                if response.status_code != 200:
                    statuses = [(response.status_code, False)] * len(pending)
                else:
                    body = response.json()
                    if len(body["items"]) != len(pending):
                        raise ValueError("Invalid bulk response length")
                    for i, item in zip(pending, body["items"], strict=True):
                        action = item["index"]
                        if action["_id"] != items[i][0].content_id:
                            raise ValueError("Invalid bulk response identity")
                        status = action["status"]
                        if type(status) is not int:
                            raise ValueError("Invalid status")
                        ok = 200 <= status < 300 and not action.get("error")
                        ok = ok and not action.get("_shards", {}).get("failed", 0)
                        statuses.append((status, ok))
            except httpx.TransportError:
                statuses = [(503, False)] * len(pending)
            except (KeyError, TypeError, ValueError, AttributeError):
                statuses = [(502, False)] * len(pending)
            retry = []
            for i, (status, ok) in zip(pending, statuses, strict=True):
                results[i] = {
                    "content_id": items[i][0].content_id,
                    "status": "indexed" if ok else "failed",
                    "http_status": status,
                    "attempts": attempt + 1,
                }
                if not ok and status in transient and attempt < self.settings.bulk_retries:
                    retry.append(i)
            pending = retry
            if pending:
                time.sleep(
                    self.settings.retry_base_seconds * (2**attempt) * random.uniform(0.5, 1.5)
                )
        return results

    def publish(
        self, alias: str, identity: dict, *, expected_current: str | None, expected_count: int
    ) -> dict:
        """Publish/rollback a validated immutable version; never delete any index.

        One authorized publisher must serialize operations (especially first publication).
        """
        pattern = r"[a-z0-9][a-z0-9_-]{0,100}"
        if not re.fullmatch(pattern, alias) or alias == self.index:
            raise ValueError("Invalid read alias")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*-v1-[a-z0-9][a-z0-9_-]*", self.index):
            raise ValueError("Publication requires a versioned name: <prefix>-v1-<build>")
        if expected_count < 1:
            raise ValueError("Publication requires a positive expected document count")
        if expected_current is not None and not re.fullmatch(pattern, expected_current):
            raise ValueError("Invalid expected current index")
        current = self.request("GET", f"/_alias/{alias}", allow_not_found=True) or {}
        if set(current) != ({expected_current} if expected_current else set()):
            raise OpenSearchError("Alias changed or has multiple targets; publication rejected")
        if expected_current:
            self.validate_read_alias(current, alias, expected_current)
        mapping = self.request("GET", f"/{self.index}/_mapping")
        if not mapping or set(mapping) != {self.index}:
            raise OpenSearchError("Publication target must be a concrete index")
        self.validate_mapping(mapping, identity)
        if self.settings.production:
            info = self.request("GET", f"/{self.index}/_settings")
            if (
                not info
                or int(info[self.index]["settings"]["index"].get("number_of_replicas", 0)) < 1
            ):
                raise OpenSearchError("Production publication requires at least one actual replica")
        health = self.request(
            "GET", f"/_cluster/health/{self.index}?wait_for_status=green&timeout=5s"
        )
        if not health or health.get("timed_out") or health.get("status") != "green":
            raise OpenSearchError("Publication requires green index health")
        # The add-block API waits for in-flight writes before acknowledging.
        blocked = self.request("PUT", f"/{self.index}/_block/write")
        indices = blocked.get("indices") if blocked else None
        complete = (
            blocked is not None
            and blocked.get("acknowledged") is True
            and blocked.get("shards_acknowledged") is True
            and isinstance(indices, list)
            and len(indices) == 1
            and isinstance(indices[0], dict)
            and indices[0].get("name") == self.index
            and indices[0].get("blocked") is True
            and not indices[0].get("exception")
            and not indices[0].get("error")
        )
        # OpenSearch 2.19 MetadataIndexStateService returns this exact no-op when the
        # final block already exists. Temporary UUID blocks are reverified instead.
        # This supports rollback without removing the retained index's write block.
        # Release operators must not install/remove blocks directly through _settings.
        already_blocked = (
            blocked is not None
            and set(blocked) == {"acknowledged", "shards_acknowledged", "indices"}
            and blocked["acknowledged"] is True
            and blocked["shards_acknowledged"] is False
            and indices == []
        )
        if already_blocked:
            info = self.request("GET", f"/{self.index}/_settings")
            try:
                already_blocked = (
                    info is not None
                    and set(info) == {self.index}
                    and (info[self.index]["settings"]["index"]["blocks"]["write"] in (True, "true"))
                )
            except (KeyError, TypeError):
                already_blocked = False
        if not complete and not already_blocked:
            raise OpenSearchError("Write block not fully acknowledged; alias unchanged")
        refreshed = self.request("POST", f"/{self.index}/_refresh")
        if not refreshed or refreshed.get("_shards", {}).get("successful", 0) < 1:
            raise OpenSearchError("Refresh not acknowledged")
        self.require_complete(refreshed)
        if self.count(Conditions()) != expected_count:
            raise OpenSearchError("Document count mismatch; target remains write blocked")
        actions = []
        if expected_current:
            actions.append(
                {"remove": {"index": expected_current, "alias": alias, "must_exist": True}}
            )
        actions.append({"add": {"index": self.index, "alias": alias, "is_write_index": False}})
        result = self.request("POST", "/_aliases", body={"actions": actions})
        if not result or result.get("acknowledged") is not True:
            raise OpenSearchError("Alias outcome uncertain; inspect alias before retry")
        actual = self.request("GET", f"/_alias/{alias}")
        if not actual or set(actual) != {self.index}:
            raise OpenSearchError("Alias verification failed; inspect before retry")
        return {
            "status": "published",
            "alias": alias,
            "index": self.index,
            "previous_index": expected_current,
            "document_count": expected_count,
        }
