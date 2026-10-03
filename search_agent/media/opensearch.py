"""OpenSearch 2.19+ Faiss/HNSW cosine vectors with identical hard filters on both branches."""

import ssl
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
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            raise OpenSearchError(
                f"OpenSearch request failed ({status or type(exc).__name__})"
            ) from exc

    def ensure_index(self, identity: dict):
        expected = index_mapping(identity)
        info = self.request("GET", "/")
        try:
            version = tuple(int(x) for x in info["version"]["number"].split(".")[:2])  # type: ignore[index]
        except (KeyError, ValueError, TypeError) as exc:
            raise OpenSearchError("Cannot determine OpenSearch version") from exc
        if version < (2, 19):
            raise OpenSearchError("Faiss cosine media index requires OpenSearch >= 2.19")
        current = self.request("GET", f"/{self.index}/_mapping", allow_not_found=True)
        if current is None:
            self.request("PUT", f"/{self.index}", body=expected)
        else:
            self.validate_mapping(current, identity)

    def validate_mapping(self, mapping: dict, identity: dict):
        try:
            actual = mapping[self.index]["mappings"]
            vector = actual["properties"]["embedding"]
            if actual["_meta"] != index_mapping(identity)["mappings"]["_meta"]:
                raise OpenSearchError(
                    "Embedding identity/schema mismatch; create a new index and reingest"
                )
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
        doc = record.model_dump()
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
        return int(response["count"])

    def search(self, body: dict) -> list[MediaRecord]:
        response = self.request("POST", f"/{self.index}/_search", body=body)
        if response is None:
            raise OpenSearchError("Missing search response")
        if response.get("timed_out") or response.get("_shards", {}).get("failed", 0):
            raise OpenSearchError("Partial/timed-out OpenSearch response rejected")
        records = []
        for hit in response["hits"]["hits"]:
            raw = dict(hit["_source"])
            content_id = raw.pop("content_id")
            raw.pop("embedding", None)
            raw.pop("search_text", None)
            record = MediaRecord.model_validate(raw)
            if record.content_id != content_id or hit["_id"] != content_id:
                raise OpenSearchError("Search hit has inconsistent content identity")
            records.append(record)
        return records
