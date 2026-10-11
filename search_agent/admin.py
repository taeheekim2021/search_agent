"""Opt-in, bounded administration using the running search service's adapters."""

import logging
import secrets
from contextlib import contextmanager
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPBearer
from pydantic import BaseModel, ConfigDict

from .config import BGE_ID, QWEN_ID, Settings
from .domain import Content, SearchRequest
from .media.config import MediaSettings
from .media.domain import MediaManifest, MediaRecord
from .media.opensearch import OpenSearchError
from .media.service import MediaSearchAgent
from .models import ModelError, TransformerAdapter
from .service import SearchAgent

MAX_BODY_BYTES = 2_000_000
MAX_PAGE_SIZE = 100
MAX_RESULT_WINDOW = 10_000
RESPONSE_HEADERS = {
    "Cache-Control": "no-store",
    "Pragma": "no-cache",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
logger = logging.getLogger(__name__)


def problem(status: int, code: str, message: str, **details: Any) -> HTTPException:
    return HTTPException(status, detail={"code": code, "message": message, **details})


def require_admin(request: Request, settings: Settings) -> None:
    """Run before reading a body; never accept keys in URLs or cookies."""
    if settings.admin_api_key is None:
        raise problem(404, "ADMIN_DISABLED", "관리자 서비스가 활성화되지 않았습니다.")
    scheme, _, credential = request.headers.get("authorization", "").partition(" ")
    expected = settings.admin_api_key.get_secret_value().encode("utf-8")
    if scheme.lower() != "bearer" or not secrets.compare_digest(
        credential.encode("utf-8"), expected
    ):
        raise HTTPException(
            401,
            detail={"code": "ADMIN_UNAUTHORIZED", "message": "관리자 키를 확인하세요."},
            headers={"WWW-Authenticate": "Bearer"},
        )


def admin_route_class(settings: Settings) -> type[APIRoute]:
    class AdminRoute(APIRoute):
        def get_route_handler(self):
            original = super().get_route_handler()

            async def handler(request: Request):
                try:
                    require_admin(request, settings)
                    length = request.headers.get("content-length")
                    if length is not None:
                        try:
                            declared = int(length)
                        except ValueError as exc:
                            raise problem(
                                400, "INVALID_REQUEST", "잘못된 요청 길이입니다."
                            ) from exc
                        if declared < 0:
                            raise problem(400, "INVALID_REQUEST", "잘못된 요청 길이입니다.")
                        if declared > MAX_BODY_BYTES:
                            raise problem(413, "REQUEST_TOO_LARGE", "요청은 2 MB 이하여야 합니다.")
                    # Bound actual streamed bytes as well as Content-Length. Authentication
                    # above also prevents unauthenticated requests from allocating this body.
                    chunks, size = [], 0
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > MAX_BODY_BYTES:
                            raise problem(413, "REQUEST_TOO_LARGE", "요청은 2 MB 이하여야 합니다.")
                        chunks.append(chunk)
                    body = b"".join(chunks)

                    async def receive():
                        return {"type": "http.request", "body": body, "more_body": False}

                    response = await original(Request(request.scope, receive=receive))
                except RequestValidationError as exc:
                    response = JSONResponse(
                        status_code=422,
                        content={
                            "detail": {
                                "code": "VALIDATION_ERROR",
                                "message": "요청 입력값을 확인하세요.",
                                "errors": [
                                    {"loc": e["loc"], "message": e["msg"], "type": e["type"]}
                                    for e in exc.errors()
                                ],
                            }
                        },
                    )
                except HTTPException as exc:
                    detail: Any = exc.detail
                    if not isinstance(detail, dict):
                        detail = {"code": "INVALID_REQUEST", "message": "요청 형식을 확인하세요."}
                    response = JSONResponse(
                        status_code=exc.status_code,
                        content={"detail": detail},
                        headers=exc.headers,
                    )
                except ModelError as exc:
                    logger.error("Admin model stage failed: %s", exc.stage)
                    response = JSONResponse(
                        status_code=503,
                        content={
                            "detail": {
                                "code": "MODEL_UNAVAILABLE",
                                "stage": exc.stage,
                                "message": "실제 모델 로딩·추론에 실패했습니다. 서버 로그를 확인하세요. "
                                "모의 모드로 자동 전환하지 않았습니다.",
                            }
                        },
                    )
                except OpenSearchError:
                    response = JSONResponse(
                        status_code=503,
                        content={
                            "detail": {
                                "code": "OPENSEARCH_UNAVAILABLE",
                                "message": "OpenSearch 연결·인덱스·응답을 확인하세요.",
                            }
                        },
                    )
                except OSError:
                    response = JSONResponse(
                        status_code=503,
                        content={
                            "detail": {
                                "code": "STORAGE_UNAVAILABLE",
                                "message": "서버의 카탈로그·캐시·적재 기록 저장소를 확인하세요.",
                            }
                        },
                    )
                except ValueError:
                    response = JSONResponse(
                        status_code=422,
                        content={
                            "detail": {
                                "code": "INVALID_REQUEST",
                                "message": "검색 조건 또는 적재 설정을 확인하세요.",
                            }
                        },
                    )
                except Exception as exc:  # noqa: BLE001 — sanitized API boundary
                    logger.error("Admin operation failed (%s)", type(exc).__name__)
                    response = JSONResponse(
                        status_code=500,
                        content={
                            "detail": {
                                "code": "ADMIN_OPERATION_FAILED",
                                "message": "관리자 작업에 실패했습니다. 서버 로그를 확인하세요.",
                            }
                        },
                    )
                response.headers.update(RESPONSE_HEADERS)
                return response

            return handler

    return AdminRoute


class ConfirmedAction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    confirmed: bool = False


class IngestRequest(ConfirmedAction):
    manifest: MediaManifest
    dry_run: bool = True


def safe_identity(identity: dict) -> dict:
    return {
        key: identity[key]
        for key in (
            "model",
            "revision",
            "resolved_revision",
            "dimensions",
            "max_length",
            "dtype",
            "device",
            "pooling",
        )
        if key in identity
    }


def complete_response(response: dict | None) -> dict:
    if not isinstance(response, dict):
        raise OpenSearchError("Missing OpenSearch response")
    try:
        if response.get("timed_out") or response.get("_shards", {}).get("failed", 0):
            raise OpenSearchError("Partial/timed-out OpenSearch response rejected")
    except (AttributeError, TypeError) as exc:
        raise OpenSearchError("Malformed OpenSearch response") from exc
    return response


def sample_item(item: Content) -> dict:
    return {
        "content_id": item.id,
        "title": item.title,
        "description": item.description,
        "tags": item.topics,
        "characters": item.characters,
        "min_age": item.min_age,
        "max_age": item.max_age,
        "language": "ko",
        "duration_seconds": item.duration_minutes * 60,
        "license": None,
        "source_id": item.id,
        "canonical_url": None,
        "is_sample": True,
    }


def media_item(item: MediaRecord) -> dict:
    # Do not return server-local media paths, full subtitles, or arbitrary source metadata.
    fields = {
        "title",
        "description",
        "tags",
        "characters",
        "min_age",
        "max_age",
        "language",
        "duration_seconds",
        "license",
        "source_id",
        "canonical_url",
        "original_url",
        "author",
        "license_url",
        "attribution",
        "metadata_provenance",
        "age_rating_source",
        "retrieved_at",
        "rights_verified",
        "rights_scope",
        "media_format",
        "metadata_license",
        "license_notes",
    }
    return dict(item.model_dump(include=fields), content_id=item.content_id, is_sample=False)


class AdminService:
    def __init__(
        self,
        settings: Settings,
        agent: SearchAgent | MediaSearchAgent,
        media_settings: MediaSettings | None = None,
    ):
        self.settings, self.agent, self.media_settings = settings, agent, media_settings

    def media(self) -> MediaSearchAgent:
        if self.settings.backend != "opensearch" or not isinstance(self.agent, MediaSearchAgent):
            raise problem(
                409,
                "BACKEND_UNSUPPORTED",
                "이 작업은 OpenSearch 미디어 백엔드에서 사용할 수 있습니다.",
            )
        return self.agent

    @contextmanager
    def operation(self):
        if not self.agent.lock.acquire(blocking=False):
            raise problem(
                409, "ADMIN_BUSY", "검색 또는 관리자 작업이 진행 중입니다. 완료 후 다시 시도하세요."
            )
        try:
            yield
        finally:
            self.agent.lock.release()

    @staticmethod
    def confirm(request: ConfirmedAction):
        if not request.confirmed:
            raise problem(
                409, "CONFIRMATION_REQUIRED", "실행을 확인한 뒤 confirmed: true로 요청하세요."
            )

    def models(self) -> dict:
        embedding = (
            self.agent.embedding
            if isinstance(self.agent, MediaSearchAgent)
            else self.agent.semantic.adapter
            if self.agent.semantic is not None
            else None
        )
        result = {}
        for name, model_id, adapter, config in (
            ("embedding", QWEN_ID, embedding, self.settings.embedding),
            ("reranker", BGE_ID, self.agent.reranker, self.settings.reranker),
        ):
            loaded: bool | None = None
            state = "adapter_unverified"
            if self.settings.mode == "mock":
                loaded, state = False, "disabled"
            elif isinstance(adapter, TransformerAdapter):
                loaded = adapter.model is not None
                state = "loaded" if loaded else "not_loaded"
            result[name] = {
                "id": model_id,
                "configured_revision": config.revision,
                "resolved_revision": getattr(adapter, "resolved_revision", None),
                "loaded": loaded,
                "state": state,
                "adapter": type(adapter).__name__ if adapter is not None else None,
                "inference_readiness": "not_checked",
            }
        return result

    def status(self) -> dict:
        media = self.settings.backend == "opensearch" and isinstance(self.agent, MediaSearchAgent)
        result: dict[str, Any] = {
            "status": "ok",
            "backend": self.settings.backend,
            "mode": self.settings.mode,
            "sample": self.settings.backend == "sample",
            "notice": (
                "직접 만든 가상 샘플 카탈로그입니다. 실제 아이들나라 데이터가 아닙니다."
                if self.settings.backend == "sample"
                else "공개 미디어 메타데이터입니다. 출처·라이선스·연령 근거를 확인하세요."
            ),
            "content_count": None,
            "operation_in_progress": self.agent.lock.locked(),
            "models": self.models(),
            "index": {
                "name": None,
                "state": "not_applicable",
                "exists": None,
                "health": None,
                "document_count": None,
                "schema": None,
                "embedding_identity": None,
                "identity_status": "not_checked",
                "message": "샘플 백엔드는 OpenSearch 인덱스를 사용하지 않습니다.",
            },
            "capabilities": {
                "browse": True,
                "search": True,
                "ingest_preview": True,
                "ingest_execute": media,
                "index_ensure": media,
                "index_refresh": media,
            },
            "limits": {
                "max_batch_size": self.settings.admin_max_batch_size,
                "max_body_bytes": MAX_BODY_BYTES,
                "max_page_size": MAX_PAGE_SIZE,
                "max_result_window": MAX_RESULT_WINDOW,
            },
            "defaults": {
                "top_k": self.settings.candidate_top_k,
                "top_n": self.settings.result_top_n,
            },
            "execution": {"ingest": "synchronous", "media_download": False, "durable_jobs": False},
            "errors": [],
        }
        if not media:
            try:
                if not isinstance(self.agent, SearchAgent):
                    raise TypeError("Missing catalog adapter")
                result["content_count"] = len(self.agent.catalog.all())
            except (ValueError, OSError, TypeError):
                result["status"] = "degraded"
                result["errors"].append(
                    {"code": "CATALOG_UNAVAILABLE", "message": "카탈로그를 읽을 수 없습니다."}
                )
            return result
        store = self.media().store
        index = result["index"]
        index.update(
            name=store.index, state="unavailable", message="인덱스 상태를 확인할 수 없습니다."
        )
        try:
            mapping = store.request("GET", f"/{store.index}/_mapping", allow_not_found=True)
            if mapping is None:
                index.update(
                    state="missing", exists=False, message="설정된 인덱스가 아직 없습니다."
                )
                result["status"] = "degraded"
                return result
            if len(mapping) != 1:
                raise OpenSearchError("Expected one index behind read alias")
            metadata = next(iter(mapping.values()))["mappings"].get("_meta", {})
            identity = metadata.get("embedding_identity")
            index.update(
                state="available",
                exists=True,
                schema=metadata.get("schema"),
                embedding_identity=safe_identity(identity) if isinstance(identity, dict) else None,
                message="인덱스 상태입니다. 현재 모델과의 호환성·추론 준비 여부는 별도 확인이 필요합니다.",
            )
            count = complete_response(
                store.request("POST", f"/{store.index}/_count", body={"query": {"match_all": {}}})
            )
            total = int(count["count"])
            if total < 0:
                raise OpenSearchError("Invalid document count")
            index["document_count"] = result["content_count"] = total
            health = complete_response(
                store.request("GET", f"/_cluster/health/{store.index}?timeout=2s")
            )
            color = health["status"]
            if color not in {"green", "yellow", "red"}:
                raise OpenSearchError("Invalid index health")
            index["health"] = color
            if color != "green":
                result["status"] = "degraded"
        except (OpenSearchError, KeyError, TypeError, ValueError, AttributeError):
            result["status"] = "degraded"
            index.update(state="unavailable", message="OpenSearch 연결·인덱스·응답을 확인하세요.")
            result["errors"].append(
                {
                    "code": "OPENSEARCH_UNAVAILABLE",
                    "message": "OpenSearch 상태 조회에 실패했습니다.",
                }
            )
        return result

    def contents(self, query: str, offset: int, limit: int) -> dict:
        if offset + limit > MAX_RESULT_WINDOW:
            raise problem(
                422, "RESULT_WINDOW_EXCEEDED", "조회 범위는 처음 10,000건 이내여야 합니다."
            )
        if self.settings.backend == "sample":
            if not isinstance(self.agent, SearchAgent):
                raise problem(503, "CATALOG_UNAVAILABLE", "카탈로그 어댑터를 확인하세요.")
            matches = [
                item
                for item in self.agent.catalog.all()
                if not query or query.casefold() in f"{item.id} {item.passage()}".casefold()
            ]
            total = len(matches)
            items = [sample_item(item) for item in matches[offset : offset + limit]]
        else:
            store = self.media().store
            response = complete_response(
                store.request(
                    "POST",
                    f"/{store.index}/_search",
                    body={
                        "from": offset,
                        "size": limit,
                        "track_total_hits": True,
                        "_source": {
                            "excludes": [
                                "embedding",
                                "search_text",
                                "local_path",
                                "subtitle_text",
                                "source_metadata",
                            ]
                        },
                        "query": {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "title^3",
                                    "description",
                                    "tags",
                                    "characters",
                                    "source_id",
                                    "content_id",
                                ],
                            }
                        }
                        if query
                        else {"match_all": {}},
                        "sort": [{"content_id": "asc"}],
                    },
                )
            )
            try:
                hits = response["hits"]
                found = hits["total"]
                if isinstance(found, dict):
                    if found.get("relation", "eq") != "eq":
                        raise OpenSearchError("Exact result count unavailable")
                    found = found["value"]
                total = int(found)
                if total < 0 or len(hits["hits"]) > limit:
                    raise OpenSearchError("Invalid result page")
                items = []
                for hit in hits["hits"]:
                    raw = dict(hit["_source"])
                    content_id = raw.pop("content_id")
                    raw.pop("embedding", None)
                    raw.pop("search_text", None)
                    record = MediaRecord.model_validate(raw)
                    if content_id != record.content_id or hit["_id"] != content_id:
                        raise OpenSearchError("Inconsistent content identity")
                    items.append(media_item(record))
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise OpenSearchError("Malformed content response") from exc
        return {
            "backend": self.settings.backend,
            "sample": self.settings.backend == "sample",
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": items,
            "has_more": offset + len(items) < min(total, MAX_RESULT_WINDOW),
            "window_limited": total > MAX_RESULT_WINDOW,
        }

    def search(self, request: SearchRequest):
        with self.operation():
            # Both existing public search() methods take the same non-reentrant lock.
            return self.agent._search(request)

    def ingest(self, request: IngestRequest) -> dict:
        from .media.ingest import Ingestor

        if len(request.manifest.entries) > self.settings.admin_max_batch_size:
            raise problem(422, "BATCH_TOO_LARGE", "설정된 최대 적재 건수를 초과했습니다.")
        if request.dry_run:
            result = Ingestor(self.media_settings or MediaSettings()).run(request.manifest)
        else:
            self.confirm(request)
            agent = self.media()
            with self.operation():
                try:
                    result = Ingestor(
                        self.media_settings or MediaSettings(),
                        embedding=agent.embedding,
                        index=agent.store,
                    ).run(request.manifest, dry_run=False)
                except ValueError as exc:
                    if str(exc) == "Another ingestion is running for this MEDIA_ROOT":
                        raise problem(409, "ADMIN_BUSY", "다른 적재 작업이 진행 중입니다.") from exc
                    raise
        return dict(result, backend=self.settings.backend, may_load_model=not request.dry_run)

    def ensure(self, request: ConfirmedAction) -> dict:
        self.confirm(request)
        agent = self.media()
        with self.operation():
            identity = agent.embedding.identity
            agent.store.ensure_index(identity)
        return {
            "status": "ready",
            "index": agent.store.index,
            "embedding_identity": safe_identity(identity),
            "model_inference_performed": False,
            "may_load_model": True,
        }

    def refresh(self, request: ConfirmedAction) -> dict:
        self.confirm(request)
        agent = self.media()
        with self.operation():
            response = complete_response(
                agent.store.request("POST", f"/{agent.store.index}/_refresh")
            )
            shards = response.get("_shards")
            if (
                not isinstance(shards, dict)
                or type(shards.get("failed")) is not int
                or shards["failed"] != 0
                or type(shards.get("successful")) is not int
                or shards["successful"] < 1
            ):
                raise OpenSearchError("Missing successful index refresh acknowledgement")
        return {
            "status": "refreshed",
            "index": agent.store.index,
            "model_inference_performed": False,
            "may_load_model": False,
        }


def create_admin_router(
    settings: Settings,
    agent: SearchAgent | MediaSearchAgent,
    *,
    media_settings: MediaSettings | None = None,
) -> APIRouter:
    service = AdminService(settings, agent, media_settings)
    router = APIRouter(
        prefix="/api/admin",
        tags=["admin"],
        route_class=admin_route_class(settings),
        dependencies=[Depends(HTTPBearer(auto_error=False))],
    )

    @router.get("/status")
    def status():
        return service.status()

    @router.get("/contents")
    def contents(
        q: str = Query(default="", max_length=500),
        offset: int = Query(default=0, ge=0, le=MAX_RESULT_WINDOW - 1),
        limit: int = Query(default=20, ge=1, le=MAX_PAGE_SIZE),
    ):
        return service.contents(q.strip(), offset, limit)

    @router.post("/search")
    def search(request: SearchRequest):
        return service.search(request)

    @router.post("/ingest")
    def ingest(request: IngestRequest):
        return service.ingest(request)

    @router.post("/index/ensure")
    def ensure(request: ConfirmedAction):
        return service.ensure(request)

    @router.post("/index/refresh")
    def refresh(request: ConfirmedAction):
        return service.refresh(request)

    return router
