import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .admin import create_admin_router
from .catalog import JsonCatalog
from .config import BGE_ID, QWEN_ID, Settings
from .domain import SearchRequest, SearchResponse
from .media.config import OpenSearchSettings
from .media.domain import MediaSearchResponse
from .media.opensearch import OpenSearchError, OpenSearchStore
from .media.service import MediaSearchAgent
from .models import BGEReranker, ModelError, QwenEmbedding
from .retrieval import EmbeddingCache, SemanticSearch
from .service import SearchAgent


def create_app(
    settings: Settings | None = None, agent: SearchAgent | MediaSearchAgent | None = None
) -> FastAPI:
    config = settings or Settings()
    if agent is None and config.backend == "opensearch":
        agent = MediaSearchAgent(
            config,
            OpenSearchStore(OpenSearchSettings.for_search(config)),
            QwenEmbedding(config.embedding, str(config.hf_cache_dir), 2560),
            BGEReranker(config.reranker, str(config.hf_cache_dir)),
        )
    if agent is None:
        semantic = None
        reranker = None
        if config.mode == "real":
            semantic = SemanticSearch(
                QwenEmbedding(config.embedding, str(config.hf_cache_dir), config.dimensions),
                EmbeddingCache(config.cache_dir),
            )
            reranker = BGEReranker(config.reranker, str(config.hf_cache_dir))
        agent = SearchAgent(config, JsonCatalog(config.catalog_path), semantic, reranker)
    app = FastAPI(title="아이들나라 SearchAgent · 콘텐츠 검색", version="0.2.0")
    static_dir = Path(__file__).parent / "static"

    @app.middleware("http")
    async def admin_response_headers(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/admin" or request.url.path.startswith(("/admin/", "/api/admin/")):
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Content-Security-Policy"] = (
                "default-src 'none'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; img-src 'self' data:; font-src 'self'; "
                "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
            )
        return response

    app.include_router(create_admin_router(config, agent))
    app.mount("/admin/assets", StaticFiles(directory=static_dir), name="admin-assets")

    @app.get("/admin", include_in_schema=False)
    @app.get("/admin/", include_in_schema=False)
    def admin_index():
        return FileResponse(static_dir / "admin.html")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "mode": config.mode,
            "sample": config.backend == "sample",
            "backend": config.backend,
            "catalog_notice": (
                "모든 콘텐츠와 캐릭터는 직접 만든 가상 샘플입니다. 실제 아이들나라 카탈로그가 아닙니다."
                if config.backend == "sample"
                else "공개 미디어 검색입니다. 한국어 제목은 번역 메타데이터일 수 있습니다. 연령 미상 자료는 나이 검색에서 제외됩니다."
            ),
            "models": {"embedding": QWEN_ID, "reranker": BGE_ID},
            "model_readiness": "not_checked",
            "default_top_k": config.candidate_top_k,
            "default_top_n": config.result_top_n,
        }

    @app.post("/api/search", response_model=SearchResponse | MediaSearchResponse)
    def search(request: SearchRequest):
        try:
            return agent.search(request)
        except OpenSearchError as exc:
            raise HTTPException(
                503, detail={"code": "OPENSEARCH_UNAVAILABLE", "message": str(exc)}
            ) from exc
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        except ModelError as exc:
            logging.getLogger(__name__).error("Model stage failed: %s", exc.stage)
            raise HTTPException(
                503,
                detail={
                    "code": "MODEL_UNAVAILABLE",
                    "stage": exc.stage,
                    "message": "실제 모델 로딩/추론 실패. 서버 로그와 모델 캐시·메모리를 확인하세요. "
                    "모의 모드로 자동 전환하지 않았습니다.",
                },
            ) from exc

    return app


app = create_app()
