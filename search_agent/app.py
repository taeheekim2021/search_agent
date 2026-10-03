import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .catalog import JsonCatalog
from .config import BGE_ID, QWEN_ID, Settings
from .domain import SearchRequest, SearchResponse
from .models import BGEReranker, ModelError, QwenEmbedding
from .retrieval import EmbeddingCache, SemanticSearch
from .service import SearchAgent


def create_app(settings: Settings | None = None, agent: SearchAgent | None = None) -> FastAPI:
    config = settings or Settings()
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
    app = FastAPI(title="아이들나라 SearchAgent · 가상 샘플", version="0.1.0")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "mode": config.mode,
            "sample": True,
            "models": {"embedding": QWEN_ID, "reranker": BGE_ID},
            "model_readiness": "not_checked",
            "default_top_k": config.candidate_top_k,
            "default_top_n": config.result_top_n,
        }

    @app.post("/api/search", response_model=SearchResponse)
    def search(request: SearchRequest):
        try:
            return agent.search(request)
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        except ModelError as exc:
            logging.getLogger(__name__).exception("Model stage failed: %s", exc.stage)
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
