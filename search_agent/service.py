from threading import Lock
from time import perf_counter

import numpy as np

from .catalog import CatalogRepository
from .config import Settings
from .domain import SearchRequest, SearchResponse, SearchResult
from .extraction import eligible, extract
from .models import ModelError, RerankerAdapter
from .retrieval import KeywordAdapter, KeywordSearch, SemanticSearch, fuse


class SearchAgent:
    def __init__(
        self,
        settings: Settings,
        catalog: CatalogRepository,
        semantic: SemanticSearch | None = None,
        reranker: RerankerAdapter | None = None,
        keyword: KeywordAdapter | None = None,
    ):
        self.settings, self.catalog = settings, catalog
        self.semantic, self.reranker = semantic, reranker
        self.keyword = keyword or KeywordSearch()
        # Single-process demo: serialize model initialization/inference to bound memory use.
        self.lock = Lock()

    def search(self, request: SearchRequest) -> SearchResponse:
        with self.lock:
            return self._search(request)

    def _search(self, request: SearchRequest) -> SearchResponse:
        start = perf_counter()
        k = request.top_k if request.top_k is not None else self.settings.candidate_top_k
        n = request.top_n if request.top_n is not None else self.settings.result_top_n
        if n > k:
            raise ValueError("top_n must be <= top_k (including configured defaults)")
        items = self.catalog.all()
        conditions = extract(request.query, items)
        items = [item for item in items if eligible(item, conditions)]
        by_id = {item.id: item for item in items}
        lexical = self.keyword.rank(request.query, items, k)
        semantic: list[str] = []
        performed = False
        ranked: list[tuple[tuple[str, float], float | None]]
        if self.settings.mode == "real" and items:
            if self.semantic is None or self.reranker is None:
                raise ModelError("configuration", "Real mode requires both model adapters")
            semantic = self.semantic.rank(request.query, items, k)
            fused = fuse([lexical, semantic], k)
            passages = [by_id[key].passage() for key, _ in fused]
            try:
                scores = self.reranker.score(request.query, passages)
                if len(scores) != len(fused) or not np.isfinite(scores).all():
                    raise ValueError("Reranker returned invalid score count or nonfinite scores")
            except ModelError:
                raise
            except Exception as exc:
                raise ModelError("reranker", str(exc)) from exc
            scored = list(zip(fused, scores))
            scored.sort(key=lambda pair: (-pair[1], pair[0][0]))
            ranked = [(pair, score) for pair, score in scored]
            performed = True
        else:
            fused = fuse([lexical], k)
            ranked = [(pair, None) for pair in fused]
        results = []
        for (key, fusion_score), score in ranked[:n]:
            item = by_id[key]
            evidence = [
                f"권장 연령 {item.min_age}~{item.max_age}세",
                f"주제: {', '.join(item.topics)}",
                f"캐릭터: {', '.join(item.characters)}",
                f"길이: {item.duration_minutes}분",
            ]
            if conditions.age is not None:
                evidence.append(f"요청 나이 {conditions.age}세가 권장 범위에 포함됨")
            results.append(
                SearchResult(
                    content=item,
                    evidence=evidence,
                    retrieval_sources=(["keyword"] if key in lexical else [])
                    + (["semantic"] if key in semantic else []),
                    fusion_score=fusion_score if self.settings.mode == "real" else None,
                    reranker_score=score,
                )
            )
        return SearchResponse(
            mode=self.settings.mode,
            model_inference_performed=performed,
            conditions=conditions,
            candidate_count=len(fused),
            results=results,
            elapsed_ms=round((perf_counter() - start) * 1000, 2),
            notice=(
                "모의 모드: 키워드 UI 데모만 실행. 의미검색·리랭커 미실행, 모델 점수 없음."
                if self.settings.mode == "mock"
                else "BGE 원시 관련도 점수는 확률이 아닙니다. 근거는 카탈로그 메타데이터입니다. "
                "조건에 맞는 항목이 없으면 모델을 실행하지 않습니다."
            ),
        )
