from threading import Lock
from time import perf_counter

import numpy as np

from search_agent.config import Settings
from search_agent.domain import SearchRequest
from search_agent.extraction import extract_vocabulary
from search_agent.models import EmbeddingAdapter, ModelError, RerankerAdapter
from search_agent.retrieval import fuse

from .domain import MediaSearchResponse, MediaSearchResult
from .opensearch import OpenSearchStore, bm25_query, vector_query


class MediaSearchAgent:
    def __init__(
        self,
        settings: Settings,
        store: OpenSearchStore,
        embedding: EmbeddingAdapter,
        reranker: RerankerAdapter,
    ):
        if settings.mode != "real" or settings.dimensions != 2560:
            raise ValueError("OpenSearch media search requires real mode and 2560 dimensions")
        self.settings, self.store = settings, store
        self.embedding, self.reranker = embedding, reranker
        self.lock = Lock()

    def search(self, request: SearchRequest) -> MediaSearchResponse:
        with self.lock:
            return self._search(request)

    def _search(self, request: SearchRequest) -> MediaSearchResponse:
        start = perf_counter()
        k = request.top_k if request.top_k is not None else self.settings.candidate_top_k
        n = request.top_n if request.top_n is not None else self.settings.result_top_n
        if n > k:
            raise ValueError("top_n must be <= top_k")
        store = self.store.snapshot() if isinstance(self.store, OpenSearchStore) else self.store
        tags, characters = store.vocabulary()
        conditions = extract_vocabulary(request.query, tags, characters)
        if store.count(conditions) == 0:
            return MediaSearchResponse(
                conditions=conditions,
                results=[],
                candidate_count=0,
                model_inference_performed=False,
                elapsed_ms=round((perf_counter() - start) * 1000, 2),
            )
        store.verify_identity(self.embedding.identity)
        try:
            query_vector = self.embedding.query(request.query)
            semantic = store.search(vector_query(query_vector, conditions, k))
        except ValueError as exc:
            raise ModelError("embedding", "Invalid query vector") from exc
        lexical = store.search(bm25_query(request.query, conditions, k))
        by_id = {r.content_id: r for r in semantic + lexical}
        lexical_ids = [r.content_id for r in lexical]
        semantic_ids = [r.content_id for r in semantic]
        candidates = fuse([lexical_ids, semantic_ids], k)
        try:
            scores = self.reranker.score(
                request.query, [by_id[key].passage() for key, _ in candidates]
            )
            if len(scores) != len(candidates) or not np.isfinite(scores).all():
                raise ValueError("Invalid reranker scores")
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError("reranker", "Media reranking failed") from exc
        ranked = sorted(zip(candidates, scores), key=lambda item: (-item[1], item[0][0]))
        results = []
        for (key, fusion), score in ranked[:n]:
            record = by_id[key]
            evidence = [
                "연령등급 미상"
                if record.min_age is None
                else f"출처 연령 범위 {record.min_age}~{record.max_age}세",
                f"원본 언어: {record.language}",
                f"주제: {', '.join(record.tags)}",
                f"저작자: {record.author}",
                (
                    f"메타데이터 라이선스: {record.license} (영상 이용허락 미확인)"
                    if record.rights_scope == "metadata"
                    else f"라이선스: {record.license}"
                ),
                f"출처: {record.canonical_url}",
                f"표시 문구: {record.attribution}",
            ]
            if record.duration_seconds is not None:
                evidence.append(f"출처 길이: {record.duration_seconds}초")
            for field, provenance in record.metadata_provenance.items():
                if provenance.method != "source":
                    evidence.append(f"{field}: {provenance.method} — {provenance.note}")
            results.append(
                MediaSearchResult(
                    content_id=key,
                    content=record,
                    evidence=evidence,
                    retrieval_sources=(["keyword"] if key in lexical_ids else [])
                    + (["semantic"] if key in semantic_ids else []),
                    fusion_score=fusion,
                    reranker_score=score,
                )
            )
        return MediaSearchResponse(
            conditions=conditions,
            results=results,
            candidate_count=len(candidates),
            model_inference_performed=True,
            elapsed_ms=round((perf_counter() - start) * 1000, 2),
        )
