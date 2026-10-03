"""Opt-in REAL inference; no fixture, alternate model, or fallback is used."""

import argparse
import json
from time import perf_counter

import numpy as np

from search_agent.config import Settings
from search_agent.models import BGEReranker, QwenEmbedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-real", action="store_true", help="Allow loading/downloading both models"
    )
    args = parser.parse_args()
    if not args.run_real:
        parser.error("Explicit --run-real required; review RAM/disk and network first")
    settings = Settings(mode="real")
    query = "5살 아이가 볼 공룡 영상"
    passages = [
        "어린이를 위한 공룡 영상. 초식 공룡의 먹이와 발자국을 알아봅니다.",
        "서버 관리자를 위한 데이터베이스 백업 및 장애 복구 강의입니다.",
    ]
    embedding = QwenEmbedding(settings.embedding, str(settings.hf_cache_dir), settings.dimensions)
    reranker = BGEReranker(settings.reranker, str(settings.hf_cache_dir))
    start = perf_counter()
    docs = embedding.documents(passages)
    q = embedding.query(query)
    semantic = docs @ q
    embedding_seconds = perf_counter() - start
    start = perf_counter()
    scores = reranker.score(query, passages)
    result = {
        "mode": "real",
        "query": query,
        "passages": passages,
        "embedding_identity": embedding.identity,
        "reranker_model": reranker.model_id,
        "reranker_revision": reranker.resolved_revision,
        "dimensions": docs.shape[1],
        "document_norms": np.linalg.norm(docs, axis=1).tolist(),
        "semantic_scores": semantic.tolist(),
        "reranker_raw_scores": scores,
        "embedding_seconds_including_load": embedding_seconds,
        "reranker_seconds_including_load": perf_counter() - start,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert docs.shape == (2, settings.dimensions)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1, atol=1e-5)
    assert semantic[0] > semantic[1], "Related passage did not rank first in Qwen smoke test"
    assert scores[0] > scores[1], "Related passage did not rank first in BGE smoke test"


if __name__ == "__main__":
    main()
