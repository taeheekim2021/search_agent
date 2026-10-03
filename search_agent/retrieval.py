import hashlib
import json
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .domain import Content
from .models import EmbeddingAdapter, ModelError, normalize


class KeywordAdapter(Protocol):
    """An external search engine can implement this ranking contract."""

    def rank(self, query: str, items: list[Content], top_k: int) -> list[str]: ...


class KeywordSearch:
    def rank(self, query: str, items: list[Content], top_k: int) -> list[str]:
        tokens = set(re.findall(r"[가-힣A-Za-z0-9]+", query.lower()))
        # Substring metadata matches accommodate common Korean particles.
        ranked = []
        for item in items:
            text = item.passage().lower()
            terms = set(item.topics + item.characters + re.findall(r"[가-힣A-Za-z]+", item.title))
            score = sum(token in text for token in tokens) + sum(t in query for t in terms)
            if score:
                ranked.append((item.id, score))
        return [key for key, _ in sorted(ranked, key=lambda pair: (-pair[1], pair[0]))[:top_k]]


def fuse(rankings: list[list[str]], top_k: int) -> list[tuple[str, float]]:
    """Equal-weight reciprocal rank fusion; one vote per unique id per source."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(dict.fromkeys(ranking), start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]


class EmbeddingDocument(Protocol):
    def model_dump(self) -> dict: ...
    def passage(self) -> str: ...


class EmbeddingCache:
    def __init__(self, path: Path):
        self.path = path

    def key(self, identity: dict, items: Sequence[EmbeddingDocument]) -> str:
        payload = {"schema": 1, "adapter": identity, "data": [item.model_dump() for item in items]}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    def documents(self, adapter: EmbeddingAdapter, items: Sequence[EmbeddingDocument]) -> NDArray:
        identity = adapter.identity  # Resolve actual model commit before accepting cached vectors.
        dimensions = identity["dimensions"]
        self.path.mkdir(parents=True, exist_ok=True)
        target = self.path / (self.key(identity, items) + ".npy")
        if target.exists():
            try:
                return normalize(np.load(target, allow_pickle=False), len(items), dimensions)
            except (OSError, ValueError):
                pass  # Recompute invalid cache using the same real model, never another method.
        vectors = normalize(adapter.documents([c.passage() for c in items]), len(items), dimensions)
        with tempfile.NamedTemporaryFile(dir=self.path, suffix=".npy", delete=False) as file:
            temp = Path(file.name)
            try:
                np.save(file, vectors, allow_pickle=False)
                file.flush()
                os.fsync(file.fileno())
                os.replace(temp, target)
            finally:
                temp.unlink(missing_ok=True)
        return vectors


class SemanticSearch:
    def __init__(self, adapter: EmbeddingAdapter, cache: EmbeddingCache):
        self.adapter, self.cache = adapter, cache

    def rank(self, query: str, items: list[Content], top_k: int) -> list[str]:
        try:
            vectors = self.cache.documents(self.adapter, items)
            q = normalize([self.adapter.query(query)], 1, vectors.shape[1])[0]
            scores = vectors @ q  # Unit vectors: inner product equals cosine similarity.
            return [
                items[i].id
                for i in sorted(range(len(items)), key=lambda i: (-float(scores[i]), items[i].id))[
                    :top_k
                ]
            ]
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError("embedding", f"Semantic search failed: {exc}") from exc
