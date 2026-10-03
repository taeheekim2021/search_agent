"""Independent lazy adapters. Errors never switch models or select mock mode."""

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .config import BGE_ID, QWEN_ID, ModelConfig

QUERY_INSTRUCTION = (
    "Instruct: Given a Korean children's content search query, retrieve relevant "
    "content descriptions matching the requested topic and character.\nQuery: "
)


class ModelError(RuntimeError):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"{stage}: {message}")


class EmbeddingAdapter(Protocol):
    @property
    def identity(self) -> dict[str, Any]: ...
    def documents(self, texts: list[str]) -> NDArray: ...
    def query(self, text: str) -> NDArray: ...


class RerankerAdapter(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


def normalize(vectors: Any, rows: int, dimensions: int) -> NDArray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.shape != (rows, dimensions) or not np.isfinite(arr).all():
        raise ValueError(f"Expected finite vectors {(rows, dimensions)}, got {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero embedding vector")
    return arr / norms


class TransformerAdapter:
    model_id: str
    stage: str

    def __init__(self, config: ModelConfig, cache_dir: str):
        self.config = config
        self.cache_dir = cache_dir
        self.model: Any = None
        self.tokenizer: Any = None
        self.resolved_revision: str | None = None

    def _load(self):
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

            factory = AutoModel if self.stage == "embedding" else AutoModelForSequenceClassification
            kwargs = {
                "revision": self.config.revision,
                "cache_dir": self.cache_dir,
                "trust_remote_code": False,
            }
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                padding_side="left" if self.stage == "embedding" else "right",
                **kwargs,
            )
            model = (
                factory.from_pretrained(
                    self.model_id,
                    torch_dtype=getattr(torch, self.config.dtype),
                    low_cpu_mem_usage=True,
                    **kwargs,
                )
                .to(self.config.device)
                .eval()
            )
            revision = getattr(model.config, "_commit_hash", None)
            if not revision:
                raise ValueError("Cannot resolve immutable model revision for cache identity")
            self.resolved_revision = revision
            self.tokenizer, self.model = tokenizer, model
        except Exception as exc:
            raise ModelError(self.stage, f"{self.model_id} loading failed: {exc}") from exc


class QwenEmbedding(TransformerAdapter):
    model_id = QWEN_ID
    stage = "embedding"

    def __init__(self, config: ModelConfig, cache_dir: str, dimensions: int = 2560):
        super().__init__(config, cache_dir)
        if not 32 <= dimensions <= 2560:
            raise ValueError("Qwen dimensions must be 32..2560")
        self.dimensions = dimensions

    @property
    def identity(self) -> dict[str, Any]:
        self._load()
        return {
            "model": self.model_id,
            "revision": self.config.revision,
            "resolved_revision": self.resolved_revision,
            "dimensions": self.dimensions,
            "max_length": self.config.max_length,
            "dtype": self.config.dtype,
            "device": self.config.device,
            "pooling": "last_token_v1",
        }

    def _encode(self, texts: list[str]) -> NDArray:
        self._load()
        try:
            import torch

            batches = []
            for start in range(0, len(texts), self.config.batch_size):
                inputs = self.tokenizer(
                    texts[start : start + self.config.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                ).to(self.config.device)
                with torch.inference_mode():
                    # Left padding guarantees each final position is the last real token.
                    hidden = self.model(**inputs).last_hidden_state[:, -1, :]
                    if hidden.shape[1] != 2560:
                        raise ValueError(f"Unexpected Qwen output dimension: {hidden.shape[1]}")
                    batches.append(hidden[:, : self.dimensions].float().cpu().numpy())
            return normalize(np.concatenate(batches), len(texts), self.dimensions)
        except Exception as exc:
            raise ModelError(self.stage, f"{self.model_id} inference failed: {exc}") from exc

    def documents(self, texts: list[str]) -> NDArray:
        return self._encode(texts)

    def query(self, text: str) -> NDArray:
        return self._encode([QUERY_INSTRUCTION + text])[0]


class BGEReranker(TransformerAdapter):
    model_id = BGE_ID
    stage = "reranker"

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        self._load()
        try:
            import torch

            scores: list[float] = []
            for start in range(0, len(passages), self.config.batch_size):
                pairs = [[query, text] for text in passages[start : start + self.config.batch_size]]
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                ).to(self.config.device)
                with torch.inference_mode():
                    logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
                scores.extend(logits.cpu().tolist())
            if len(scores) != len(passages) or not np.isfinite(scores).all():
                raise ValueError("Invalid cross-encoder scores")
            return scores
        except Exception as exc:
            raise ModelError(self.stage, f"{self.model_id} inference failed: {exc}") from exc
