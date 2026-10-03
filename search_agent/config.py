from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

QWEN_ID = "Qwen/Qwen3-Embedding-4B"
BGE_ID = "BAAI/bge-reranker-v2-m3"


class ModelConfig(BaseModel):
    revision: str = Field(default="main", min_length=1)
    device: Literal["cpu", "cuda"] = "cpu"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    batch_size: int = Field(default=1, ge=1, le=32)
    max_length: int = Field(default=512, ge=32, le=8192)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SEARCH_", env_nested_delimiter="__", env_file=".env", extra="ignore"
    )
    mode: Literal["real", "mock"] = "real"
    catalog_path: Path = Path("data/sample_catalog.json")
    cache_dir: Path = Path(".cache/embeddings")
    hf_cache_dir: Path = Path(".cache/huggingface")
    embedding: ModelConfig = Field(default_factory=ModelConfig)
    reranker: ModelConfig = Field(default_factory=lambda: ModelConfig(max_length=512))
    dimensions: int = Field(default=2560, ge=32, le=2560)
    candidate_top_k: int = Field(default=20, ge=1, le=100)
    result_top_n: int = Field(default=5, ge=1, le=100)

    @model_validator(mode="after")
    def limits(self):
        if self.result_top_n > self.candidate_top_k:
            raise ValueError("result_top_n must be <= candidate_top_k")
        if self.reranker.max_length > 512:
            raise ValueError("BGE max_length must be <= 512 in this implementation")
        return self
