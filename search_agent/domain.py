from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Content(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    min_age: int = Field(ge=0, le=18)
    max_age: int = Field(ge=0, le=18)
    topics: list[str]
    characters: list[str]
    duration_minutes: int = Field(gt=0)
    is_sample: Literal[True] = True

    @model_validator(mode="after")
    def ages(self):
        if self.min_age > self.max_age:
            raise ValueError("min_age exceeds max_age")
        return self

    def passage(self) -> str:
        return (
            f"{self.title}. {self.description} "
            f"주제: {', '.join(self.topics)}. 캐릭터: {', '.join(self.characters)}. "
            f"권장 나이: {self.min_age}~{self.max_age}세."
        )


class Conditions(BaseModel):
    age: int | None = None
    topics: list[str] = Field(default_factory=list)
    characters: list[str] = Field(default_factory=list)
    extraction_method: str = "catalog_dictionary_and_age_rules"


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)
    query: str = Field(min_length=1, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=100)
    top_n: int | None = Field(default=None, ge=1, le=100)


class SearchResult(BaseModel):
    content: Content
    evidence: list[str]
    retrieval_sources: list[str]
    fusion_score: float | None = None
    reranker_score: float | None = None


class SearchResponse(BaseModel):
    mode: Literal["real", "mock"]
    sample_notice: str = "모든 콘텐츠·캐릭터는 직접 만든 가상 샘플입니다."
    model_inference_performed: bool
    conditions: Conditions
    candidate_count: int
    results: list[SearchResult]
    elapsed_ms: float
    notice: str
