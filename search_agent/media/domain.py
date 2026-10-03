import hashlib
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from search_agent.domain import Conditions


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["source", "human_translation", "assistant_translation", "editorial", "unknown"]
    source_url: str | None = None
    note: str = Field(min_length=1, max_length=2000)


def public_https_url(value: str) -> str:
    """Syntax only. Download policy separately checks allowlist and public DNS at each hop."""
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
        or parsed.port not in (None, 443)
        or any(ord(c) <= 32 or ord(c) == 127 for c in value)
        or "\\" in value
    ):
        raise ValueError("Expected an HTTPS URL without credentials, fragments, or custom ports")
    return value


class MediaEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    canonical_url: str = Field(max_length=4096)
    original_url: str = Field(max_length=4096)
    title: str = Field(min_length=1, max_length=500)
    description: str = Field(min_length=1, max_length=10000)
    tags: list[str] = Field(default_factory=list, max_length=100)
    characters: list[str] = Field(default_factory=list, max_length=100)
    language: str = Field(pattern=r"^[a-z]{2,3}(?:-[A-Za-z0-9]+)*$", default="und")
    media_format: Literal["webm", "ogv", "mp4", "ogg", "mp3", "wav", "flac"]
    duration_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    author: str = Field(min_length=1, max_length=1000)
    license: Literal[
        "CC0-1.0",
        "CC-BY-4.0",
        "CC-BY-SA-4.0",
        "CC-BY-3.0",
        "CC-BY-SA-3.0",
        "Public-Domain",
        "PD-USGov-NASA",
    ]
    license_url: str
    attribution: str = Field(min_length=1, max_length=4000)
    # Explicit review assertion, never inferred from a hostname or a search snippet.
    rights_verified: Literal[True]
    metadata_provenance: dict[str, Provenance]
    license_notes: str = Field(default="", max_length=4000)
    metadata_license: str = Field(default="", max_length=500)
    source_metadata: dict = Field(default_factory=dict)
    min_age: int | None = Field(default=None, ge=0, le=18)
    max_age: int | None = Field(default=None, ge=0, le=18)
    age_rating_source: str | None = None
    subtitle_text: str | None = Field(default=None, max_length=100000)
    subtitle_source_url: str | None = None
    subtitle_license: str | None = Field(default=None, max_length=500)
    expected_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @field_validator("canonical_url", "original_url", "license_url")
    @classmethod
    def urls(cls, value):
        return public_https_url(value)

    @field_validator("age_rating_source", "subtitle_source_url")
    @classmethod
    def optional_urls(cls, value):
        return public_https_url(value) if value else value

    @field_validator("tags", "characters")
    @classmethod
    def terms(cls, values):
        if any(not v or len(v) > 100 for v in values):
            raise ValueError("Tags/characters must have 1..100 characters")
        return sorted(set(values))

    @model_validator(mode="after")
    def provenance(self):
        required = {"title", "description", "tags", "language", "duration_seconds"}
        if required - self.metadata_provenance.keys():
            raise ValueError(
                "Metadata provenance required for title/description/tags/language/duration_seconds"
            )
        if (self.min_age is None) != (self.max_age is None):
            raise ValueError("Both age bounds must be known or both null")
        if self.min_age is not None:
            if self.max_age is None or self.min_age > self.max_age or not self.age_rating_source:
                raise ValueError("Known age bounds require a real rating source and valid range")
        elif self.age_rating_source:
            raise ValueError("Unknown age bounds cannot claim an age rating source")
        if self.subtitle_text and not (self.subtitle_source_url and self.subtitle_license):
            raise ValueError("Subtitles require an explicit source and reuse license")
        return self

    @property
    def content_id(self) -> str:
        # Stable across edits/download rendition changes. Use one canonical source URL per work.
        return "media-" + hashlib.sha256(self.canonical_url.encode()).hexdigest()

    def passage(self) -> str:
        # Metadata text only, never feed media bytes to a text embedding model.
        return f"{self.title}. {self.description} 주제: {', '.join(self.tags)}. " + (
            f"캐릭터: {', '.join(self.characters)}. {self.subtitle_text or ''}"
        )


class MediaManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1]
    entries: list[MediaEntry] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def deduplicate(self):
        unique: dict[str, MediaEntry] = {}
        for entry in self.entries:
            if entry.content_id in unique and unique[entry.content_id] != entry:
                raise ValueError("Conflicting entries for the same canonical URL")
            unique[entry.content_id] = entry
        self.entries = list(unique.values())
        return self


class MediaRecord(MediaEntry):
    checksum_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    local_path: str
    size_bytes: int = Field(gt=0)
    retrieved_at: str


class MediaSearchResult(BaseModel):
    content_id: str
    content: MediaRecord
    evidence: list[str]
    retrieval_sources: list[str]
    fusion_score: float
    reranker_score: float


class MediaSearchResponse(BaseModel):
    mode: Literal["real"] = "real"
    backend: Literal["opensearch"] = "opensearch"
    sample_notice: str = (
        "공개 미디어입니다. 원본 언어·출처·라이선스와 메타데이터 provenance를 확인하세요."
    )
    model_inference_performed: bool
    conditions: Conditions
    candidate_count: int
    results: list[MediaSearchResult]
    elapsed_ms: float
    notice: str = "연령 미상 자료는 나이를 지정한 검색에서 제외됩니다. 점수는 확률이 아닙니다."
