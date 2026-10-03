from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MediaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEDIA_", env_file=".env", extra="ignore")
    root: Path = Path(".media")
    allowed_domains: list[str] = Field(default_factory=list)
    max_bytes: int = Field(default=50 * 1024 * 1024, ge=1, le=1024 * 1024 * 1024)
    total_max_bytes: int = Field(default=160_000_000, ge=1, le=10_000_000_000)
    timeout_seconds: float = Field(default=60, gt=0, le=600)
    retries: int = Field(default=1, ge=0, le=3)
    max_redirects: int = Field(default=3, ge=0, le=5)


class OpenSearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENSEARCH_", env_file=".env", extra="ignore")
    url: str = "https://localhost:9200"
    index: str = Field(default="kids-media-v1", pattern=r"^[a-z0-9][a-z0-9_-]{0,100}$")
    username: str | None = None
    password: SecretStr | None = None
    ca_certs: Path | None = None
    timeout_seconds: float = Field(default=15, gt=0, le=120)
    # Explicit local-only exception. Certificate verification cannot be disabled.
    allow_http_local: bool = False

    @model_validator(mode="after")
    def endpoint(self):
        url = urlsplit(self.url)
        if not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("OpenSearch endpoint must not contain credentials, query, or fragment")
        if url.path not in ("", "/"):
            raise ValueError("OpenSearch URL must be an origin without a path")
        if url.scheme != "https" and not (
            url.scheme == "http"
            and self.allow_http_local
            and url.hostname in ("localhost", "127.0.0.1", "::1")
        ):
            raise ValueError(
                "OpenSearch requires HTTPS; HTTP exception is explicit and loopback-only"
            )
        if bool(self.username) != bool(self.password):
            raise ValueError("Both OpenSearch username and password must be set together")
        return self
