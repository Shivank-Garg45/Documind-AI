"""Application configuration.

All settings are read from environment variables (or a local ``.env`` file)
via ``pydantic-settings`` so that nothing sensitive is ever hardcoded.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project base dir = backend/   (this file is backend/app/config.py)
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed, validated application settings."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM ----
    # Which provider to use for answer generation: "gemini" (free tier) or "openai".
    llm_provider: str = Field(default="gemini", alias="LLM_PROVIDER")
    llm_timeout_seconds: float = Field(default=30.0, alias="LLM_TIMEOUT_SECONDS")

    # OpenAI
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    # Google Gemini
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-3.1-flash-lite", alias="GEMINI_MODEL")

    # ---- Embeddings ----
    embedding_backend: str = Field(
        default="sentence_transformers", alias="EMBEDDING_BACKEND"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )

    # ---- Retrieval / chunking ----
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")

    # ---- Uploads ----
    max_upload_mb: int = Field(default=10, alias="MAX_UPLOAD_MB")

    # ---- Storage ----
    chroma_dir: str = Field(default="./data/chroma", alias="CHROMA_DIR")
    upload_dir: str = Field(default="./data/uploads", alias="UPLOAD_DIR")

    # ---- CORS ----
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        alias="CORS_ORIGINS",
    )

    # --- Derived helpers -------------------------------------------------
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def chroma_path(self) -> Path:
        p = Path(self.chroma_dir)
        return p if p.is_absolute() else (BASE_DIR / p)

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        return p if p.is_absolute() else (BASE_DIR / p)

    @property
    def llm_enabled(self) -> bool:
        """True only when the active provider has a non-placeholder API key."""
        if self.llm_provider.lower() == "openai":
            key = (self.openai_api_key or "").strip()
            return bool(key) and key != "sk-your-key-here"
        key = (self.google_api_key or "").strip()
        return bool(key) and key != "your-google-api-key-here"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton (safe to call from anywhere)."""
    return Settings()
