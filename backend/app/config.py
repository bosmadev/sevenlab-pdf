"""Centralised settings. All secrets flow from `.env` only."""
from functools import lru_cache
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Service
    app_env: str = Field(default="development")
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8700)
    workers: int = Field(default=4)

    # Auth
    app_password_hash: str = Field(default="")
    session_secret: str = Field(default="")
    # `.env.sample` documents this as SESSION_MAX_AGE; the longer name is kept for compatibility.
    session_max_age_seconds: int = Field(
        default=7 * 24 * 3600,
        validation_alias=AliasChoices("SESSION_MAX_AGE", "SESSION_MAX_AGE_SECONDS"),
    )

    # Databases
    mongodb_uri: str = Field(default="")
    mongodb_db_name: str = Field(default="sevenlab_pdf")
    postgresql_uri: str = Field(default="")

    # Azure AI
    azure_ai_api_base: str = Field(default="https://15-ai1-swe.services.ai.azure.com")
    azure_ai_api_key: str = Field(default="")

    # Upload limits
    max_pdf_size_mb: int = Field(default=50)
    max_pdf_pages: int = Field(default=500)

    # Embeddings
    embedding_model: str = Field(default="Cohere-Embed-V5-Pro")
    embedding_dimensions: int = Field(default=2048)

    # OCR (T-005) — tier 3 is Azure Computer Vision v4 Image Analysis.
    # OCR_ENABLE_AZURE=false keeps OCR at tiers 1-2 (zero cost).
    # AZURE_CV_ENDPOINT / AZURE_CV_KEY override the host + key for the Vision
    # REST path; when unset the AZURE_AI_API_BASE / AZURE_AI_API_KEY pair is
    # reused. Set them when Vision is served from a separate
    # *.cognitiveservices.azure.com resource.
    ocr_enable_azure: bool = Field(default=True)
    azure_cv_endpoint: str = Field(default="")
    azure_cv_key: str = Field(default="")

    # Chunking (T-006)
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)

    # LLM (T-012) — Azure OpenAI chat completion
    llm_deployment: str = Field(default="gpt-4o")
    llm_api_version: str = Field(default="2024-10-21")
    llm_max_tokens: int = Field(default=1024)
    llm_temperature: float = Field(default=0.2)
    llm_timeout_seconds: int = Field(default=60)

    # RAG (T-012)
    rag_top_k: int = Field(default=8)
    rag_max_context_chars: int = Field(default=12000)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
