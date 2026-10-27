"""Centralised settings. All secrets flow from `.env` only."""
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Service
    app_env: str = Field(default="development")
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8700)
    workers: int = Field(default=4)

    # Auth
    app_password_hash: str = Field(default="")
    session_secret: str = Field(default="")
    session_max_age_seconds: int = Field(default=7 * 24 * 3600)

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
