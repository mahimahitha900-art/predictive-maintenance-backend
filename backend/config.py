"""
Configuration management using pydantic-settings.
All settings are loaded from environment variables with sensible defaults.
"""
from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "Predictive Maintenance Agent"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # MongoDB
    MONGODB_URI: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection string (Atlas or local)",
    )
    MONGODB_DATABASE: str = Field(
        default="fathom",
        description="MongoDB database name",
    )

    # CORS
    CORS_ORIGINS: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173,https://predictive-maintenance-agent-fronte.vercel.app",
        description="Allowed CORS origins (comma-separated)",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse comma-separated CORS origins."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    # External APIs
    GRADIO_API_URL: str = Field(
        default="https://vvsgyuv123-predictive-maintenance-demo.hf.space/gradio_api",
        description="Gradio API base URL for fallback predictions",
    )

    # LLM recommendations via OpenRouter (OpenAI-compatible API)
    OPENROUTER_API_KEY: str = Field(
        default="",
        description="OpenRouter API key (empty disables LLM recommendations)",
    )
    OPENROUTER_BASE_URL: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )
    OPENROUTER_MODEL: str = Field(
        default="deepseek/deepseek-chat",
        description="OpenRouter model used for recommendation options",
    )
    LLM_TIMEOUT_SECONDS: float = Field(
        default=20.0,
        description="Timeout in seconds for OpenRouter chat completions calls",
    )

    # Server
    PORT: int = Field(default=8000, description="Server port")

    # Rate Limiting (future use)
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()


settings = get_settings()