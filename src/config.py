"""
src/config.py — Application-wide configuration using Pydantic Settings.

All values are loaded from environment variables (.env file in development,
Secret Manager / GCP env in production). Never hardcode secrets.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ──────────────────────────────────────────────────────────
    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    cors_origins: list[AnyHttpUrl] = Field(default=["http://localhost:5173"])

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: PostgresDsn = Field(
        default="postgresql://user:password@localhost:5432/disease_prediction"
    )

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")
    prediction_cache_ttl_seconds: int = 300  # 5 minutes

    # ── JWT Authentication ────────────────────────────────────────────────────
    jwt_secret: str = Field(default="change-me-in-production-replace-this!!", min_length=32)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"

    # ── Model Registry ────────────────────────────────────────────────────────
    #   Maps disease key → MLflow registered model name
    registered_models: dict[str, str] = Field(
        default={
            "heart": "disease-prediction-heart",
            "diabetes": "disease-prediction-diabetes",
            "cancer": "disease-prediction-cancer",
            "kidney": "disease-prediction-kidney",
        }
    )

    # ── GCP (production) ──────────────────────────────────────────────────────
    gcp_project_id: str = ""
    gcs_bucket_name: str = ""

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_general: str = "100/minute"
    rate_limit_prediction: str = "20/minute"

    # ── SMTP ─────────────────────────────────────────────────────────────────
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance — use this everywhere instead of Settings()."""
    return Settings()
