"""Application settings, loaded from environment / .env (never hardcode secrets)."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="CREDIT_"
    )

    app_name: str = "Credit Scoring API"
    app_version: str = "1.0.0"
    environment: str = "development"          # "production" tightens behavior

    # Comma-separated list of valid API keys. REQUIRED — the API refuses to score without keys.
    api_keys: str = ""

    # CORS: exact origins allowed to call the API from a browser (never "*" with credentials).
    allowed_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Rate limit applied per API key (falls back to client IP).
    rate_limit: str = "30/minute"

    # Expose interactive /docs only when true (turn off in production).
    enable_docs: bool = True

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


settings = Settings()
