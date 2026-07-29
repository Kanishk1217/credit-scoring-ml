"""Application settings, loaded from environment / .env (never hardcode secrets)."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="CREDIT_"
    )

    app_name: str = "Credit Scoring API"
    app_version: str = "1.0.0"
    environment: str = "development"          # "production" tightens behavior

    # Which trained model directory to serve: "models" (synthetic, default — keeps the existing
    # test suite's synthetic-data assumptions valid), "models_real" (7-feature, real Home Credit,
    # self-reportable schema — the public/consumer-facing choice), or "models_real_rich"
    # (38-feature, real Home Credit + bureau/EXT_SOURCE — internal loan-officer use only, since
    # those features require data a self-service applicant can't provide themselves).
    model_dir: str = "models"

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
