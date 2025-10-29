from typing import Annotated, Any, Literal

from pydantic import AnyUrl, BeforeValidator, HttpUrl, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_cors(value: Any) -> list[str] | str:
    if isinstance(value, str) and not value.startswith("["):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, str)):
        return value
    raise ValueError(value)


class Settings(BaseSettings):
    """Application configuration without any database specific settings."""

    model_config = SettingsConfigDict(
        env_file="../.env",
        env_ignore_empty=True,
        extra="ignore",
    )

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Alphabet Sign Recognition Portal API"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"
    FRONTEND_HOST: str = "http://localhost:5173"
    BACKEND_CORS_ORIGINS: Annotated[
        list[AnyUrl] | str, BeforeValidator(parse_cors)
    ] = []
    SENTRY_DSN: HttpUrl | None = None
    ASL_CLASSIFIER_PATH: str | None = None
    ASL_MIN_DETECTION_CONFIDENCE: float = 0.5
    ASL_MIN_TRACKING_CONFIDENCE: float = 0.5
    ASL_MAX_NUM_HANDS: int = 1

    @computed_field  # type: ignore[prop-decorator]
    @property
    def all_cors_origins(self) -> list[str]:
        origins = [str(origin).rstrip("/") for origin in self.BACKEND_CORS_ORIGINS]
        frontend_host = self.FRONTEND_HOST.rstrip("/")
        if frontend_host not in origins:
            origins.append(frontend_host)
        return origins


settings = Settings()  # type: ignore
