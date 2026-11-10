"""Settings of the API, read from environment variables (PLAN.md Phase 5 step 1).

Every setting has a documented default and is validated when the service starts, so a bad value
stops the start-up with a clear message instead of misbehaving later. ``.env.example`` lists them.
Names are the upper-case environment variable names (``APP_ENV``, ``MODEL_DIR``, ...).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from limits import parse as parse_rate_limit
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_ORIGIN = re.compile(r"^https?://[A-Za-z0-9.\-]+(:\d{1,5})?$")


class Settings(BaseSettings):
    """Configuration of one running API instance."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),  # fields named model_* are ours, not pydantic's
    )

    # --- environment and model ---------------------------------------------------------------
    app_env: Literal["development", "test", "production"] = "development"
    model_dir: Path | None = None  # a packaged or calibrated artifact folder
    model_version: str | None = None  # if set, the artifact's version must equal it
    evaluation_dir: Path | None = None  # folder with the evaluation JSON files served at /metrics

    # --- HTTP --------------------------------------------------------------------------------
    allowed_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    max_upload_mb: int = Field(default=10, ge=1, le=50)
    rate_limit: str = "10/minute"  # applied to POST /api/v1/inspections, per client
    trusted_proxy_hops: int = Field(default=0, ge=0, le=5)  # see app/core/clientip.py
    log_level: str = "INFO"

    # --- inference ---------------------------------------------------------------------------
    torch_num_threads: int = Field(default=1, ge=1, le=16)
    inference_queue_timeout_s: float = Field(default=10.0, gt=0, le=120)

    # --- uploads (PLAN.md section 6.3) -------------------------------------------------------
    min_image_side: int = Field(default=64, ge=1)
    max_image_side: int = Field(default=8000, ge=64)
    max_image_pixels: int = Field(default=40_000_000, ge=4096)

    # --- Release 2 (not built): must stay false in Release 1 ---------------------------------
    persistence_enabled: bool = False

    @field_validator("model_dir", "model_version", "evaluation_dir", mode="before")
    @classmethod
    def _empty_means_unset(cls, value: object) -> object:
        """``MODEL_DIR=`` (an empty value in .env) means not set, not the current folder."""
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("allowed_origins")
    @classmethod
    def _check_origins(cls, origins: list[str]) -> list[str]:
        for origin in origins:
            if not _ORIGIN.match(origin):
                raise ValueError(
                    f"ALLOWED_ORIGINS entry {origin!r} must look like https://example.com "
                    "(scheme, host, optional port; no path, no wildcard)"
                )
        return origins

    @field_validator("rate_limit")
    @classmethod
    def _check_rate_limit(cls, value: str) -> str:
        try:
            parse_rate_limit(value)
        except ValueError as error:
            msg = f"RATE_LIMIT {value!r} is not a valid limit like '10/minute'"
            raise ValueError(msg) from error
        return value

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in _LOG_LEVELS:
            raise ValueError(f"LOG_LEVEL must be one of {_LOG_LEVELS}")
        return level

    @field_validator("persistence_enabled")
    @classmethod
    def _no_persistence_yet(cls, value: bool) -> bool:
        if value:
            raise ValueError("PERSISTENCE_ENABLED=true is Release 2 and is not implemented")
        return value

    @model_validator(mode="after")
    def _check_together(self) -> Settings:
        if self.min_image_side >= self.max_image_side:
            raise ValueError("MIN_IMAGE_SIDE must be smaller than MAX_IMAGE_SIDE")
        if self.app_env == "production" and self.model_dir is None:
            raise ValueError("MODEL_DIR is required when APP_ENV=production")
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """The settings of this process (read once)."""
    return Settings()
