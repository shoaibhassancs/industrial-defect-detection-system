"""Tests for app.core.settings: defaults, parsing from the environment, and refusals."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.core.settings import Settings
from pydantic import ValidationError


def make(**values: object) -> Settings:
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_defaults_are_the_documented_ones() -> None:
    s = make()
    assert s.app_env == "development" and s.model_dir is None and s.model_version is None
    assert s.allowed_origins == [] and s.max_upload_mb == 10 and s.rate_limit == "10/minute"
    assert s.trusted_proxy_hops == 0 and s.torch_num_threads == 1 and s.log_level == "INFO"
    assert (s.min_image_side, s.max_image_side) == (64, 8000)
    assert s.persistence_enabled is False and s.max_upload_bytes == 10 * 1024 * 1024


def test_settings_are_read_from_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "APP_ENV": "production", "MODEL_DIR": "/models/pkg", "MODEL_VERSION": "1.0.0",
        "ALLOWED_ORIGINS": "https://a.example, http://localhost:5173",
        "MAX_UPLOAD_MB": "5", "RATE_LIMIT": "3/second", "TRUSTED_PROXY_HOPS": "1",
        "TORCH_NUM_THREADS": "2", "LOG_LEVEL": "debug",
    }.items():  # fmt: skip
        monkeypatch.setenv(name, value)
    s = Settings(_env_file=None)
    assert s.app_env == "production" and s.model_dir == Path("/models/pkg")
    assert s.allowed_origins == ["https://a.example", "http://localhost:5173"]
    assert (s.max_upload_mb, s.rate_limit, s.trusted_proxy_hops) == (5, "3/second", 1)
    assert s.torch_num_threads == 2 and s.log_level == "DEBUG"


def test_an_empty_value_means_not_set_not_the_current_folder() -> None:
    s = make(model_dir="", model_version=" ", evaluation_dir="")
    assert s.model_dir is None and s.model_version is None and s.evaluation_dir is None


@pytest.mark.parametrize(
    "bad",
    [
        {"max_upload_mb": 0}, {"max_upload_mb": 51}, {"torch_num_threads": 0},
        {"torch_num_threads": 17}, {"trusted_proxy_hops": -1}, {"trusted_proxy_hops": 6},
        {"inference_queue_timeout_s": 0}, {"inference_queue_timeout_s": 500},
        {"rate_limit": "ten per minute"}, {"rate_limit": "10/fortnight"},
        {"log_level": "LOUD"}, {"app_env": "staging"},
        {"allowed_origins": ["*"]}, {"allowed_origins": "https://a.example/path"},
        {"allowed_origins": "example.com"}, {"min_image_side": 9000},
        {"persistence_enabled": True}, {"app_env": "production"},
    ],
)  # fmt: skip
def test_bad_values_are_refused_at_start_up(bad: dict) -> None:
    with pytest.raises(ValidationError):
        make(**bad)


def test_production_needs_a_model_folder() -> None:
    assert make(app_env="production", model_dir="/models/pkg").app_env == "production"


def test_unknown_environment_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOMETHING_ELSE", "1")
    assert Settings(_env_file=None).app_env == "development"
