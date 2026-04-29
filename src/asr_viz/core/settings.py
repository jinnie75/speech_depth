from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _get_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return int(value)


def _get_csv(name: str, default: str = "") -> tuple[str, ...]:
    raw_value = os.getenv(name, default)
    return tuple(item.strip() for item in raw_value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    app_env: Literal["development", "test", "production"] = os.getenv("APP_ENV", "development")  # type: ignore[assignment]
    public_app_url: str | None = os.getenv("PUBLIC_APP_URL")
    cors_origins: tuple[str, ...] = _get_csv(
        "CORS_ORIGINS",
        "http://localhost:4173,http://127.0.0.1:4173",
    )
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./asr_viz.db")
    huggingface_token: str | None = os.getenv("HUGGINGFACE_TOKEN")
    asr_model: str = os.getenv("ASR_MODEL", "small")
    asr_device: str = os.getenv("ASR_DEVICE", "cpu")
    asr_compute_type: str = os.getenv("ASR_COMPUTE_TYPE", "int8")
    asr_cpu_threads: int | None = _get_optional_int("ASR_CPU_THREADS")
    enable_diarization: bool = _get_bool("ENABLE_DIARIZATION", True)
    diarization_model: str = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
    diarization_num_speakers: int | None = (
        int(os.getenv("DIARIZATION_NUM_SPEAKERS"))
        if os.getenv("DIARIZATION_NUM_SPEAKERS")
        else None
    )
    diarization_min_speakers: int | None = (
        int(os.getenv("DIARIZATION_MIN_SPEAKERS"))
        if os.getenv("DIARIZATION_MIN_SPEAKERS")
        else 1
    )
    diarization_max_speakers: int | None = (
        int(os.getenv("DIARIZATION_MAX_SPEAKERS"))
        if os.getenv("DIARIZATION_MAX_SPEAKERS")
        else 3
    )
    job_poll_interval_seconds: float = float(os.getenv("JOB_POLL_INTERVAL_SECONDS", "2.0"))
    enable_mock_transcription: bool = _get_bool("ENABLE_MOCK_TRANSCRIPTION", False)
    media_storage_dir: str = os.getenv("MEDIA_STORAGE_DIR", "./.media")
    auth_provider: Literal["disabled", "clerk"] = os.getenv("AUTH_PROVIDER", "disabled")  # type: ignore[assignment]
    require_auth: bool = _get_bool("REQUIRE_AUTH", False)
    auth_dev_user_id: str = os.getenv("AUTH_DEV_USER_ID", "local-dev-user")
    anonymous_session_cookie_name: str = os.getenv("ANONYMOUS_SESSION_COOKIE_NAME", "asr_viz_anon_session")
    anonymous_session_cookie_max_age_seconds: int = _get_int(
        "ANONYMOUS_SESSION_COOKIE_MAX_AGE_SECONDS",
        60 * 60 * 24 * 180,
    )
    anonymous_session_cookie_secure: bool = _get_bool("ANONYMOUS_SESSION_COOKIE_SECURE", False)
    anonymous_session_cookie_domain: str | None = os.getenv("ANONYMOUS_SESSION_COOKIE_DOMAIN")
    clerk_issuer_url: str | None = os.getenv("CLERK_ISSUER_URL")
    clerk_audience: str | None = os.getenv("CLERK_AUDIENCE")
    clerk_jwks_url: str | None = os.getenv("CLERK_JWKS_URL")
    storage_backend: Literal["local", "r2"] = os.getenv("STORAGE_BACKEND", "local")  # type: ignore[assignment]
    signed_url_ttl_seconds: int = _get_int("SIGNED_URL_TTL_SECONDS", 900)
    r2_account_id: str | None = os.getenv("R2_ACCOUNT_ID")
    r2_bucket_name: str | None = os.getenv("R2_BUCKET_NAME")
    r2_access_key_id: str | None = os.getenv("R2_ACCESS_KEY_ID")
    r2_secret_access_key: str | None = os.getenv("R2_SECRET_ACCESS_KEY")
    max_upload_size_bytes: int = _get_int("MAX_UPLOAD_SIZE_BYTES", 100 * 1024 * 1024)
    per_user_storage_quota_bytes: int = _get_int("PER_USER_STORAGE_QUOTA_BYTES", 500 * 1024 * 1024)
    live_mode_enabled: bool = _get_bool("LIVE_MODE_ENABLED", False)
    worker_name: str = os.getenv("WORKER_NAME", "render-cpu-worker")
    clerk_network_timeout_seconds: int = _get_int("CLERK_NETWORK_TIMEOUT_SECONDS", 5)
    auto_create_schema: bool = _get_bool("AUTO_CREATE_SCHEMA", True)
    upload_expected_size_hint_bytes: int | None = _get_optional_int("UPLOAD_EXPECTED_SIZE_HINT_BYTES")


settings = Settings()
