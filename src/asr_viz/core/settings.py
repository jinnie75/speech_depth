from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./asr_viz.db")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    huggingface_token: str | None = os.getenv("HUGGINGFACE_TOKEN")
    asr_model: str = os.getenv("ASR_MODEL", "small")
    analysis_model: str = os.getenv("ANALYSIS_MODEL", "gpt-4.1-mini")
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
        else 2
    )
    job_poll_interval_seconds: float = float(os.getenv("JOB_POLL_INTERVAL_SECONDS", "2.0"))
    enable_mock_transcription: bool = os.getenv("ENABLE_MOCK_TRANSCRIPTION", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    media_storage_dir: str = os.getenv("MEDIA_STORAGE_DIR", "./.media")


settings = Settings()
