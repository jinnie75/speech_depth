from __future__ import annotations

from datetime import datetime, timezone
import enum
import uuid

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    return str(uuid.uuid4())


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )


class IdMixin:
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStage(str, enum.Enum):
    INGESTION = "ingestion"
    TRANSCRIPTION = "transcription"
    DIARIZATION = "diarization"
    ANALYSIS = "analysis"
    COMPLETE = "complete"
    FAILED = "failed"
