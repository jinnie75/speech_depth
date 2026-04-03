from __future__ import annotations

from sqlalchemy import JSON, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from asr_viz.db.base import Base
from asr_viz.models.common import IdMixin, TimestampMixin


class StreamIngestionSession(IdMixin, TimestampMixin, Base):
    __tablename__ = "stream_ingestion_sessions"

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open", index=True)
    mime_type: Mapped[str | None] = mapped_column(String(255))
    original_filename: Mapped[str | None] = mapped_column(String(512))
    storage_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    total_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    received_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    diarization_enabled: Mapped[bool] = mapped_column(default=False, nullable=False)
    ingest_metadata: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    error_message: Mapped[str | None] = mapped_column(String(2048))
    processing_job_id: Mapped[str | None] = mapped_column(ForeignKey("processing_jobs.id"), unique=True)

    processing_job = relationship("ProcessingJob", foreign_keys=[processing_job_id])
