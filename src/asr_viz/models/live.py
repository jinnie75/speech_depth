from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from asr_viz.db.base import Base
from asr_viz.models.common import IdMixin, TimestampMixin


class LiveSession(IdMixin, TimestampMixin, Base):
    __tablename__ = "live_sessions"

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open", index=True)
    mime_type: Mapped[str | None] = mapped_column(String(255))
    original_filename: Mapped[str | None] = mapped_column(String(512))
    sample_rate_hz: Mapped[int | None] = mapped_column(Integer)
    channel_count: Mapped[int | None] = mapped_column(Integer)
    storage_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    total_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    received_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=-1)
    session_metadata: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    error_message: Mapped[str | None] = mapped_column(String(2048))
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finalized_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    transcript_id: Mapped[str | None] = mapped_column(ForeignKey("transcripts.id"), unique=True)
    media_asset_id: Mapped[str | None] = mapped_column(ForeignKey("media_assets.id"), unique=True)
    processing_job_id: Mapped[str | None] = mapped_column(ForeignKey("processing_jobs.id"), unique=True)

    transcript = relationship("Transcript", foreign_keys=[transcript_id])
    media_asset = relationship("MediaAsset", foreign_keys=[media_asset_id])
    processing_job = relationship("ProcessingJob", foreign_keys=[processing_job_id])
    events = relationship(
        "LiveTranscriptEvent",
        back_populates="live_session",
        cascade="all, delete-orphan",
        order_by="LiveTranscriptEvent.event_index",
    )


class LiveTranscriptEvent(IdMixin, TimestampMixin, Base):
    __tablename__ = "live_transcript_events"

    live_session_id: Mapped[str] = mapped_column(ForeignKey("live_sessions.id"), nullable=False, index=True)
    event_index: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    utterance_key: Mapped[str | None] = mapped_column(String(255), index=True)
    start_ms: Mapped[int | None] = mapped_column(Integer)
    end_ms: Mapped[int | None] = mapped_column(Integer)
    text: Mapped[str | None] = mapped_column(Text)
    speaker_id: Mapped[str | None] = mapped_column(String(255))
    is_final: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    live_session = relationship("LiveSession", back_populates="events")
