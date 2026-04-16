from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from asr_viz.db.base import Base
from asr_viz.models.common import IdMixin, JobStage, JobStatus, TimestampMixin


class ProcessingJob(IdMixin, TimestampMixin, Base):
    __tablename__ = "processing_jobs"

    media_asset_id: Mapped[str] = mapped_column(ForeignKey("media_assets.id"), nullable=False, index=True)
    transcript_id: Mapped[str | None] = mapped_column(ForeignKey("transcripts.id"), nullable=True, unique=True)
    status: Mapped[str] = mapped_column(String(32), default=JobStatus.QUEUED.value, nullable=False, index=True)
    current_stage: Mapped[str] = mapped_column(String(32), default=JobStage.INGESTION.value, nullable=False)
    error_message: Mapped[str | None] = mapped_column(String(2048))
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    diarization_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    asr_model_version: Mapped[str | None] = mapped_column(String(255))
    diarization_model_version: Mapped[str | None] = mapped_column(String(255))
    analysis_model_version: Mapped[str | None] = mapped_column(String(255))
    stage_details: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    media_asset = relationship("MediaAsset", back_populates="jobs")
    transcript = relationship("Transcript", foreign_keys=[transcript_id], post_update=True)

    @property
    def media_source_uri(self) -> str | None:
        return self.media_asset.source_uri if self.media_asset is not None else None

    @property
    def media_mime_type(self) -> str | None:
        return self.media_asset.mime_type if self.media_asset is not None else None

    @property
    def media_ingest_metadata(self) -> dict:
        return self.media_asset.ingest_metadata if self.media_asset is not None else {}

    @property
    def media_display_name(self) -> str | None:
        if self.media_asset is None:
            return None

        original_filename = self.media_asset.ingest_metadata.get("original_filename")
        if isinstance(original_filename, str) and original_filename.strip():
            return original_filename

        source_uri = self.media_asset.source_uri
        source_name = Path(source_uri).name
        return source_name or source_uri

    @property
    def conversation_title(self) -> str | None:
        if self.transcript is None:
            return None
        return self.transcript.conversation_title

    @property
    def review_status(self) -> str:
        if self.transcript is None:
            return "not_started"
        return self.transcript.review_status

    @property
    def archive_preview(self) -> dict | None:
        if self.transcript is None:
            return None
        return self.transcript.archive_preview
