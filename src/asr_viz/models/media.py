from __future__ import annotations

from sqlalchemy import JSON, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from asr_viz.db.base import Base
from asr_viz.models.common import IdMixin, TimestampMixin


class MediaAsset(IdMixin, TimestampMixin, Base):
    __tablename__ = "media_assets"

    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_uri: Mapped[str] = mapped_column(String(2048), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(255))
    checksum: Mapped[str | None] = mapped_column(String(128))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    ingest_metadata: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    jobs = relationship("ProcessingJob", back_populates="media_asset", cascade="all, delete-orphan")
    transcripts = relationship("Transcript", back_populates="media_asset", cascade="all, delete-orphan")
