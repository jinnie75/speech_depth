from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from asr_viz.db.base import Base
from asr_viz.models.common import IdMixin, TimestampMixin


class Transcript(IdMixin, TimestampMixin, Base):
    __tablename__ = "transcripts"

    media_asset_id: Mapped[str] = mapped_column(ForeignKey("media_assets.id"), nullable=False, index=True)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("processing_jobs.id"), nullable=True, unique=True)
    language_code: Mapped[str | None] = mapped_column(String(32))
    full_text: Mapped[str] = mapped_column(Text, nullable=False)
    transcript_metadata: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    media_asset = relationship("MediaAsset", back_populates="transcripts")
    segments = relationship("TranscriptSegment", back_populates="transcript", cascade="all, delete-orphan")
    sentence_units = relationship("SentenceUnit", back_populates="transcript", cascade="all, delete-orphan")

    @property
    def conversation_title(self) -> str | None:
        value = self.transcript_metadata.get("conversation_title")
        return value.strip() if isinstance(value, str) and value.strip() else None

    @property
    def speaker_labels(self) -> dict[str, str]:
        raw_value = self.transcript_metadata.get("speaker_labels")
        if not isinstance(raw_value, dict):
            return {}
        return {
            str(speaker_id): label.strip()
            for speaker_id, label in raw_value.items()
            if isinstance(label, str) and label.strip()
        }

    @property
    def review_status(self) -> str:
        value = self.transcript_metadata.get("review_status")
        return value if isinstance(value, str) and value.strip() else "not_started"

    @property
    def reviewed_at(self) -> datetime | None:
        value = self.transcript_metadata.get("reviewed_at")
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None


class TranscriptSegment(IdMixin, TimestampMixin, Base):
    __tablename__ = "transcript_segments"

    transcript_id: Mapped[str] = mapped_column(ForeignKey("transcripts.id"), nullable=False, index=True)
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    avg_logprob: Mapped[float | None] = mapped_column(Float)
    no_speech_prob: Mapped[float | None] = mapped_column(Float)
    words_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    transcript = relationship("Transcript", back_populates="segments")


class SentenceUnit(IdMixin, TimestampMixin, Base):
    __tablename__ = "sentence_units"

    transcript_id: Mapped[str] = mapped_column(ForeignKey("transcripts.id"), nullable=False, index=True)
    utterance_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    speaker_id: Mapped[str | None] = mapped_column(String(255))
    speaker_confidence: Mapped[float | None] = mapped_column(Float)
    source_segment_ids: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    sentence_metadata: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    transcript = relationship("Transcript", back_populates="sentence_units")
    analysis_result = relationship(
        "AnalysisResult",
        back_populates="sentence_unit",
        cascade="all, delete-orphan",
        uselist=False,
    )

    @property
    def manual_text(self) -> str | None:
        value = self.sentence_metadata.get("manual_text")
        return value if isinstance(value, str) else None

    @property
    def manual_speaker_id(self) -> str | None:
        value = self.sentence_metadata.get("manual_speaker_id")
        return value if isinstance(value, str) and value.strip() else None

    @property
    def display_text(self) -> str:
        manual_text = self.manual_text
        return manual_text if manual_text is not None else self.text

    @property
    def display_speaker_id(self) -> str | None:
        manual_speaker_id = self.manual_speaker_id
        return manual_speaker_id if manual_speaker_id is not None else self.speaker_id

    @property
    def is_edited(self) -> bool:
        return self.manual_text is not None or self.manual_speaker_id is not None
