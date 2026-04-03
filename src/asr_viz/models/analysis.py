from __future__ import annotations

from sqlalchemy import JSON, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from asr_viz.db.base import Base
from asr_viz.models.common import IdMixin, TimestampMixin


class AnalysisResult(IdMixin, TimestampMixin, Base):
    __tablename__ = "analysis_results"

    sentence_unit_id: Mapped[str] = mapped_column(ForeignKey("sentence_units.id"), nullable=False, unique=True, index=True)
    politeness_score: Mapped[float] = mapped_column(Float, nullable=False)
    semantic_confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    main_message_likelihood: Mapped[float] = mapped_column(Float, nullable=False)
    analysis_model: Mapped[str] = mapped_column(String(255), nullable=False)
    analysis_payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    sentence_unit = relationship("SentenceUnit", back_populates="analysis_result")
