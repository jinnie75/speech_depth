from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ASRWord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    word: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    probability: float | None = Field(default=None, ge=0.0, le=1.0)


class ASRSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_index: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str = Field(min_length=1)
    avg_logprob: float | None = None
    no_speech_prob: float | None = Field(default=None, ge=0.0, le=1.0)
    words: list[ASRWord] = Field(default_factory=list)
    raw_payload: dict = Field(default_factory=dict)

    @field_validator("end_ms")
    @classmethod
    def validate_segment_end(cls, value: int, info):
        start_ms = info.data.get("start_ms", 0)
        if value < start_ms:
            raise ValueError("segment end_ms must be greater than or equal to start_ms")
        return value


class TranscriptResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language_code: str | None = None
    full_text: str
    segments: list[ASRSegment]
    metadata: dict = Field(default_factory=dict)


class SentenceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    utterance_index: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str = Field(min_length=1)
    source_segment_ids: list[int] = Field(default_factory=list)
    speaker_id: str | None = None
    speaker_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    sentence_metadata: dict = Field(default_factory=dict)

    @field_validator("end_ms")
    @classmethod
    def validate_sentence_end(cls, value: int, info):
        start_ms = info.data.get("start_ms", 0)
        if value < start_ms:
            raise ValueError("sentence end_ms must be greater than or equal to start_ms")
        return value


class SentenceAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    politeness_score: float = Field(ge=0.0, le=1.0)
    semantic_confidence_score: float = Field(ge=0.0, le=1.0)
    main_message_likelihood: float = Field(ge=0.0, le=1.0)
    analysis_payload: dict = Field(default_factory=dict)


class SentenceAnalysisBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[SentenceAnalysis]
