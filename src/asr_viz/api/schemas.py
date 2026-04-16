from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PreferredLanguage = Literal["auto", "en", "ko"]


class CreateJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_uri: str = Field(min_length=1)
    source_type: str | None = None
    diarization_enabled: bool = False
    mime_type: str | None = None
    checksum: str | None = None
    preferred_language: PreferredLanguage = "auto"
    ingest_metadata: dict = Field(default_factory=dict)


class CreateStreamSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mime_type: str | None = None
    original_filename: str | None = None
    diarization_enabled: bool = False
    preferred_language: PreferredLanguage = "auto"
    ingest_metadata: dict = Field(default_factory=dict)


class CreateLiveSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mime_type: str | None = None
    original_filename: str | None = None
    sample_rate_hz: int | None = Field(default=None, ge=1)
    channel_count: int | None = Field(default=None, ge=1)
    preferred_language: PreferredLanguage = "auto"
    session_metadata: dict = Field(default_factory=dict)


class ArchivePreviewSpeakerResponse(BaseModel):
    id: str
    label: str
    side: Literal["left", "right"]
    opacity: float
    total_duration_ms: int
    average_politeness: float


class ArchivePreviewUtteranceResponse(BaseModel):
    id: str
    speaker_id: str
    politeness_score: float
    contour_signal_count: float
    progress: float
    order: int


class ArchivePreviewTranscriptTokenResponse(BaseModel):
    id: str
    kind: Literal["text", "word"]
    text: str
    start: int | None = None
    end: int | None = None
    is_hedge: bool | None = None
    is_substance: bool | None = None
    drop_start_ms: int | None = None
    drop_duration_ms: int | None = None


class ArchivePreviewTranscriptResponse(BaseModel):
    utterance_id: str
    current_time_ms: int
    tokens: list[ArchivePreviewTranscriptTokenResponse]


class ArchivePreviewMarginNoteResponse(BaseModel):
    id: str
    text: str
    speaker_id: str
    utterance_id: str
    appear_at_ms: int
    source_start: int
    source_end: int
    settle_duration_ms: int


class ArchivePreviewResponse(BaseModel):
    speakers: list[ArchivePreviewSpeakerResponse]
    utterances: list[ArchivePreviewUtteranceResponse]
    active_speaker_id: str | None = None
    active_transcript: ArchivePreviewTranscriptResponse | None = None
    margin_notes: list[ArchivePreviewMarginNoteResponse]
    current_time_ms: int


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    media_asset_id: str
    transcript_id: str | None
    status: str
    current_stage: str
    error_message: str | None
    retry_count: int
    diarization_enabled: bool
    asr_model_version: str | None
    diarization_model_version: str | None
    analysis_model_version: str | None
    stage_details: dict
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime
    media_source_uri: str | None = None
    media_mime_type: str | None = None
    media_ingest_metadata: dict = Field(default_factory=dict)
    media_display_name: str | None = None
    conversation_title: str | None = None
    review_status: str = "not_started"
    archive_preview: ArchivePreviewResponse | None = None


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    total: int


class StreamSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    mime_type: str | None
    original_filename: str | None
    storage_path: str
    total_bytes: int
    received_chunks: int
    diarization_enabled: bool
    ingest_metadata: dict
    error_message: str | None
    processing_job_id: str | None
    processing_job_status: str | None = None
    processing_stage: str | None = None
    transcript_id: str | None = None
    created_at: datetime
    updated_at: datetime


class LiveAnalysisEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    politeness_score: float = Field(ge=0.0, le=1.0)
    semantic_confidence_score: float = Field(ge=0.0, le=1.0)
    main_message_likelihood: float = Field(ge=0.0, le=1.0)
    analysis_model: str = Field(default="live-heuristic:mvp", min_length=1)
    analysis_payload: dict = Field(default_factory=dict)


class AppendLiveTranscriptEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: Literal["transcript.delta", "transcript.final", "analysis.delta", "session.state"]
    utterance_key: str | None = None
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    text: str | None = None
    speaker_id: str | None = None
    is_final: bool = False
    payload: dict = Field(default_factory=dict)
    analysis: LiveAnalysisEventRequest | None = None


class LiveTranscriptEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    live_session_id: str
    event_index: int
    event_type: str
    utterance_key: str | None
    start_ms: int | None
    end_ms: int | None
    text: str | None
    speaker_id: str | None
    is_final: bool
    payload: dict
    created_at: datetime
    updated_at: datetime


class LiveSessionEventListResponse(BaseModel):
    events: list[LiveTranscriptEventResponse]


class LiveSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    mime_type: str | None
    original_filename: str | None
    sample_rate_hz: int | None
    channel_count: int | None
    storage_path: str
    total_bytes: int
    received_chunks: int
    last_chunk_index: int
    session_metadata: dict
    error_message: str | None
    transcript_id: str | None
    media_asset_id: str | None
    processing_job_id: str | None
    stopped_at: datetime | None
    finalized_at: datetime | None
    created_at: datetime
    updated_at: datetime


class MediaAssetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    source_type: str
    source_uri: str
    mime_type: str | None
    checksum: str | None
    duration_ms: int | None
    ingest_metadata: dict


class AnalysisResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    politeness_score: float
    semantic_confidence_score: float
    main_message_likelihood: float
    analysis_model: str
    analysis_payload: dict


class SentenceUnitResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    utterance_index: int
    start_ms: int
    end_ms: int
    text: str
    speaker_id: str | None
    speaker_confidence: float | None
    display_text: str
    display_speaker_id: str | None
    manual_text: str | None
    manual_speaker_id: str | None
    is_edited: bool
    source_segment_ids: list
    sentence_metadata: dict
    analysis_result: AnalysisResultResponse | None


class TranscriptSegmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    segment_index: int
    start_ms: int
    end_ms: int
    text: str
    avg_logprob: float | None
    no_speech_prob: float | None
    words_json: list
    raw_payload: dict


class TranscriptResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    media_asset_id: str
    job_id: str | None
    language_code: str | None
    full_text: str
    transcript_metadata: dict
    conversation_title: str | None = None
    speaker_labels: dict[str, str] = Field(default_factory=dict)
    review_status: str = "not_started"
    reviewed_at: datetime | None = None
    created_at: datetime
    segments: list[TranscriptSegmentResponse]
    sentence_units: list[SentenceUnitResponse]


class SentenceReviewUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentence_unit_id: str = Field(min_length=1)
    manual_text: str | None = None
    manual_speaker_id: str | None = None


class TranscriptReviewUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_title: str | None = None
    speaker_labels: dict[str, str] = Field(default_factory=dict)
    review_status: str = Field(default="in_progress", min_length=1)
    sentence_overrides: list[SentenceReviewUpdateRequest] = Field(default_factory=list)
