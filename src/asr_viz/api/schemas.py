from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CreateJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_uri: str = Field(min_length=1)
    source_type: str | None = None
    diarization_enabled: bool = False
    mime_type: str | None = None
    checksum: str | None = None
    ingest_metadata: dict = Field(default_factory=dict)


class CreateStreamSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mime_type: str | None = None
    original_filename: str | None = None
    diarization_enabled: bool = False
    ingest_metadata: dict = Field(default_factory=dict)


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


class JobListResponse(BaseModel):
    jobs: list[JobResponse]


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
    created_at: datetime
    segments: list[TranscriptSegmentResponse]
    sentence_units: list[SentenceUnitResponse]
