from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from asr_viz.api.schemas import (
    AppendLiveTranscriptEventRequest,
    CreateJobRequest,
    CreateLiveSessionRequest,
    TranscriptReviewUpdateRequest,
    CreateStreamSessionRequest,
    JobListResponse,
    JobResponse,
    LiveSessionEventListResponse,
    LiveSessionResponse,
    LiveTranscriptEventResponse,
    StreamSessionResponse,
    TranscriptResponse,
)
from asr_viz.db.session import get_session
from asr_viz.models.job import ProcessingJob
from asr_viz.models.live import LiveSession
from asr_viz.models.stream import StreamIngestionSession
from asr_viz.models.transcript import SentenceUnit, Transcript
from asr_viz.services.bootstrap import init_db
from asr_viz.services.jobs import create_job, infer_source_type
from asr_viz.services.archive_previews import update_transcript_archive_preview
from asr_viz.services.live_sessions import (
    append_live_chunk,
    append_live_event,
    create_live_session,
    finalize_live_session,
    list_live_events,
    stop_live_session,
)
from asr_viz.services.streaming import (
    append_stream_chunk,
    create_stream_session,
    finalize_stream_session,
    refresh_stream_session_status,
)

app = FastAPI(title="ASR Viz Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4173", "http://127.0.0.1:4173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_REVIEW_STATUSES = {"not_started", "in_progress", "completed"}


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


def _with_preferred_language(ingest_metadata: dict | None, preferred_language: str) -> dict:
    return {
        **(ingest_metadata or {}),
        "preferred_language": preferred_language,
    }


def _serialize_stream_session(stream_session: StreamIngestionSession) -> StreamSessionResponse:
    job = stream_session.processing_job
    return StreamSessionResponse(
        id=stream_session.id,
        status=stream_session.status,
        mime_type=stream_session.mime_type,
        original_filename=stream_session.original_filename,
        storage_path=stream_session.storage_path,
        total_bytes=stream_session.total_bytes,
        received_chunks=stream_session.received_chunks,
        diarization_enabled=stream_session.diarization_enabled,
        ingest_metadata=stream_session.ingest_metadata,
        error_message=stream_session.error_message,
        processing_job_id=stream_session.processing_job_id,
        processing_job_status=job.status if job else None,
        processing_stage=job.current_stage if job else None,
        transcript_id=job.transcript_id if job else None,
        created_at=stream_session.created_at,
        updated_at=stream_session.updated_at,
    )


def _serialize_live_session(live_session: LiveSession) -> LiveSessionResponse:
    return LiveSessionResponse(
        id=live_session.id,
        status=live_session.status,
        mime_type=live_session.mime_type,
        original_filename=live_session.original_filename,
        sample_rate_hz=live_session.sample_rate_hz,
        channel_count=live_session.channel_count,
        storage_path=live_session.storage_path,
        total_bytes=live_session.total_bytes,
        received_chunks=live_session.received_chunks,
        last_chunk_index=live_session.last_chunk_index,
        session_metadata=live_session.session_metadata,
        error_message=live_session.error_message,
        transcript_id=live_session.transcript_id,
        media_asset_id=live_session.media_asset_id,
        processing_job_id=live_session.processing_job_id,
        stopped_at=live_session.stopped_at,
        finalized_at=live_session.finalized_at,
        created_at=live_session.created_at,
        updated_at=live_session.updated_at,
    )


@app.get("/jobs", response_model=JobListResponse)
def list_jobs(
    limit: int = 20,
    offset: int = 0,
    completed_only: bool = False,
    session: Session = Depends(get_session),
) -> JobListResponse:
    capped_limit = max(1, min(limit, 100))
    safe_offset = max(0, offset)
    base_query = select(ProcessingJob)
    total_query = select(func.count()).select_from(ProcessingJob)

    if completed_only:
        completion_filters = (
            ProcessingJob.status == "completed",
            ProcessingJob.transcript_id.is_not(None),
        )
        base_query = base_query.where(*completion_filters)
        total_query = total_query.where(*completion_filters)

    total = session.scalar(total_query) or 0
    jobs = session.scalars(
        base_query
        .order_by(ProcessingJob.created_at.desc())
        .offset(safe_offset)
        .limit(capped_limit)
        .options(selectinload(ProcessingJob.media_asset), selectinload(ProcessingJob.transcript))
    ).all()
    transcripts_needing_preview = [job.transcript for job in jobs if job.transcript is not None and job.transcript.archive_preview is None]
    if transcripts_needing_preview:
        transcript_ids = [transcript.id for transcript in transcripts_needing_preview]
        hydrated_transcripts = session.scalars(
            select(Transcript)
            .where(Transcript.id.in_(transcript_ids))
            .options(selectinload(Transcript.sentence_units).selectinload(SentenceUnit.analysis_result))
        ).all()
        for transcript in hydrated_transcripts:
            update_transcript_archive_preview(transcript)
        session.commit()
    return JobListResponse(jobs=jobs, total=int(total))


@app.post("/jobs", response_model=JobResponse, status_code=201)
def create_processing_job(
    request: CreateJobRequest,
    session: Session = Depends(get_session),
) -> ProcessingJob:
    source_type = request.source_type or infer_source_type(request.source_uri)
    job = create_job(
        session,
        source_uri=request.source_uri,
        source_type=source_type,
        diarization_enabled=request.diarization_enabled,
        mime_type=request.mime_type,
        checksum=request.checksum,
        ingest_metadata=_with_preferred_language(request.ingest_metadata, request.preferred_language),
    )
    return job


@app.post("/live-sessions", response_model=LiveSessionResponse, status_code=201)
def create_live_capture_session(
    request: CreateLiveSessionRequest,
    session: Session = Depends(get_session),
) -> LiveSessionResponse:
    live_session = create_live_session(
        session,
        mime_type=request.mime_type,
        original_filename=request.original_filename,
        sample_rate_hz=request.sample_rate_hz,
        channel_count=request.channel_count,
        session_metadata=_with_preferred_language(request.session_metadata, request.preferred_language),
    )
    return _serialize_live_session(live_session)


@app.get("/live-sessions/{session_id}", response_model=LiveSessionResponse)
def get_live_capture_session(session_id: str, session: Session = Depends(get_session)) -> LiveSessionResponse:
    live_session = session.get(LiveSession, session_id)
    if live_session is None:
        raise HTTPException(status_code=404, detail="live session not found")
    return _serialize_live_session(live_session)


@app.put("/live-sessions/{session_id}/chunks", status_code=202)
async def upload_live_session_chunk(
    session_id: str,
    request: Request,
    chunk_index: int | None = None,
    session: Session = Depends(get_session),
) -> Response:
    live_session = session.get(LiveSession, session_id)
    if live_session is None:
        raise HTTPException(status_code=404, detail="live session not found")

    chunk = await request.body()
    if not chunk:
        raise HTTPException(status_code=400, detail="empty chunk")

    try:
        append_live_chunk(session, live_session, chunk, chunk_index=chunk_index)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return Response(status_code=202)


@app.post("/live-sessions/{session_id}/events", response_model=LiveTranscriptEventResponse, status_code=201)
def append_live_capture_event(
    session_id: str,
    request: AppendLiveTranscriptEventRequest,
    session: Session = Depends(get_session),
) -> LiveTranscriptEventResponse:
    live_session = session.get(LiveSession, session_id)
    if live_session is None:
        raise HTTPException(status_code=404, detail="live session not found")

    try:
        event = append_live_event(
            session,
            live_session,
            event_type=request.event_type,
            utterance_key=request.utterance_key,
            start_ms=request.start_ms,
            end_ms=request.end_ms,
            text=request.text,
            speaker_id=request.speaker_id,
            is_final=request.is_final,
            payload=request.payload,
            analysis=request.analysis.model_dump() if request.analysis else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return event


@app.get("/live-sessions/{session_id}/events", response_model=LiveSessionEventListResponse)
def get_live_capture_events(session_id: str, session: Session = Depends(get_session)) -> LiveSessionEventListResponse:
    live_session = session.get(LiveSession, session_id)
    if live_session is None:
        raise HTTPException(status_code=404, detail="live session not found")
    return LiveSessionEventListResponse(events=list_live_events(session, session_id))


@app.post("/live-sessions/{session_id}/stop", response_model=LiveSessionResponse)
def stop_live_capture_session(session_id: str, session: Session = Depends(get_session)) -> LiveSessionResponse:
    live_session = session.get(LiveSession, session_id)
    if live_session is None:
        raise HTTPException(status_code=404, detail="live session not found")

    try:
        live_session = stop_live_session(session, live_session)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return _serialize_live_session(live_session)


@app.post("/live-sessions/{session_id}/finalize", response_model=LiveSessionResponse)
def finalize_live_capture_session(session_id: str, session: Session = Depends(get_session)) -> LiveSessionResponse:
    live_session = session.get(LiveSession, session_id)
    if live_session is None:
        raise HTTPException(status_code=404, detail="live session not found")

    try:
        live_session = finalize_live_session(session, live_session)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _serialize_live_session(live_session)


@app.post("/stream-sessions", response_model=StreamSessionResponse, status_code=201)
def create_streaming_session(
    request: CreateStreamSessionRequest,
    session: Session = Depends(get_session),
) -> StreamSessionResponse:
    stream_session = create_stream_session(
        session,
        mime_type=request.mime_type,
        original_filename=request.original_filename,
        diarization_enabled=request.diarization_enabled,
        ingest_metadata=_with_preferred_language(request.ingest_metadata, request.preferred_language),
    )
    return _serialize_stream_session(stream_session)


@app.put("/stream-sessions/{session_id}/chunks", status_code=202)
async def upload_stream_chunk(
    session_id: str,
    request: Request,
    session: Session = Depends(get_session),
) -> Response:
    stream_session = session.get(StreamIngestionSession, session_id)
    if stream_session is None:
        raise HTTPException(status_code=404, detail="stream session not found")

    chunk = await request.body()
    if not chunk:
        raise HTTPException(status_code=400, detail="empty chunk")

    try:
        append_stream_chunk(session, stream_session, chunk)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return Response(status_code=202)


@app.post("/stream-sessions/{session_id}/finalize", response_model=StreamSessionResponse)
def finalize_streaming_session(session_id: str, session: Session = Depends(get_session)) -> StreamSessionResponse:
    stream_session = session.get(StreamIngestionSession, session_id)
    if stream_session is None:
        raise HTTPException(status_code=404, detail="stream session not found")

    try:
        stream_session = finalize_stream_session(session, stream_session)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session.refresh(stream_session)
    return _serialize_stream_session(stream_session)


@app.get("/stream-sessions/{session_id}", response_model=StreamSessionResponse)
def get_streaming_session(session_id: str, session: Session = Depends(get_session)) -> StreamSessionResponse:
    stream_session = session.scalar(
        select(StreamIngestionSession)
        .where(StreamIngestionSession.id == session_id)
        .options(selectinload(StreamIngestionSession.processing_job))
    )
    if stream_session is None:
        raise HTTPException(status_code=404, detail="stream session not found")

    refresh_stream_session_status(stream_session)
    session.commit()
    session.refresh(stream_session)
    return _serialize_stream_session(stream_session)


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, session: Session = Depends(get_session)) -> ProcessingJob:
    job = session.scalar(
        select(ProcessingJob)
        .where(ProcessingJob.id == job_id)
        .options(selectinload(ProcessingJob.media_asset), selectinload(ProcessingJob.transcript))
    )
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/jobs/{job_id}/media")
def get_job_media(job_id: str, session: Session = Depends(get_session)) -> Response:
    job = session.scalar(
        select(ProcessingJob)
        .where(ProcessingJob.id == job_id)
        .options(selectinload(ProcessingJob.media_asset))
    )
    if job is None or job.media_asset is None:
        raise HTTPException(status_code=404, detail="job not found")

    source_uri = job.media_asset.source_uri
    if source_uri.startswith(("http://", "https://")):
        return RedirectResponse(url=source_uri)

    media_path = Path(source_uri)
    if not media_path.exists() or not media_path.is_file():
        raise HTTPException(status_code=404, detail="media file not found")

    return FileResponse(
        path=media_path,
        media_type=job.media_asset.mime_type,
        filename=job.media_display_name or media_path.name,
    )


@app.get("/transcripts/{transcript_id}", response_model=TranscriptResponse)
def get_transcript(transcript_id: str, session: Session = Depends(get_session)) -> Transcript:
    transcript = session.scalar(
        select(Transcript)
        .where(Transcript.id == transcript_id)
        .options(
            selectinload(Transcript.segments),
            selectinload(Transcript.sentence_units).selectinload(SentenceUnit.analysis_result),
        )
    )
    if transcript is None:
        raise HTTPException(status_code=404, detail="transcript not found")
    return transcript


@app.patch("/transcripts/{transcript_id}/review", response_model=TranscriptResponse)
def update_transcript_review(
    transcript_id: str,
    request: TranscriptReviewUpdateRequest,
    session: Session = Depends(get_session),
) -> Transcript:
    transcript = session.scalar(
        select(Transcript)
        .where(Transcript.id == transcript_id)
        .options(
            selectinload(Transcript.segments),
            selectinload(Transcript.sentence_units).selectinload(SentenceUnit.analysis_result),
        )
    )
    if transcript is None:
        raise HTTPException(status_code=404, detail="transcript not found")

    allowed_speaker_ids = {sentence.speaker_id for sentence in transcript.sentence_units if sentence.speaker_id}

    cleaned_speaker_labels: dict[str, str] = {}
    for speaker_id, label in request.speaker_labels.items():
        if speaker_id not in allowed_speaker_ids:
            raise HTTPException(status_code=400, detail=f"invalid speaker id in speaker_labels: {speaker_id}")
        if isinstance(label, str) and label.strip():
            cleaned_speaker_labels[speaker_id] = label.strip()

    sentence_units_by_id = {sentence.id: sentence for sentence in transcript.sentence_units}
    seen_sentence_ids: set[str] = set()
    for override in request.sentence_overrides:
        if override.sentence_unit_id not in sentence_units_by_id:
            raise HTTPException(status_code=400, detail=f"invalid sentence id: {override.sentence_unit_id}")
        if override.sentence_unit_id in seen_sentence_ids:
            raise HTTPException(status_code=400, detail=f"duplicate sentence id: {override.sentence_unit_id}")
        seen_sentence_ids.add(override.sentence_unit_id)
        if override.manual_speaker_id is not None and override.manual_speaker_id not in allowed_speaker_ids:
            raise HTTPException(status_code=400, detail=f"invalid speaker id in sentence override: {override.manual_speaker_id}")

    review_status = request.review_status.strip()
    if review_status not in ALLOWED_REVIEW_STATUSES:
        raise HTTPException(status_code=400, detail=f"invalid review status: {request.review_status}")

    transcript_metadata = dict(transcript.transcript_metadata)
    conversation_title = request.conversation_title.strip() if isinstance(request.conversation_title, str) else ""
    transcript_metadata["conversation_title"] = conversation_title or None
    transcript_metadata["speaker_labels"] = cleaned_speaker_labels
    transcript_metadata["review_status"] = review_status
    transcript_metadata["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    transcript.transcript_metadata = transcript_metadata

    for override in request.sentence_overrides:
        sentence = sentence_units_by_id[override.sentence_unit_id]
        sentence_metadata = dict(sentence.sentence_metadata)

        if override.manual_text is None:
            sentence_metadata.pop("manual_text", None)
        else:
            sentence_metadata["manual_text"] = override.manual_text

        if override.manual_speaker_id is None:
            sentence_metadata.pop("manual_speaker_id", None)
        else:
            sentence_metadata["manual_speaker_id"] = override.manual_speaker_id

        if "manual_text" in sentence_metadata or "manual_speaker_id" in sentence_metadata:
            sentence_metadata["edited_at"] = datetime.now(timezone.utc).isoformat()
        else:
            sentence_metadata.pop("edited_at", None)

        sentence.sentence_metadata = sentence_metadata

    update_transcript_archive_preview(transcript)
    session.commit()
    session.refresh(transcript)
    return transcript


def _delete_transcript(transcript_id: str, session: Session) -> Response:
    transcript = session.scalar(
        select(Transcript)
        .where(Transcript.id == transcript_id)
        .options(selectinload(Transcript.segments), selectinload(Transcript.sentence_units))
    )
    if transcript is None:
        raise HTTPException(status_code=404, detail="transcript not found")

    linked_job = session.scalar(select(ProcessingJob).where(ProcessingJob.transcript_id == transcript.id))
    if linked_job is None and transcript.job_id:
        linked_job = session.get(ProcessingJob, transcript.job_id)

    if linked_job is not None:
        linked_job.transcript_id = None

    session.delete(transcript)
    session.commit()
    return Response(status_code=204)


@app.delete("/transcripts/{transcript_id}", status_code=204)
def delete_transcript(transcript_id: str, session: Session = Depends(get_session)) -> Response:
    return _delete_transcript(transcript_id, session)


@app.post("/transcripts/{transcript_id}/delete", status_code=204)
def delete_transcript_via_post(transcript_id: str, session: Session = Depends(get_session)) -> Response:
    return _delete_transcript(transcript_id, session)
