from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from asr_viz.core.settings import settings
from asr_viz.models.common import JobStatus
from asr_viz.models.stream import StreamIngestionSession
from asr_viz.services.jobs import create_job
from asr_viz.services.media import checksum_for_local_file


_MIME_EXTENSIONS = {
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "text/plain": ".txt",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
}


def create_stream_session(
    session: Session,
    *,
    mime_type: str | None,
    original_filename: str | None,
    diarization_enabled: bool,
    ingest_metadata: dict | None,
) -> StreamIngestionSession:
    storage_dir = Path(settings.media_storage_dir) / "stream_ingestion"
    storage_dir.mkdir(parents=True, exist_ok=True)

    session_record = StreamIngestionSession(
        status="open",
        mime_type=mime_type,
        original_filename=original_filename,
        storage_path=str(storage_dir / _build_storage_filename("pending", original_filename, mime_type)),
        diarization_enabled=diarization_enabled,
        ingest_metadata=ingest_metadata or {},
    )
    session.add(session_record)
    session.flush()

    storage_path = storage_dir / _build_storage_filename(session_record.id, original_filename, mime_type)
    session_record.storage_path = str(storage_path)
    storage_path.touch(exist_ok=True)
    session.commit()
    session.refresh(session_record)
    return session_record


def append_stream_chunk(session: Session, stream_session: StreamIngestionSession, chunk: bytes) -> StreamIngestionSession:
    if stream_session.status != "open":
        raise ValueError("stream session is not open for uploads")

    path = Path(stream_session.storage_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(chunk)

    stream_session.total_bytes += len(chunk)
    stream_session.received_chunks += 1
    session.commit()
    session.refresh(stream_session)
    return stream_session


def finalize_stream_session(session: Session, stream_session: StreamIngestionSession) -> StreamIngestionSession:
    if stream_session.status != "open":
        raise ValueError("stream session has already been finalized")
    if stream_session.total_bytes <= 0:
        raise ValueError("stream session has no uploaded bytes")

    media_path = Path(stream_session.storage_path)
    if not media_path.exists():
        raise FileNotFoundError(f"stream media not found: {stream_session.storage_path}")

    job = create_job(
        session,
        source_uri=str(media_path),
        source_type="file",
        diarization_enabled=stream_session.diarization_enabled,
        mime_type=stream_session.mime_type,
        checksum=checksum_for_local_file(str(media_path)),
        ingest_metadata={
            **stream_session.ingest_metadata,
            "stream_ingestion_session_id": stream_session.id,
            "original_filename": stream_session.original_filename,
            "received_chunks": stream_session.received_chunks,
            "total_bytes": stream_session.total_bytes,
        },
    )

    stream_session.processing_job_id = job.id
    stream_session.status = "queued"
    stream_session.error_message = None
    session.commit()
    session.refresh(stream_session)
    return stream_session


def refresh_stream_session_status(stream_session: StreamIngestionSession) -> None:
    job = stream_session.processing_job
    if job is None:
        return

    if job.status == JobStatus.QUEUED.value:
        stream_session.status = "queued"
    elif job.status == JobStatus.PROCESSING.value:
        stream_session.status = "processing"
    elif job.status == JobStatus.COMPLETED.value:
        stream_session.status = "completed"
    elif job.status == JobStatus.FAILED.value:
        stream_session.status = "failed"
        stream_session.error_message = job.error_message


def _build_storage_filename(session_id: str, original_filename: str | None, mime_type: str | None) -> str:
    provided_suffix = Path(original_filename or "").suffix
    suffix = provided_suffix or _MIME_EXTENSIONS.get((mime_type or "").lower(), ".bin")
    return f"{session_id}{suffix}"
