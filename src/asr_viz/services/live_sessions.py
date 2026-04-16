from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from asr_viz.core.settings import settings
from asr_viz.models.analysis import AnalysisResult
from asr_viz.models.common import utcnow
from asr_viz.models.live import LiveSession, LiveTranscriptEvent
from asr_viz.models.media import MediaAsset
from asr_viz.models.transcript import SentenceUnit, Transcript
from asr_viz.services.archive_previews import update_transcript_archive_preview
from asr_viz.services.media import checksum_for_local_file

_MIME_EXTENSIONS = {
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/webm": ".webm",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
}

_FINALIZABLE_STATUSES = {"recording", "stopped"}
_EVENT_WRITABLE_STATUSES = {"open", "recording", "stopped", "finalizing"}


def create_live_session(
    session: Session,
    *,
    mime_type: str | None,
    original_filename: str | None,
    sample_rate_hz: int | None,
    channel_count: int | None,
    session_metadata: dict | None,
) -> LiveSession:
    storage_dir = Path(settings.media_storage_dir) / "live_sessions"
    storage_dir.mkdir(parents=True, exist_ok=True)

    live_session = LiveSession(
        status="open",
        mime_type=mime_type,
        original_filename=original_filename,
        sample_rate_hz=sample_rate_hz,
        channel_count=channel_count,
        storage_path=str(storage_dir / _build_storage_filename("pending", original_filename, mime_type)),
        session_metadata=session_metadata or {},
    )
    session.add(live_session)
    session.flush()

    storage_path = storage_dir / _build_storage_filename(live_session.id, original_filename, mime_type)
    storage_path.touch(exist_ok=True)
    live_session.storage_path = str(storage_path)

    media_asset = MediaAsset(
        source_type="live",
        source_uri=str(storage_path),
        mime_type=mime_type,
        checksum=None,
        ingest_metadata={
            **(session_metadata or {}),
            "live_session_id": live_session.id,
            "original_filename": original_filename,
            "capture_mode": "live",
        },
    )
    session.add(media_asset)
    session.flush()

    transcript = Transcript(
        media_asset_id=media_asset.id,
        language_code=None,
        full_text="",
        transcript_metadata={
            "source": "live",
            "live_session_id": live_session.id,
            "review_status": "not_started",
        },
    )
    session.add(transcript)
    session.flush()

    live_session.media_asset_id = media_asset.id
    live_session.transcript_id = transcript.id

    session.commit()
    session.refresh(live_session)
    return live_session


def append_live_chunk(
    session: Session,
    live_session: LiveSession,
    chunk: bytes,
    *,
    chunk_index: int | None = None,
) -> LiveSession:
    if live_session.status not in {"open", "recording"}:
        raise ValueError("live session is not open for chunk uploads")

    if chunk_index is not None and chunk_index != live_session.last_chunk_index + 1:
        raise ValueError("chunk index must be sequential")

    path = Path(live_session.storage_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(chunk)

    live_session.status = "recording"
    live_session.total_bytes += len(chunk)
    live_session.received_chunks += 1
    if chunk_index is not None:
        live_session.last_chunk_index = chunk_index
    elif live_session.last_chunk_index < 0:
        live_session.last_chunk_index = 0
    else:
        live_session.last_chunk_index += 1

    session.commit()
    session.refresh(live_session)
    return live_session


def append_live_event(
    session: Session,
    live_session: LiveSession,
    *,
    event_type: str,
    utterance_key: str | None,
    start_ms: int | None,
    end_ms: int | None,
    text: str | None,
    speaker_id: str | None,
    is_final: bool,
    payload: dict | None,
    analysis: dict | None,
) -> LiveTranscriptEvent:
    if live_session.status not in _EVENT_WRITABLE_STATUSES:
        raise ValueError("live session is not accepting transcript events")

    if event_type.startswith("transcript.") and not utterance_key:
        raise ValueError("transcript events require an utterance_key")
    if event_type == "analysis.delta" and not utterance_key:
        raise ValueError("analysis events require an utterance_key")

    event_index = _next_event_index(session, live_session.id)
    combined_payload = dict(payload or {})
    if analysis is not None:
        combined_payload["analysis"] = analysis

    event = LiveTranscriptEvent(
        live_session_id=live_session.id,
        event_index=event_index,
        event_type=event_type,
        utterance_key=utterance_key,
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
        speaker_id=speaker_id,
        is_final=is_final,
        payload=combined_payload,
    )
    session.add(event)
    session.flush()

    if event_type == "transcript.final":
        _upsert_sentence_from_live_event(session, live_session, event)
    elif event_type == "analysis.delta":
        _upsert_analysis_from_live_event(session, live_session, utterance_key, analysis)

    session.commit()
    session.refresh(event)
    return event


def stop_live_session(session: Session, live_session: LiveSession) -> LiveSession:
    if live_session.status not in {"open", "recording", "stopped"}:
        raise ValueError("live session cannot be stopped from its current status")

    live_session.status = "stopped"
    if live_session.stopped_at is None:
        live_session.stopped_at = utcnow()

    session.commit()
    session.refresh(live_session)
    return live_session


def finalize_live_session(session: Session, live_session: LiveSession) -> LiveSession:
    if live_session.status not in _FINALIZABLE_STATUSES:
        raise ValueError("live session must be stopped before finalization")
    if live_session.total_bytes <= 0:
        raise ValueError("live session has no recorded media")

    live_session.status = "finalizing"
    if live_session.stopped_at is None:
        live_session.stopped_at = utcnow()

    media_path = Path(live_session.storage_path)
    if not media_path.exists():
        raise FileNotFoundError(f"live media not found: {live_session.storage_path}")

    if live_session.media_asset is not None:
        live_session.media_asset.checksum = checksum_for_local_file(str(media_path))
        live_session.media_asset.ingest_metadata = {
            **(live_session.media_asset.ingest_metadata or {}),
            "live_session_status": "completed",
            "received_chunks": live_session.received_chunks,
            "total_bytes": live_session.total_bytes,
        }

    if live_session.transcript is not None:
        live_session.transcript.transcript_metadata = {
            **(live_session.transcript.transcript_metadata or {}),
            "live_session_status": "completed",
        }

    live_session.status = "completed"
    live_session.finalized_at = utcnow()
    live_session.error_message = None

    session.commit()
    session.refresh(live_session)
    return live_session


def list_live_events(session: Session, live_session_id: str) -> list[LiveTranscriptEvent]:
    return session.scalars(
        select(LiveTranscriptEvent)
        .where(LiveTranscriptEvent.live_session_id == live_session_id)
        .order_by(LiveTranscriptEvent.event_index.asc())
    ).all()


def _build_storage_filename(session_id: str, original_filename: str | None, mime_type: str | None) -> str:
    provided_suffix = Path(original_filename or "").suffix
    suffix = provided_suffix or _MIME_EXTENSIONS.get((mime_type or "").lower(), ".bin")
    return f"{session_id}{suffix}"


def _next_event_index(session: Session, live_session_id: str) -> int:
    current = session.scalar(
        select(func.max(LiveTranscriptEvent.event_index)).where(LiveTranscriptEvent.live_session_id == live_session_id)
    )
    return 0 if current is None else int(current) + 1


def _upsert_sentence_from_live_event(session: Session, live_session: LiveSession, event: LiveTranscriptEvent) -> None:
    transcript = session.get(Transcript, live_session.transcript_id)
    if transcript is None or event.utterance_key is None or event.text is None:
        return

    existing_sentence = None
    for sentence in session.scalars(
        select(SentenceUnit)
        .where(SentenceUnit.transcript_id == transcript.id)
        .order_by(SentenceUnit.utterance_index.asc())
    ).all():
        if sentence.sentence_metadata.get("live_utterance_key") == event.utterance_key:
            existing_sentence = sentence
            break

    if existing_sentence is None:
        max_index = session.scalar(
            select(func.max(SentenceUnit.utterance_index)).where(SentenceUnit.transcript_id == transcript.id)
        )
        existing_sentence = SentenceUnit(
            transcript_id=transcript.id,
            utterance_index=0 if max_index is None else int(max_index) + 1,
            start_ms=event.start_ms or 0,
            end_ms=event.end_ms or event.start_ms or 0,
            text=event.text,
            speaker_id=event.speaker_id,
            speaker_confidence=None,
            source_segment_ids=[],
            sentence_metadata={
                "live_utterance_key": event.utterance_key,
                "live_is_final": True,
            },
        )
        session.add(existing_sentence)
    else:
        existing_sentence.start_ms = event.start_ms or existing_sentence.start_ms
        existing_sentence.end_ms = event.end_ms or existing_sentence.end_ms
        existing_sentence.text = event.text
        existing_sentence.speaker_id = event.speaker_id
        existing_sentence.sentence_metadata = {
            **(existing_sentence.sentence_metadata or {}),
            "live_utterance_key": event.utterance_key,
            "live_is_final": True,
        }

    session.flush()
    transcript.full_text = _build_transcript_full_text(session, transcript.id)
    update_transcript_archive_preview(transcript)


def _upsert_analysis_from_live_event(
    session: Session,
    live_session: LiveSession,
    utterance_key: str | None,
    analysis: dict | None,
) -> None:
    if utterance_key is None or analysis is None or live_session.transcript_id is None:
        return

    target_sentence = None
    for sentence in session.scalars(
        select(SentenceUnit)
        .where(SentenceUnit.transcript_id == live_session.transcript_id)
        .order_by(SentenceUnit.utterance_index.asc())
    ).all():
        if sentence.sentence_metadata.get("live_utterance_key") == utterance_key:
            target_sentence = sentence
            break

    if target_sentence is None:
        raise ValueError("cannot attach analysis before a finalized transcript sentence exists")

    transcript = session.get(Transcript, live_session.transcript_id)
    existing_result = session.scalar(
        select(AnalysisResult).where(AnalysisResult.sentence_unit_id == target_sentence.id)
    )
    if existing_result is None:
        session.add(
            AnalysisResult(
                sentence_unit_id=target_sentence.id,
                politeness_score=float(analysis["politeness_score"]),
                semantic_confidence_score=float(analysis["semantic_confidence_score"]),
                main_message_likelihood=float(analysis["main_message_likelihood"]),
                analysis_model=str(analysis.get("analysis_model") or "live-heuristic:mvp"),
                analysis_payload=dict(analysis.get("analysis_payload") or {}),
            )
        )
        if transcript is not None:
            session.flush()
            update_transcript_archive_preview(transcript)
        return

    existing_result.politeness_score = float(analysis["politeness_score"])
    existing_result.semantic_confidence_score = float(analysis["semantic_confidence_score"])
    existing_result.main_message_likelihood = float(analysis["main_message_likelihood"])
    existing_result.analysis_model = str(analysis.get("analysis_model") or existing_result.analysis_model)
    existing_result.analysis_payload = dict(analysis.get("analysis_payload") or {})
    if transcript is not None:
        update_transcript_archive_preview(transcript)


def _build_transcript_full_text(session: Session, transcript_id: str) -> str:
    sentences = session.scalars(
        select(SentenceUnit)
        .where(SentenceUnit.transcript_id == transcript_id)
        .order_by(SentenceUnit.utterance_index.asc(), SentenceUnit.start_ms.asc())
    ).all()
    return " ".join(sentence.text.strip() for sentence in sentences if sentence.text.strip()).strip()
