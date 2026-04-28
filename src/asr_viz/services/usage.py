from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from asr_viz.core.settings import settings
from asr_viz.models.live import LiveSession
from asr_viz.models.media import MediaAsset
from asr_viz.models.stream import StreamIngestionSession


class StorageQuotaExceededError(ValueError):
    """Raised when a user attempts to exceed the configured storage quota."""


class UploadTooLargeError(ValueError):
    """Raised when a single upload exceeds the configured max file size."""


def ensure_upload_within_limits(
    session: Session,
    *,
    owner_user_id: str,
    current_upload_bytes: int,
    incoming_chunk_bytes: int,
) -> None:
    next_upload_size = current_upload_bytes + incoming_chunk_bytes
    if next_upload_size > settings.max_upload_size_bytes:
        raise UploadTooLargeError(
            f"upload exceeds max size of {settings.max_upload_size_bytes} bytes"
        )

    current_usage = compute_user_storage_usage(session, owner_user_id)
    if current_usage + incoming_chunk_bytes > settings.per_user_storage_quota_bytes:
        raise StorageQuotaExceededError(
            f"user storage exceeds quota of {settings.per_user_storage_quota_bytes} bytes"
        )


def compute_user_storage_usage(session: Session, owner_user_id: str) -> int:
    media_total = session.scalar(
        select(func.coalesce(func.sum(MediaAsset.size_bytes), 0)).where(MediaAsset.owner_user_id == owner_user_id)
    )
    pending_stream_total = session.scalar(
        select(func.coalesce(func.sum(StreamIngestionSession.total_bytes), 0)).where(
            StreamIngestionSession.owner_user_id == owner_user_id,
            StreamIngestionSession.processing_job_id.is_(None),
        )
    )
    in_progress_live_total = session.scalar(
        select(func.coalesce(func.sum(LiveSession.total_bytes), 0)).where(
            LiveSession.owner_user_id == owner_user_id,
            LiveSession.finalized_at.is_(None),
        )
    )
    return int(media_total or 0) + int(pending_stream_total or 0) + int(in_progress_live_total or 0)
