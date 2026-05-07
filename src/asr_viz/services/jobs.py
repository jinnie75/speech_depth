from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from asr_viz.core.settings import settings
from asr_viz.models.job import ProcessingJob
from asr_viz.models.media import MediaAsset
from asr_viz.models.common import JobStage, JobStatus
from asr_viz.services.media import checksum_for_local_file


def create_job(
    session: Session,
    *,
    owner_user_id: str | None = None,
    source_uri: str,
    source_type: str,
    mime_type: str | None,
    checksum: str | None,
    ingest_metadata: dict | None,
) -> ProcessingJob:
    resolved_owner_user_id = owner_user_id or settings.auth_dev_user_id
    resolved_checksum = checksum or checksum_for_local_file(source_uri)
    size_bytes = _resolve_size_bytes(source_uri, source_type)
    media_asset = MediaAsset(
        owner_user_id=resolved_owner_user_id,
        source_uri=source_uri,
        source_type=source_type,
        mime_type=mime_type,
        checksum=resolved_checksum,
        size_bytes=size_bytes,
        ingest_metadata=ingest_metadata or {},
    )
    session.add(media_asset)
    session.flush()

    job = ProcessingJob(
        owner_user_id=resolved_owner_user_id,
        media_asset_id=media_asset.id,
        status=JobStatus.QUEUED.value,
        current_stage=JobStage.INGESTION.value,
        stage_details={"status": "queued"},
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def infer_source_type(source_uri: str) -> str:
    if source_uri.startswith(("http://", "https://")):
        return "url"
    if Path(source_uri).exists():
        return "file"
    return "opaque"


def _resolve_size_bytes(source_uri: str, source_type: str) -> int | None:
    if source_type != "file":
        return None

    path = Path(source_uri)
    if not path.exists() or not path.is_file():
        return None
    return path.stat().st_size
