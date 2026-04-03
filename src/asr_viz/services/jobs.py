from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from asr_viz.models.job import ProcessingJob
from asr_viz.models.media import MediaAsset
from asr_viz.models.common import JobStage, JobStatus
from asr_viz.services.media import checksum_for_local_file


def create_job(
    session: Session,
    *,
    source_uri: str,
    source_type: str,
    diarization_enabled: bool,
    mime_type: str | None,
    checksum: str | None,
    ingest_metadata: dict | None,
) -> ProcessingJob:
    resolved_checksum = checksum or checksum_for_local_file(source_uri)
    media_asset = MediaAsset(
        source_uri=source_uri,
        source_type=source_type,
        mime_type=mime_type,
        checksum=resolved_checksum,
        ingest_metadata=ingest_metadata or {},
    )
    session.add(media_asset)
    session.flush()

    job = ProcessingJob(
        media_asset_id=media_asset.id,
        status=JobStatus.QUEUED.value,
        current_stage=JobStage.INGESTION.value,
        diarization_enabled=diarization_enabled,
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
