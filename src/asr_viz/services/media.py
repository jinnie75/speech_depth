from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

from asr_viz.core.settings import settings
from asr_viz.models.job import ProcessingJob
from asr_viz.models.media import MediaAsset


def resolve_media_source(job: ProcessingJob, media_asset: MediaAsset) -> str:
    if media_asset.source_type == "file":
        path = Path(media_asset.source_uri)
        if not path.exists():
            raise FileNotFoundError(f"media file not found: {media_asset.source_uri}")
        return str(path)

    if media_asset.source_type == "url":
        storage_dir = Path(settings.media_storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(urlparse(media_asset.source_uri).path).suffix or ".bin"
        destination = storage_dir / f"{job.id}{suffix}"
        if not destination.exists():
            urlretrieve(media_asset.source_uri, destination)
        return str(destination)

    return media_asset.source_uri


def checksum_for_local_file(source_uri: str) -> str | None:
    path = Path(source_uri)
    if not path.exists() or not path.is_file():
        return None

    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()
