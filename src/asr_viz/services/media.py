from __future__ import annotations

from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse
from urllib.request import urlretrieve

from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse

from asr_viz.core.settings import settings
from asr_viz.models.job import ProcessingJob
from asr_viz.models.media import MediaAsset


def project_root_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def media_storage_root() -> Path:
    configured_path = Path(settings.media_storage_dir).expanduser()
    if configured_path.is_absolute():
        return configured_path
    return (project_root_dir() / configured_path).resolve()


def resolve_local_media_path(path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate

    project_relative_candidate = (project_root_dir() / candidate).resolve()
    if project_relative_candidate.exists():
        return project_relative_candidate

    storage_relative_candidate = (media_storage_root() / candidate).resolve()
    if storage_relative_candidate.exists():
        return storage_relative_candidate

    return project_relative_candidate


def configured_storage_backend() -> str:
    return str(getattr(settings, "storage_backend", "local") or "local").strip().lower()


def upload_local_file_to_configured_storage(
    *,
    local_path: str,
    owner_user_id: str | None,
    media_category: str,
    media_id: str,
    original_filename: str | None,
    mime_type: str | None,
) -> tuple[str, str]:
    if configured_storage_backend() != "r2":
        return "file", str(resolve_local_media_path(local_path))

    path = resolve_local_media_path(local_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"media file not found: {local_path}")

    object_key = _build_r2_object_key(
        owner_user_id=owner_user_id,
        media_category=media_category,
        media_id=media_id,
        original_filename=original_filename,
        mime_type=mime_type,
    )
    _upload_local_file_to_r2(path, object_key, mime_type=mime_type)
    return "r2", _build_r2_uri(object_key)


def resolve_media_source(job: ProcessingJob, media_asset: MediaAsset) -> str:
    if media_asset.source_type == "r2" or media_asset.source_uri.startswith("r2://"):
        storage_dir = media_storage_root() / "r2_cache"
        storage_dir.mkdir(parents=True, exist_ok=True)
        object_key = _parse_r2_uri(media_asset.source_uri)[1]
        suffix = Path(object_key).suffix or Path(media_asset.source_uri).suffix or ".bin"
        destination = storage_dir / f"{job.id}{suffix}"
        if not destination.exists():
            _download_r2_uri_to_path(media_asset.source_uri, destination)
        return str(destination)

    if media_asset.source_type == "file":
        path = resolve_local_media_path(media_asset.source_uri)
        if not path.exists():
            raise FileNotFoundError(f"media file not found: {media_asset.source_uri}")
        return str(path)

    if media_asset.source_type == "url":
        storage_dir = media_storage_root()
        storage_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(urlparse(media_asset.source_uri).path).suffix or ".bin"
        destination = storage_dir / f"{job.id}{suffix}"
        if not destination.exists():
            urlretrieve(media_asset.source_uri, destination)
        return str(destination)

    return media_asset.source_uri


def build_media_response(media_asset: MediaAsset, *, filename: str | None = None):
    source_uri = media_asset.source_uri
    if source_uri.startswith(("http://", "https://")):
        return RedirectResponse(url=source_uri)

    if media_asset.source_type == "r2" or source_uri.startswith("r2://"):
        headers = {}
        resolved_filename = filename or Path(source_uri).name
        if resolved_filename:
            headers["content-disposition"] = f'inline; filename="{resolved_filename}"'
        media_type = media_asset.mime_type
        if not media_type:
            media_type = _get_r2_object_metadata(source_uri).get("content_type")
        return StreamingResponse(
            _iter_r2_uri_chunks(source_uri),
            media_type=media_type,
            headers=headers,
        )

    media_path = resolve_local_media_path(source_uri)
    if not media_path.exists() or not media_path.is_file():
        raise FileNotFoundError(f"media file not found: {source_uri}")

    return FileResponse(
        path=media_path,
        media_type=media_asset.mime_type,
        filename=filename or media_path.name,
    )


def checksum_for_local_file(source_uri: str) -> str | None:
    path = resolve_local_media_path(source_uri)
    if not path.exists() or not path.is_file():
        return None

    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_local_media_file(source_uri: str) -> None:
    path = resolve_local_media_path(source_uri)
    if not path.exists():
        return

    with suppress(OSError):
        path.unlink()


def _build_r2_object_key(
    *,
    owner_user_id: str | None,
    media_category: str,
    media_id: str,
    original_filename: str | None,
    mime_type: str | None,
) -> str:
    safe_owner = _slug_storage_token(owner_user_id or "anonymous")
    safe_category = _slug_storage_token(media_category)
    suffix = Path(original_filename or "").suffix or _mime_extension(mime_type)
    return f"{safe_category}/{safe_owner}/{media_id}{suffix}"


def _slug_storage_token(value: str) -> str:
    normalized = "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in value)
    collapsed = "-".join(part for part in normalized.split("-") if part)
    return collapsed or "media"


def _mime_extension(mime_type: str | None) -> str:
    extensions = {
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/webm": ".webm",
        "text/plain": ".txt",
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/webm": ".webm",
    }
    return extensions.get((mime_type or "").lower(), ".bin")


def _build_r2_uri(object_key: str) -> str:
    bucket_name = getattr(settings, "r2_bucket_name", None)
    if not bucket_name:
        raise RuntimeError("R2_BUCKET_NAME must be configured for R2 storage")
    return f"r2://{bucket_name}/{object_key}"


def _parse_r2_uri(source_uri: str) -> tuple[str, str]:
    if not source_uri.startswith("r2://"):
        raise ValueError(f"unsupported R2 URI: {source_uri}")
    remainder = source_uri.removeprefix("r2://")
    bucket_name, separator, object_key = remainder.partition("/")
    if not separator or not bucket_name or not object_key:
        raise ValueError(f"invalid R2 URI: {source_uri}")
    return bucket_name, object_key


def _upload_local_file_to_r2(local_path: Path, object_key: str, *, mime_type: str | None = None) -> None:
    client = _build_r2_client()
    extra_args: dict[str, str] = {}
    if mime_type:
        extra_args["ContentType"] = mime_type
    with local_path.open("rb") as handle:
        if extra_args:
            client.upload_fileobj(handle, getattr(settings, "r2_bucket_name"), object_key, ExtraArgs=extra_args)
        else:
            client.upload_fileobj(handle, getattr(settings, "r2_bucket_name"), object_key)


def _download_r2_uri_to_path(source_uri: str, destination: Path) -> None:
    bucket_name, object_key = _parse_r2_uri(source_uri)
    client = _build_r2_client()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        client.download_fileobj(bucket_name, object_key, handle)


def _get_r2_object_metadata(source_uri: str) -> dict[str, str | None]:
    bucket_name, object_key = _parse_r2_uri(source_uri)
    client = _build_r2_client()
    response = client.head_object(Bucket=bucket_name, Key=object_key)
    return {
        "content_type": response.get("ContentType"),
    }


def _iter_r2_uri_chunks(source_uri: str) -> Iterator[bytes]:
    bucket_name, object_key = _parse_r2_uri(source_uri)
    client = _build_r2_client()
    response = client.get_object(Bucket=bucket_name, Key=object_key)
    body = response["Body"]
    try:
        yield from body.iter_chunks(chunk_size=8192)
    finally:
        body.close()


def _build_r2_client():
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError("boto3 is required for R2-backed storage") from exc

    account_id = getattr(settings, "r2_account_id", None)
    bucket_name = getattr(settings, "r2_bucket_name", None)
    access_key_id = getattr(settings, "r2_access_key_id", None)
    secret_access_key = getattr(settings, "r2_secret_access_key", None)
    if not all([account_id, bucket_name, access_key_id, secret_access_key]):
        raise RuntimeError("R2 storage is enabled, but R2 credentials are incomplete")

    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )
