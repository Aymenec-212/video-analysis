"""Uploaded file ingestion (SPEC 5.1).

we kept in mind to domain rules for this feature

**Never trust the extension.** A file named `.mp4` proves nothing about its
contents. Validation is by probing the actual data, so a renamed archive or a
truncated download is rejected at the boundary with `UNREADABLE_MEDIA` rather
than producing a confusing failure three stages later.

**Never buffer the whole upload.** Bytes are streamed to disk with the size cap
enforced *during* the write, and the partial file is removed on breach. Reading
an upload into memory to measure it means a large upload is a denial of service
against the process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.audio.probe import probe_media
from app.core.config import IngestionConfig
from app.core.errors import MediaTooLargeError, MediaTooLongError, UnreadableMediaError
from app.core.logging import get_logger

from .metadata import UPLOAD_SOURCE, MediaSource, SourceKind

logger = get_logger(__name__)

#: 1 MiB balances syscall overhead against peak memory per concurrent upload.
UPLOAD_CHUNK_BYTES = 1024 * 1024


class AsyncByteReader(Protocol):
    """The subset of a streaming upload this module needs.

    Structural rather than nominal so it is satisfied by FastAPI's `UploadFile`
    and by a plain test double, without either importing the other.
    """

    async def read(self, size: int = -1) -> bytes: ...


def _safe_stem(filename: str | None) -> str:
    """Derive a display title from an uploaded filename.

    Only the stem is kept: directory components in a supplied filename are a
    path-traversal attempt, not information.
    """
    if not filename:
        return "Uploaded media"
    stem = Path(filename).name.rsplit(".", 1)[0].strip()
    return stem or "Uploaded media"


async def save_upload(
    reader: AsyncByteReader,
    destination: Path,
    max_bytes: int,
) -> Path:
    """Stream an upload to disk, enforcing the size cap as it is written.

    Raises `MediaTooLargeError` the moment the cap is exceeded, having removed
    the partial file — an aborted upload should not leave bytes on disk.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    try:
        with destination.open("wb") as handle:
            while chunk := await reader.read(UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > max_bytes:
                    raise MediaTooLargeError(
                        "The uploaded file exceeds the configured size limit.",
                        detail={"limit_bytes": max_bytes},
                    )
                handle.write(chunk)
    except MediaTooLargeError:
        destination.unlink(missing_ok=True)
        raise

    if written == 0:
        destination.unlink(missing_ok=True)
        raise UnreadableMediaError("The uploaded file is empty.")

    return destination


async def load_local_file(
    path: Path, config: IngestionConfig, *, original_filename: str | None = None
) -> MediaSource:
    """Validate a file already on disk and build the unified representation.

    Used both for genuine local files and for uploads already streamed to disk,
    so the two share one validation path.
    """
    log = logger.bind(path=str(path))

    if not path.exists() or not path.is_file():
        raise UnreadableMediaError(
            "The media file does not exist.", detail={"path": path.name}
        )

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise UnreadableMediaError("The media file is empty.")
    if size_bytes > config.max_bytes:
        raise MediaTooLargeError(
            "The media file exceeds the configured size limit.",
            detail={"size_bytes": size_bytes, "limit": config.max_bytes},
        )

    # Probing is the validation. Raises UNREADABLE_MEDIA for anything that is
    # not decodable media, whatever it claims to be by name.
    probe = await probe_media(path)

    if probe.duration is not None and probe.duration > config.max_duration_sec:
        raise MediaTooLongError(
            f"The media runs {probe.duration:.0f}s, above the "
            f"{config.max_duration_sec:.0f}s limit.",
            detail={"duration": probe.duration},
        )

    source = MediaSource(
        kind=SourceKind.UPLOAD,
        path=path,
        title=_safe_stem(original_filename or path.name),
        duration=probe.duration,
        # Uploads have no competing metadata source: ffprobe is the only
        # authority, so a present duration is always a verified one.
        duration_verified=probe.duration is not None,
        # An uploaded file carries no description, and inferring one would be
        # exactly the fabrication the brief forbids.
        description=None,
        source=UPLOAD_SOURCE,
        size_bytes=size_bytes,
        original_url=None,
    )

    log.info(
        "ingested upload",
        title=source.title,
        duration=source.duration,
        size_bytes=size_bytes,
    )
    return source


__all__ = [
    "UPLOAD_CHUNK_BYTES",
    "AsyncByteReader",
    "load_local_file",
    "save_upload",
]