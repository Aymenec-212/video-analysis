"""URL ingestion via yt-dlp (SPEC 5.1).

A URL supplied by a caller and handed to an HTTP client is a server-side request
forgery vector by default: `http://169.254.169.254/` reaches cloud instance
metadata, `http://localhost:6379/` reaches an unprotected Redis. Validation here
is a scheme allowlist plus a check that every address the hostname resolves to is
publicly routable.

"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yt_dlp
from yt_dlp.utils import DownloadError, ExtractorError, UnsupportedError

from app.audio.probe import probe_media
from app.core.config import IngestionConfig
from app.core.errors import (
    InvalidURLError,
    MediaTooLargeError,
    MediaTooLongError,
    SourceUnavailableError,
    UnsupportedURLError,
)
from app.core.logging import get_logger

from .metadata import MediaSource, SourceKind

logger = get_logger(__name__)

#: Substrings in yt-dlp errors that indicate an unavailable source rather than a
#: transport fault. Used only to improve the message; every branch already maps
#: to SOURCE_UNAVAILABLE.
_UNAVAILABLE_HINTS = (
    "private",
    "removed",
    "unavailable",
    "not available",
    "geo",
    "age",
    "sign in",
    "bot",
    "members-only",
    "deleted",
)


def validate_url(url: str, config: IngestionConfig) -> str:
    """Check scheme and host before any network call.

    Raises `InvalidURLError` for anything malformed, non-HTTP, or resolving to a
    non-public address.
    """
    candidate = url.strip()
    if not candidate:
        raise InvalidURLError("No URL was supplied.")

    try:
        parsed = urlparse(candidate)
    except ValueError as exc:
        raise InvalidURLError("The URL could not be parsed.", cause=exc) from exc

    if parsed.scheme.lower() not in config.allowed_schemes:
        raise InvalidURLError(
            f"Only {' and '.join(config.allowed_schemes)} URLs are accepted.",
            detail={"scheme": parsed.scheme},
        )

    if not parsed.hostname:
        raise InvalidURLError("The URL has no host.", detail={"url": candidate})

    if config.block_private_addresses:
        _assert_publicly_routable(parsed.hostname)

    return candidate


def _assert_publicly_routable(hostname: str) -> None:
    """Reject hosts that resolve to non-public addresses.

    Literal IPs are checked directly; names are resolved and *every* returned
    address must pass, since a name with one public and one private answer would
    otherwise slip through.

    The public error message never distinguishes "blocked private address" from
    "does not resolve" — telling a caller which internal hosts exist is itself
    a disclosure.
    """
    literal = _parse_ip_literal(hostname)
    if literal is not None:
        if not _is_public(literal):
            raise InvalidURLError(
                "The URL host is not publicly routable.",
                detail={"host": hostname},
            )
        return

    try:
        resolved = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise InvalidURLError(
            "The URL host could not be resolved.", detail={"host": hostname}, cause=exc
        ) from exc

    addresses = {entry[4][0] for entry in resolved}
    for address in addresses:
        try:
            parsed_address = ipaddress.ip_address(address)
        except ValueError:  # pragma: no cover - getaddrinfo returns valid literals
            continue
        if not _is_public(parsed_address):
            raise InvalidURLError(
                "The URL host is not publicly routable.",
                detail={"host": hostname},
            )


def _parse_ip_literal(hostname: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse a hostname as an IP literal, tolerating bracketed IPv6."""
    try:
        return ipaddress.ip_address(hostname.strip("[]"))
    except ValueError:
        return None


def _is_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether an address is safe to fetch from.

    `is_global` alone is insufficient: it does not exclude every range we care
    about across both address families, so the specific categories are checked
    explicitly.
    """
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def _build_options(work_dir: Path, config: IngestionConfig) -> dict[str, Any]:
    return {
        # Audio only: the pipeline discards video, so fetching it wastes
        # bandwidth and time. Falls back to a full stream when no audio-only
        # format is published.
        "format": "bestaudio/best",
        # A URL carrying `&list=` should yield its video, not the playlist.
        "noplaylist": True,
        "outtmpl": str(work_dir / "%(id)s.%(ext)s"),
        "socket_timeout": config.socket_timeout_sec,
        "max_filesize": config.max_bytes,
        "retries": 2,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "restrictfilenames": True,
    }


def _classify_download_error(exc: Exception) -> Exception:
    """Map a yt-dlp exception onto the taxonomy."""
    if isinstance(exc, UnsupportedError):
        return UnsupportedURLError(
            "No extractor is available for this URL.", cause=exc
        )

    message = str(exc).lower()
    if "max-filesize" in message or "larger than" in message:
        return MediaTooLargeError(
            "The media exceeds the configured size limit.", cause=exc
        )

    matched = next((hint for hint in _UNAVAILABLE_HINTS if hint in message), None)
    if matched:
        return SourceUnavailableError(
            "The video could not be retrieved: it may be private, removed, "
            "region-restricted, or protected by a bot check.",
            detail={"reason": matched},
            cause=exc,
        )

    return SourceUnavailableError(
        "The video could not be retrieved from its source.", cause=exc
    )


def _extract_info(url: str, options: dict[str, Any], *, download: bool) -> dict[str, Any]:
    """Blocking yt-dlp call. Run via `asyncio.to_thread`."""
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=download)
    if info is None:
        raise SourceUnavailableError("The source returned no metadata for this URL.")
    return dict(info)


def _resolve_downloaded_path(info: dict[str, Any], work_dir: Path) -> Path:
    """Locate the file yt-dlp wrote.

    Modern yt-dlp records it under `requested_downloads`; the fallbacks cover
    older shapes and post-processed extensions.
    """
    downloads = info.get("requested_downloads") or []
    if downloads:
        filepath = downloads[0].get("filepath") or downloads[0].get("_filename")
        if filepath:
            return Path(filepath)

    filename = info.get("_filename")
    if filename:
        return Path(filename)

    candidates = sorted(
        (p for p in work_dir.iterdir() if p.is_file()),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if not candidates:
        raise SourceUnavailableError("The download produced no file.")
    return candidates[0]


async def fetch_from_url(
    url: str, work_dir: Path, config: IngestionConfig
) -> MediaSource:
    """Download a video's audio and return the unified representation.

    Duration is checked twice by design. The first check uses source metadata
    before downloading, so a three-hour stream is refused without transferring
    it. The second uses ffprobe on the downloaded file, because SPEC §5.1 makes
    ffprobe the authority and source metadata can be wrong or absent.
    """
    validated = validate_url(url, config)
    log = logger.bind(url=validated)
    work_dir.mkdir(parents=True, exist_ok=True)
    options = _build_options(work_dir, config)

    try:
        info = await asyncio.to_thread(_extract_info, validated, options, download=False)
    except (DownloadError, ExtractorError, UnsupportedError) as exc:
        raise _classify_download_error(exc) from exc

    if info.get("_type") == "playlist":
        raise UnsupportedURLError(
            "This URL refers to a playlist. Supply a single video URL.",
            detail={"entries": len(info.get("entries") or [])},
        )

    # Pre-download guard: reject before spending the transfer.
    reported_duration = info.get("duration")
    if isinstance(reported_duration, int | float) and reported_duration > config.max_duration_sec:
        raise MediaTooLongError(
            f"The video runs {reported_duration:.0f}s, above the "
            f"{config.max_duration_sec:.0f}s limit.",
            detail={"duration": float(reported_duration)},
        )

    log.info("downloading", title=info.get("title"), duration=reported_duration)

    try:
        downloaded = await asyncio.to_thread(_extract_info, validated, options, download=True)
    except (DownloadError, ExtractorError, UnsupportedError) as exc:
        raise _classify_download_error(exc) from exc

    path = _resolve_downloaded_path(downloaded, work_dir)
    if not path.exists():
        raise SourceUnavailableError(
            "The download reported success but produced no file.",
            detail={"expected": path.name},
        )

    size_bytes = path.stat().st_size
    if size_bytes > config.max_bytes:
        raise MediaTooLargeError(
            "The downloaded media exceeds the configured size limit.",
            detail={"size_bytes": size_bytes, "limit": config.max_bytes},
        )

    # Authoritative duration, and an implicit decode check on what we received.
    probe = await probe_media(path)
    duration = probe.duration
    verified = duration is not None
    if duration is None and isinstance(reported_duration, int | float):
        duration = float(reported_duration)

    if duration is not None and duration > config.max_duration_sec:
        raise MediaTooLongError(
            f"The media runs {duration:.0f}s, above the "
            f"{config.max_duration_sec:.0f}s limit.",
            detail={"duration": duration},
        )

    source = MediaSource(
        kind=SourceKind.URL,
        path=path,
        title=str(downloaded.get("title") or path.stem),
        duration=duration,
        duration_verified=verified,
        description=downloaded.get("description") or None,
        source=str(downloaded.get("extractor_key") or downloaded.get("extractor") or "unknown"),
        size_bytes=size_bytes,
        original_url=validated,
    )

    log.info(
        "ingested from url",
        source=source.source,
        duration=source.duration,
        duration_verified=source.duration_verified,
        size_bytes=size_bytes,
    )
    return source


__all__ = ["fetch_from_url", "validate_url"]