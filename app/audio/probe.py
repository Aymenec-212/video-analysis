"""Media inspection via ffprobe (SPEC 5.1, 5.2).

This module reports facts and does not decide policy. It answers "what is this
file" — duration, streams, format — and raises only when the file cannot be read
at all.

The distinction matters for one measured case: a video with no audio track probes
*successfully*. FFmpeg only fails later, with an opaque exit code and the message
"Output file does not contain any stream". Probing first turns that into a
precise `NO_AUDIO_STREAM` instead of a misleading `UNREADABLE_MEDIA`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.core.errors import UnreadableMediaError
from app.core.logging import get_logger
from app.core.process import CommandTimeout, run_command

logger = get_logger(__name__)

#: Probing reads headers, not the whole file. A file that cannot be described in
#: this long is not one we want to spend a request on.
PROBE_TIMEOUT_SEC = 60.0


class AudioStreamInfo(BaseModel):
    """A single audio stream as reported by ffprobe."""

    index: int
    codec_name: str | None = None
    sample_rate: int | None = None
    channels: int | None = None


class MediaProbe(BaseModel):
    """Structured ffprobe output.

    `duration` is optional because some containers genuinely do not carry one.
    Reporting `None` is honest; inventing a number to keep the type simple would
    be the same failure mode the brief warns about, in miniature.
    """

    duration: float | None = None
    size_bytes: int = 0
    format_name: str | None = None
    audio_streams: list[AudioStreamInfo] = []
    video_stream_count: int = 0

    @property
    def has_audio(self) -> bool:
        return bool(self.audio_streams)

    @property
    def has_video(self) -> bool:
        return self.video_stream_count > 0


def _coerce_float(value: Any) -> float | None:
    """ffprobe reports numbers as strings, and sometimes as 'N/A'."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    # Negative or non-finite durations appear on malformed headers.
    return parsed if parsed > 0 and parsed == parsed and parsed != float("inf") else None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_duration(payload: dict[str, Any], streams: list[dict[str, Any]]) -> float | None:
    """Resolve duration from the container, falling back to stream durations.

    Container-level duration is present for ordinary files. Some streamed or
    remuxed containers omit it while individual streams still carry one, so the
    longest stream is used as a fallback before giving up.
    """
    container = _coerce_float(payload.get("format", {}).get("duration"))
    if container is not None:
        return container

    stream_durations = [
        value for value in (_coerce_float(s.get("duration")) for s in streams) if value is not None
    ]
    return max(stream_durations) if stream_durations else None


async def probe_media(path: Path) -> MediaProbe:
    """Inspect a media file.

    Raises `UnreadableMediaError` when ffprobe cannot decode the file — which
    covers corrupt data, truncated downloads, non-media files with a media
    extension, and missing paths. An extension proves nothing, which is why
    uploads are validated here rather than by filename.
    """
    log = logger.bind(path=str(path))

    try:
        result = await run_command(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                path,
            ],
            timeout_sec=PROBE_TIMEOUT_SEC,
        )
    except CommandTimeout as exc:
        raise UnreadableMediaError(
            "The media file could not be inspected within the time limit.",
            detail={"timeout_sec": exc.timeout_sec},
            cause=exc,
        ) from exc

    if not result.ok:
        log.warning("ffprobe rejected the file", returncode=result.returncode)
        raise UnreadableMediaError(
            "The media file could not be decoded.",
            detail={"ffprobe_error": result.stderr_summary()},
        )

    try:
        payload: dict[str, Any] = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        # ffprobe exited cleanly but produced unparseable output — treat as
        # unreadable rather than crashing on a malformed response.
        raise UnreadableMediaError(
            "ffprobe returned output that could not be parsed.", cause=exc
        ) from exc

    streams: list[dict[str, Any]] = payload.get("streams", []) or []

    audio_streams = [
        AudioStreamInfo(
            index=_coerce_int(stream.get("index")) or 0,
            codec_name=stream.get("codec_name"),
            sample_rate=_coerce_int(stream.get("sample_rate")),
            channels=_coerce_int(stream.get("channels")),
        )
        for stream in streams
        if stream.get("codec_type") == "audio"
    ]

    probe = MediaProbe(
        duration=_extract_duration(payload, streams),
        size_bytes=_coerce_int(payload.get("format", {}).get("size")) or 0,
        format_name=payload.get("format", {}).get("format_name"),
        audio_streams=audio_streams,
        video_stream_count=sum(1 for s in streams if s.get("codec_type") == "video"),
    )

    log.debug(
        "probed media",
        duration=probe.duration,
        audio_streams=len(probe.audio_streams),
        has_video=probe.has_video,
    )
    return probe


__all__ = ["PROBE_TIMEOUT_SEC", "AudioStreamInfo", "MediaProbe", "probe_media"]
