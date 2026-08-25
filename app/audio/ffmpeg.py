"""Audio extraction and normalisation (SPEC 5.2).

Every input, whatever container or codec it arrived in, is reduced to one
representation: **mono, 16 kHz, FLAC**.

*Mono* because stereo is just redundant data, and diarization
can distinguish speakers via voice, regardless of channel type.
 *16 kHz* because it is the native rate of the speech models.

*FLAC* because it is lossless.

**No denoising by default.**

"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel

from app.core.config import AudioConfig
from app.core.errors import NoAudioStreamError, UnreadableMediaError
from app.core.logging import get_logger
from app.core.process import CommandTimeout, run_command

from .probe import probe_media

logger = get_logger(__name__)

#: Transcoding is bounded by input length. Generous enough for a long file on a
#: slow CPU, short enough that a stuck process cannot hold a request forever.
FFMPEG_TIMEOUT_SEC = 900.0

_HASH_CHUNK_BYTES = 1024 * 1024


class NormalizedAudio(BaseModel):
    """The canonical audio artefact handed to the speech backend."""

    path: Path
    #: SHA-256 of the normalised bytes — the AD-2 cache key. Reproducible across
    #: machines and FFmpeg versions because the encode is bitexact.
    sha256: str
    duration: float | None
    size_bytes: int
    sample_rate: int
    channels: int


def compute_sha256(path: Path) -> str:
    """Hash a file in chunks.

    Read incrementally so hashing a large file costs bounded memory rather than
    its full size.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def build_ffmpeg_command(
    source: Path, destination: Path, config: AudioConfig
) -> list[str | Path]:
    """Assemble the normalisation command.

    Split out from execution so the exact flags are unit-testable without
    invoking FFmpeg — in particular that the reproducibility flags are always
    present and that denoising only appears when explicitly enabled.
    """
    command: list[str | Path] = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        source,
        "-vn",
        "-ac",
        str(config.channels),
        "-ar",
        str(config.sample_rate),
    ]

    if config.denoise:
        # FFT-based denoiser. Off by default; after we run some tests, we will decide
        # whether or not adding denoiser is a good choice.
        command += ["-af", "afftdn"]

    command += [
        "-c:a",
        "flac",
        # Reproducibility: strip source metadata and the encoder version stamp so
        # the output hash depends only on the audio (AD-2).
        "-map_metadata",
        "-1",
        "-bitexact",
        destination,
    ]
    return command


async def normalize_audio(
    source: Path, work_dir: Path, config: AudioConfig
) -> NormalizedAudio:
    """Extract and normalise the speech audio from a media file.

    Raises `NoAudioStreamError` when the container carries no audio, and
    `UnreadableMediaError` when it cannot be decoded.

    The audio-stream check happens before FFmpeg runs. Audioless media probes
    successfully, so without this check the failure would surface as an opaque
    FFmpeg exit and be reported as unreadable — a misleading error for a file
    that is perfectly readable and simply silent-by-construction.
    """
    log = logger.bind(source=str(source))

    probe = await probe_media(source)
    if not probe.has_audio:
        log.warning("media has no audio stream", format_name=probe.format_name)
        raise NoAudioStreamError(
            "The media contains no audio stream, so there is nothing to transcribe.",
            detail={"format": probe.format_name, "video_streams": probe.video_stream_count},
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    destination = work_dir / f"normalized.{config.audio_format}"

    try:
        result = await run_command(
            build_ffmpeg_command(source, destination, config),
            timeout_sec=FFMPEG_TIMEOUT_SEC,
        )
    except CommandTimeout as exc:
        raise UnreadableMediaError(
            "Audio extraction exceeded the time limit.",
            detail={"timeout_sec": exc.timeout_sec},
            cause=exc,
        ) from exc

    if not result.ok:
        log.error("ffmpeg failed", returncode=result.returncode)
        raise UnreadableMediaError(
            "Audio extraction failed.",
            detail={"ffmpeg_error": result.stderr_summary()},
        )

    if not destination.exists() or destination.stat().st_size == 0:
        # FFmpeg occasionally reports success while producing nothing usable.
        # Returning an empty file downstream would surface as a confusing
        # transcription failure instead of the extraction failure it is.
        raise UnreadableMediaError(
            "Audio extraction produced an empty file.",
            detail={"destination": destination.name},
        )

    # Re-probe the output rather than assuming the requested parameters were
    # applied, so recorded metadata describes the file that exists.
    normalized_probe = await probe_media(destination)
    stream = normalized_probe.audio_streams[0] if normalized_probe.audio_streams else None

    audio = NormalizedAudio(
        path=destination,
        sha256=compute_sha256(destination),
        duration=normalized_probe.duration or probe.duration,
        size_bytes=destination.stat().st_size,
        sample_rate=stream.sample_rate if stream and stream.sample_rate else config.sample_rate,
        channels=stream.channels if stream and stream.channels else config.channels,
    )

    log.info(
        "normalised audio",
        duration=audio.duration,
        size_bytes=audio.size_bytes,
        sha256=audio.sha256[:12],
        denoise=config.denoise,
    )
    return audio


__all__ = [
    "FFMPEG_TIMEOUT_SEC",
    "NormalizedAudio",
    "build_ffmpeg_command",
    "compute_sha256",
    "normalize_audio",
]