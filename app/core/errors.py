"""Error taxonomy for the analysis pipeline (SPEC §6.3).

*Fatal* errors (4xx/5xx) mean no useful output exists. The request could not
produce a transcript, so there is nothing to return but the error.

*Non-fatal* errors (200) mean a later stage failed but earlier stages produced
output. `NO_SPEECH_DETECTED` and `ANALYSIS_FAILED` return 200 carrying
whatever genuinely succeeded, with the failed fields explicitly null or empty
and an entry in `errors[]`. A failed stage never invents content, and never
destroys the work that did succeed.

Because `is_fatal` is derived from the status code rather than set by hand,
adding a code to this taxonomy forces a decision about which kind it is.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, Field


class Stage(StrEnum):
    """Pipeline stages, in execution order.

    Also the key set of the `stages` object in the response (SPEC 6.2), which
    reports per-stage status so a caller can see exactly how far the pipeline got.
    """

    INGESTION = "ingestion"
    AUDIO = "audio"
    TRANSCRIPTION = "transcription"
    DIARIZATION = "diarization"
    ANALYSIS = "analysis"


class StageStatus(StrEnum):
    """Outcome of a single stage."""

    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"
    DEGRADED = "degraded"


class ErrorCode(StrEnum):
    """failure identifiers (SPEC 6.3).
    """

    INVALID_URL = "INVALID_URL"
    UNSUPPORTED_URL = "UNSUPPORTED_URL"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    MEDIA_TOO_LONG = "MEDIA_TOO_LONG"
    MEDIA_TOO_LARGE = "MEDIA_TOO_LARGE"
    NO_AUDIO_STREAM = "NO_AUDIO_STREAM"
    UNREADABLE_MEDIA = "UNREADABLE_MEDIA"
    STT_FAILED = "STT_FAILED"
    NO_SPEECH_DETECTED = "NO_SPEECH_DETECTED"
    ANALYSIS_FAILED = "ANALYSIS_FAILED"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"


#: HTTP status per error code. The two 200s are the design statement of AD-9:
#: the pipeline succeeded far enough to return real work, so the response is a
#: success carrying an explicit record of what could not be produced.
HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_URL: 400,
    ErrorCode.UNSUPPORTED_URL: 400,
    ErrorCode.SOURCE_UNAVAILABLE: 422,
    ErrorCode.MEDIA_TOO_LONG: 422,
    ErrorCode.MEDIA_TOO_LARGE: 413,
    ErrorCode.NO_AUDIO_STREAM: 422,
    ErrorCode.UNREADABLE_MEDIA: 422,
    ErrorCode.STT_FAILED: 502,
    ErrorCode.NO_SPEECH_DETECTED: 200,
    ErrorCode.ANALYSIS_FAILED: 200,
    ErrorCode.CONFIGURATION_ERROR: 500,
}


class ErrorEntry(BaseModel):
    """A failure as it appears in the response `errors[]` array.

    This is the serialisable form of a `PipelineError`, the shape the caller
    sees, decoupled from the exception used internally.
    """

    stage: Stage
    code: ErrorCode
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class PipelineError(Exception):
    """Base class for every failure this pipeline raises deliberately.

    Subclasses declare `code` and `stage` as class attributes; `__init_subclass__`
    enforces that at import time, so a subclass missing either fails immediately
    rather than at the moment it is first raised in production.

    Unexpected exceptions from third-party libraries are *not* instances of this
    class. Each stage is responsible for catching those at its boundary and
    translating them into the appropriate `PipelineError`, so that anything
    reaching the API layer is either a known taxonomy member or a genuine bug.
    """

    code: ClassVar[ErrorCode]
    stage: ClassVar[Stage]
    default_message: ClassVar[str] = "Pipeline error"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Intermediate/abstract subclasses may legitimately defer these, but a
        # class that defines neither and is never subclassed further is a bug we
        # want surfaced at import.
        if not hasattr(cls, "code") or not hasattr(cls, "stage"):
            raise TypeError(
                f"{cls.__name__} must define both `code` and `stage` class attributes"
            )

    def __init__(
        self,
        message: str | None = None,
        *,
        detail: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.detail = detail or {}
        self.cause = cause
        super().__init__(self.message)

    @property
    def http_status(self) -> int:
        return HTTP_STATUS[self.code]

    @property
    def is_fatal(self) -> bool:
        """True when no useful output can be returned.

        Derived from the status code rather than declared separately, so the two
        can never drift apart.
        """
        return self.http_status >= 400

    def to_entry(self) -> ErrorEntry:
        return ErrorEntry(
            stage=self.stage,
            code=self.code,
            message=self.message,
            detail=self.detail,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code.value}, message={self.message!r})"


# ---------------------------------------------------------------------------
# Ingestion (SPEC 5.1)
# ---------------------------------------------------------------------------


class InvalidURLError(PipelineError):
    """Malformed URL, disallowed scheme, or an address we refuse to fetch.

    Also covers private/loopback addresses blocked for SSRF safety.
    """

    code = ErrorCode.INVALID_URL
    stage = Stage.INGESTION
    default_message = "The supplied URL is malformed or uses a disallowed scheme."


class UnsupportedURLError(PipelineError):
    """Well-formed URL, but no extractor can handle this site."""

    code = ErrorCode.UNSUPPORTED_URL
    stage = Stage.INGESTION
    default_message = "No extractor is available for this URL."


class SourceUnavailableError(PipelineError):
    """The video exists as a URL but cannot be retrieved.

    Removed, private, geo-blocked, age-gated, or behind a bot check.
    """

    code = ErrorCode.SOURCE_UNAVAILABLE
    stage = Stage.INGESTION
    default_message = "The video could not be retrieved from its source."


class MediaTooLongError(PipelineError):
    """Duration exceeds the configured cap."""

    code = ErrorCode.MEDIA_TOO_LONG
    stage = Stage.INGESTION
    default_message = "The media exceeds the configured duration limit."


class MediaTooLargeError(PipelineError):
    """Payload exceeds the configured size cap."""

    code = ErrorCode.MEDIA_TOO_LARGE
    stage = Stage.INGESTION
    default_message = "The media exceeds the configured size limit."


# ---------------------------------------------------------------------------
# Audio (SPEC 5.2)
# ---------------------------------------------------------------------------


class NoAudioStreamError(PipelineError):
    """Container decodes, but carries no audio track."""

    code = ErrorCode.NO_AUDIO_STREAM
    stage = Stage.AUDIO
    default_message = "The media contains no audio stream."


class UnreadableMediaError(PipelineError):
    """ffprobe or FFmpeg cannot decode the file.

    Corrupt, truncated, or not media at all, an extension proves nothing, which
    is why uploads are validated by probing rather than by filename.
    """

    code = ErrorCode.UNREADABLE_MEDIA
    stage = Stage.AUDIO
    default_message = "The media file could not be decoded."


# ---------------------------------------------------------------------------
# Transcription (SPEC 5.3)
# ---------------------------------------------------------------------------


class STTFailedError(PipelineError):
    """The speech backend failed after retries were exhausted.

    Fatal: without a transcript there is nothing downstream worth returning.
    """

    code = ErrorCode.STT_FAILED
    stage = Stage.TRANSCRIPTION
    default_message = "Speech-to-text failed after retries."


class NoSpeechDetectedError(PipelineError):
    """Transcription succeeded and found no speech.

    Non-fatal (200). This is a correct answer about silent or music-only audio,
    not a malfunction.
    """

    code = ErrorCode.NO_SPEECH_DETECTED
    stage = Stage.TRANSCRIPTION
    default_message = "No speech was detected in the audio."


# ---------------------------------------------------------------------------
# Analysis (SPEC 5.5, AD-9)
# ---------------------------------------------------------------------------


class AnalysisFailedError(PipelineError):
    """The LLM stage failed; the transcript passed.

    Non-fatal (200). The response keeps the full transcript and sets `summary`
    to null with empty `key_points` and `topics`. This is the single most
    important behaviour in the taxonomy.
    """

    code = ErrorCode.ANALYSIS_FAILED
    stage = Stage.ANALYSIS
    default_message = "Transcript analysis failed; the transcript is unaffected."


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigurationError(PipelineError):
    """Invalid or missing configuration detected before any work is attempted.

    Raised at startup.
    """

    code = ErrorCode.CONFIGURATION_ERROR
    stage = Stage.INGESTION
    default_message = "The application is misconfigured."


__all__ = [
    "HTTP_STATUS",
    "AnalysisFailedError",
    "ConfigurationError",
    "ErrorCode",
    "ErrorEntry",
    "InvalidURLError",
    "MediaTooLargeError",
    "MediaTooLongError",
    "NoAudioStreamError",
    "NoSpeechDetectedError",
    "PipelineError",
    "STTFailedError",
    "SourceUnavailableError",
    "Stage",
    "StageStatus",
    "UnreadableMediaError",
    "UnsupportedURLError",
]