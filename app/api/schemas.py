"""API request and response contracts (SPEC §6.2).

The response shape follows the structure the brief illustrates, with additions
where a bare value would be misleading rather than merely terse.

**Nothing is omitted when null.** `summary: null` must appear in the body when
analysis failed, because that is the response saying *we could not produce this*.
Dropping the key would leave a caller unable to distinguish "no summary" from
"this API does not return summaries", and would quietly undo the guarantee AD-9
exists to make visible.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from app.core.errors import ErrorEntry


class AnalysisStatus(StrEnum):
    """Overall outcome (SPEC §6.2).

    All three are `200`. `no_speech` and `partial_success` are successful
    responses that report what could not be produced, rather than errors that
    discard the work that succeeded.
    """

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    NO_SPEECH = "no_speech"


class AnalyzeUrlRequest(BaseModel):
    """JSON body: `{"url": "https://..."}`."""

    url: str


class LanguageInfo(BaseModel):
    """Detected language.

    `confidence_is_meaningful` travels with the score because the provider's
    detector only accounts for its 35 supported languages: audio outside that set
    still receives a confidence value, and a caller thresholding on the number
    alone would silently mislabel it.
    """

    code: str | None = None
    confidence: float | None = None
    confidence_is_meaningful: bool = False


class TranscriptSegment(BaseModel):
    """One speaker-attributed, timestamped segment.

    `speaker_confidence` is included beyond the brief's illustration: it is the
    diarizer's own certainty, it is what AD-4's smoothing decision is measured
    from, and a reviewer can use it to judge attribution quality directly.
    """

    start: float
    end: float
    speaker: str
    text: str
    speaker_confidence: float | None = None


class Provenance(BaseModel):
    """What actually produced this result.

    Recorded because several choices are made server-side and are otherwise
    invisible: the speech model can be silently downgraded when a detected
    language is unavailable, and `diarize_model=latest` resolves to whatever is
    current. Without this, an unexpected result has no explanation.
    """

    resolved_model: str | None = None
    diarizer_arch: str | None = None
    map_model: str | None = None
    reduce_model: str | None = None
    transcription_cached: bool = False

    #: Excerpts the transcript was split into for analysis, and how many failed.
    #: Direct evidence for the brief's long-content requirement: a value above 1
    #: shows the transcript was never sent to the model in a single prompt.
    chunk_count: int = 0
    failed_chunks: int = 0


class AnalyzeVideoResponse(BaseModel):
    """The full analysis contract."""

    status: AnalysisStatus

    title: str
    duration: float | None = None
    source: str
    description: str | None = None

    language: LanguageInfo = Field(default_factory=LanguageInfo)
    number_of_speakers: int = 0
    transcript: list[TranscriptSegment] = Field(default_factory=list)

    #: Null when analysis did not run or failed. Never a placeholder string.
    summary: str | None = None
    key_points: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)

    #: Per-stage outcome, so a caller can see exactly how far the pipeline got.
    stages: dict[str, str] = Field(default_factory=dict)

    #: Populated for anything that failed without ending the request.
    errors: list[ErrorEntry] = Field(default_factory=list)

    #: True when the result is usable but incomplete — partial analysis coverage,
    #: unattributed speech, or segment defects.
    degraded: bool = False

    #: Why. Non-empty whenever `degraded` is true, and empty whenever it is not.
    #:
    #: A boolean on its own tells a caller something is wrong and gives them no
    #: way to find out what. Segment overlap during genuine crosstalk and an
    #: analysis built from partial coverage both set the same flag while meaning
    #: entirely different things, and only one of them is worth acting on.
    degraded_reasons: list[str] = Field(default_factory=list)

    provenance: Provenance = Field(default_factory=Provenance)


class ErrorResponse(BaseModel):
    """Body returned for fatal failures.

    Uses the same `errors[]` shape as a successful-but-degraded response, so a
    caller parses one structure regardless of outcome.
    """

    status: str = "error"
    errors: list[ErrorEntry] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Liveness plus configuration visibility.

    The two credential flags report whether a key is *present*, never any part of
    its value. They exist because "transcription failed" and "no key configured"
    look identical from outside, and this turns a support question into a glance.
    """

    status: str = "ok"
    version: str
    deepgram_configured: bool
    openai_configured: bool


__all__ = [
    "AnalysisStatus",
    "AnalyzeUrlRequest",
    "AnalyzeVideoResponse",
    "ErrorResponse",
    "HealthResponse",
    "LanguageInfo",
    "Provenance",
    "TranscriptSegment",
]
