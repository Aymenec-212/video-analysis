"""Provider-agnostic domain models.

These types are the seam between the speech provider and everything downstream.
`stt/` produces them; `transcript/` and `analysis/` consume them and never see a
Deepgram-shaped payload. That is what makes AD-11's decision to own request and
response handling cheap: the blast radius of a provider change stops here.

The split between `Word` and `Segment` is deliberate. Deepgram returns words,
each independently tagged with a speaker. Segments are ours to construct (AD-3),
because the brief asks us to explain how transcript segments are associated with
speaker segments — and consuming pre-fused utterances would mean we never
performed that step.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Word(BaseModel):
    """A single recognised word with timing and speaker attribution.

    `speaker` is the provider's own integer label, kept raw at this layer. The
    mapping to `SPEAKER_01`-style identifiers happens in `transcript/speakers.py`
    by order of first appearance, so our public labels never depend on how a
    provider happens to number its clusters.
    """

    text: str
    start: float
    end: float
    confidence: float | None = None

    #: Provider speaker index, or None when diarization was not applied.
    speaker: int | None = None

    #: Per-word diarization confidence. Present on pre-recorded requests only,
    #: and the measurement AD-4 uses to decide whether smoothing is justified.
    speaker_confidence: float | None = None

    #: Populated only under code-switching (`language=multi`), where each word
    #: carries its own language tag.
    language: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class DetectedLanguage(BaseModel):
    """Language detection outcome for the audio.

    `confidence_is_meaningful` exists because the provider's score only accounts
    for its 35 supported languages: audio in an unsupported language still
    receives a score, and thresholding on it would silently mislabel the content.
    Carrying the caveat alongside the number means a consumer cannot use one
    without the other.
    """

    code: str | None = None
    confidence: float | None = None
    confidence_is_meaningful: bool = True

    #: All languages observed under code-switching, most frequent first. Empty in
    #: single-language mode.
    languages: list[str] = Field(default_factory=list)


class TranscriptionProvenance(BaseModel):
    """What actually produced a result.

    Recorded because several relevant choices are made server-side and are
    invisible otherwise: the model can be silently downgraded when a detected
    language is unavailable, and `diarize_model=latest` resolves to whatever is
    current. Without this, an unreproducible result has no explanation.
    """

    backend: str
    request_id: str | None = None

    #: Model that actually ran, which may differ from the one requested.
    resolved_model: str | None = None

    diarizer_arch: str | None = None
    diarizer_model_uuid: str | None = None

    #: True when served from the local cache rather than the network (AD-2).
    from_cache: bool = False


class TranscriptionResult(BaseModel):
    """Everything the speech stage hands downstream.

    An empty `words` list is a valid outcome, not an error: silent or music-only
    audio genuinely contains no speech. Classifying that as `NO_SPEECH_DETECTED`
    is a decision for the pipeline, so this layer reports the fact and leaves the
    interpretation alone.
    """

    words: list[Word] = Field(default_factory=list)
    language: DetectedLanguage = Field(default_factory=DetectedLanguage)
    provenance: TranscriptionProvenance

    #: Duration the provider measured, for cross-checking against ffprobe.
    audio_duration: float | None = None

    #: Provider-fused utterance count, requested purely as a cross-check against
    #: our own segmentation (AD-3). Never the source of truth.
    utterance_count: int | None = None

    @property
    def is_empty(self) -> bool:
        return not self.words

    @property
    def speaker_ids(self) -> set[int]:
        """Distinct provider speaker labels observed."""
        return {w.speaker for w in self.words if w.speaker is not None}


class Segment(BaseModel):
    """A contiguous stretch of speech by one speaker (SPEC §6.1).

    The unit the API returns and the unit LLM chunking respects. Built from words
    by `transcript/segmentation.py` rather than taken from the provider, which is
    what lets us answer the brief's question about how transcript segments are
    associated with speaker segments.
    """

    start: float
    end: float

    #: Public label — `SPEAKER_01`, or `SPEAKER_UNKNOWN` when diarization
    #: produced no attribution for this speech.
    speaker: str

    text: str

    #: Mean per-word diarization confidence across the segment. Summary only;
    #: AD-4's measurement uses the raw per-word values.
    speaker_confidence: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class Transcript(BaseModel):
    """The speaker-attributed transcript plus what we know about its speakers."""

    segments: list[Segment] = Field(default_factory=list)

    #: Distinct *identified* speakers. Unattributed speech is excluded, because
    #: counting it would mean asserting a number we did not measure.
    number_of_speakers: int = 0

    #: True when some speech carries no speaker attribution. Surfaced so the
    #: pipeline can mark the result degraded instead of quietly under-reporting.
    has_unattributed_speech: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.segments


__all__ = [
    "DetectedLanguage",
    "Segment",
    "Transcript",
    "TranscriptionProvenance",
    "TranscriptionResult",
    "Word",
]
