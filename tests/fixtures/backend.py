"""A `TranscriptionBackend` test double.

Satisfies the Protocol structurally, so the pipeline and API are exercised with
no network, no keys, and no provider SDK.
"""

from __future__ import annotations

from app.audio.ffmpeg import NormalizedAudio
from app.core.models import (
    DetectedLanguage,
    TranscriptionProvenance,
    TranscriptionResult,
    Word,
)


def french_conversation() -> list[Word]:
    """Two speakers, matching the exchange the brief uses as its example."""
    return [
        Word(text="Bonjour", start=0.5, end=1.1, speaker=0, speaker_confidence=0.96),
        Word(text="et", start=1.1, end=1.3, speaker=0, speaker_confidence=0.96),
        Word(text="bienvenue.", start=1.3, end=2.0, speaker=0, speaker_confidence=0.95),
        Word(text="Merci", start=2.8, end=3.2, speaker=1, speaker_confidence=0.94),
        Word(text="de", start=3.2, end=3.4, speaker=1, speaker_confidence=0.93),
        Word(text="m'avoir", start=3.4, end=3.9, speaker=1, speaker_confidence=0.94),
        Word(text="invité.", start=3.9, end=4.5, speaker=1, speaker_confidence=0.95),
    ]


def transcription_result(
    words: list[Word] | None = None,
    *,
    language: str | None = "fr",
    from_cache: bool = False,
) -> TranscriptionResult:
    return TranscriptionResult(
        words=french_conversation() if words is None else words,
        language=DetectedLanguage(
            code=language, confidence=0.97, confidence_is_meaningful=True
        ),
        provenance=TranscriptionProvenance(
            backend="fake",
            request_id="test-request",
            resolved_model="nova-3-general",
            diarizer_arch="v2",
            from_cache=from_cache,
        ),
        audio_duration=12.0,
        utterance_count=2,
    )


class FakeBackend:
    """Returns a scripted transcription, or raises a scripted error."""

    def __init__(
        self,
        result: TranscriptionResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result if result is not None else transcription_result()
        self._error = error
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake"

    async def transcribe(self, audio: NormalizedAudio) -> TranscriptionResult:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._result
