"""The speech backend interface.

The rest of the pipeline depends only on this interface.
We could implement another backend that satisfies the same contract.

We deliberately created an explicit boundary between our pipeline and the ASR provider,
so the rest of the system depends on a stable contract instead of Deepgram-specific code;
this also makes replacing Deepgram or testing without it straightforward.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.audio.ffmpeg import NormalizedAudio
from app.core.models import TranscriptionResult


@runtime_checkable
class TranscriptionBackend(Protocol):
    """Turns normalised audio into a provider-agnostic word stream.

    Implementations must:

    * return words carrying start, end, and speaker attribution where the
      provider supplies it;
    * return an **empty** word list for audio containing no speech, rather than
      raising — silence is a finding, and classifying it belongs to the pipeline;
    * raise `STTFailedError` when transcription genuinely fails, so no stage
      downstream ever receives a fabricated or partial word stream.
    """

    @property
    def name(self) -> str:
        """Short backend identifier, recorded in result provenance."""
        ...

    async def transcribe(self, audio: NormalizedAudio) -> TranscriptionResult:
        """Transcribe and diarize normalised audio."""
        ...


__all__ = ["TranscriptionBackend"]
