"""Word stream to speaker-attributed segments (AD-3).

This is the module the brief specifically asks us to explain: *how transcription
segments are associated with speaker segments*. Deepgram will happily return
pre-fused `utterances`, and consuming those would mean the association step was
purchased rather than performed. We request `utterances=true` only as a
cross-check and build segments here, from the word stream, where the rules are
ours and are testable.

**The algorithm.** Walk words in order and close the current segment when any of
these holds:

1. *The speaker changes.* The defining boundary — a segment belongs to one
   speaker by definition.
2. *The gap since the previous word exceeds `pause_threshold_sec`.* A long
   silence inside one speaker's turn is a natural break, and keeping it inside
   one segment would produce timings that span silence the speaker did not fill.
3. *The segment has run past `max_segment_sec` and the previous word ends a
   sentence.* A soft ceiling that breaks where a reader expects.
4. *The segment has run past `hard_max_segment_sec`.* Applied regardless of
   punctuation, so unpunctuated speech cannot grow one segment without bound.

Rules 3 and 4 exist because segments feed LLM chunking (AD-7), which cuts only on
segment boundaries. A single enormous segment would defeat that.

**Smoothing is not applied here.** Short spurious speaker flips are a known
diarization artefact, and the obvious fix — collapse brief turns into their
neighbours — is exactly the kind of heuristic AD-4 refuses to ship before the
problem is measured. `speaker_confidence` on every word makes it measurable, so
the decision waits for data. The hook exists and defaults to off.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.config import SegmentationConfig
from app.core.logging import get_logger
from app.core.models import Segment, Transcript, Word

from .speakers import (
    build_speaker_map,
    count_speakers,
    has_unattributed_speech,
    label_for,
)

logger = get_logger(__name__)

#: Characters that end a sentence. Trailing quotes and brackets are stripped
#: before the check, so `bienvenue."` still counts as a boundary.
_SENTENCE_ENDINGS = (".", "!", "?", "…")
_TRAILING_MARKS = "\"'»”’)]}"


def ends_sentence(text: str) -> bool:
    """Whether a word's text closes a sentence."""
    stripped = text.rstrip(_TRAILING_MARKS).rstrip()
    return stripped.endswith(_SENTENCE_ENDINGS)


def _mean_confidence(words: Sequence[Word]) -> float | None:
    """Mean diarization confidence, or None when the provider supplied none."""
    scores = [w.speaker_confidence for w in words if w.speaker_confidence is not None]
    return sum(scores) / len(scores) if scores else None


def _build_segment(words: list[Word], label: str) -> Segment:
    """Assemble one segment from the words accumulated for it."""
    return Segment(
        start=words[0].start,
        end=words[-1].end,
        speaker=label,
        # Words carry their punctuated form, so joining on a single space
        # reproduces normal prose.
        text=" ".join(w.text for w in words).strip(),
        speaker_confidence=_mean_confidence(words),
    )


def _should_close(
    current: list[Word], candidate: Word, config: SegmentationConfig
) -> bool:
    """Whether `candidate` begins a new segment rather than extending this one."""
    previous = current[-1]

    if candidate.speaker != previous.speaker:
        return True

    if candidate.start - previous.end > config.pause_threshold_sec:
        return True

    duration = previous.end - current[0].start
    if duration >= config.hard_max_segment_sec:
        return True

    return duration >= config.max_segment_sec and ends_sentence(previous.text)


def build_segments(
    words: Sequence[Word], config: SegmentationConfig | None = None
) -> list[Segment]:
    """Group a word stream into speaker-attributed segments."""
    settings = config or SegmentationConfig()
    if not words:
        return []

    # Sorting defends against out-of-order words: the boundary rules compare
    # adjacent timings, so a single misordered word would otherwise produce a
    # spurious cut and a negative gap.
    ordered = sorted(words, key=lambda w: (w.start, w.end))
    mapping = build_speaker_map(ordered)

    segments: list[Segment] = []
    current: list[Word] = [ordered[0]]

    for word in ordered[1:]:
        if _should_close(current, word, settings):
            segments.append(_build_segment(current, label_for(current[0].speaker, mapping)))
            current = [word]
        else:
            current.append(word)

    segments.append(_build_segment(current, label_for(current[0].speaker, mapping)))

    if settings.smoothing_enabled:
        segments = smooth_speaker_flips(segments, settings)

    return segments


def smooth_speaker_flips(
    segments: list[Segment], config: SegmentationConfig
) -> list[Segment]:
    """Absorb brief, low-confidence speaker turns into their surroundings.

    **Disabled by default (AD-4).** Enabled only if the measured
    `speaker_confidence` distribution shows that short speaker flips genuinely
    cluster at low confidence. Gating on *both* brevity and low confidence is
    what separates this from a magic number: a short turn the diarizer was sure
    about is a real interjection, and collapsing it would destroy correct output.

    A flip is absorbed only when the segments on either side share a speaker —
    otherwise there is no unambiguous owner to merge into.
    """
    if len(segments) < 3:
        return segments

    result = [segments[0]]

    index = 1
    while index < len(segments) - 1:
        candidate = segments[index]
        previous, following = result[-1], segments[index + 1]

        is_brief = candidate.duration <= config.smoothing_max_turn_sec
        is_uncertain = (
            candidate.speaker_confidence is not None
            and candidate.speaker_confidence < config.smoothing_min_confidence
        )
        is_enclosed = previous.speaker == following.speaker != candidate.speaker

        if is_brief and is_uncertain and is_enclosed:
            merged = Segment(
                start=previous.start,
                end=candidate.end,
                speaker=previous.speaker,
                text=f"{previous.text} {candidate.text}".strip(),
                speaker_confidence=previous.speaker_confidence,
            )
            result[-1] = merged
        else:
            result.append(candidate)
        index += 1

    result.append(segments[-1])
    return result


def build_transcript(
    words: Sequence[Word], config: SegmentationConfig | None = None
) -> Transcript:
    """Build the full transcript, including speaker count and attribution status."""
    segments = build_segments(words, config)
    transcript = Transcript(
        segments=segments,
        number_of_speakers=count_speakers(words),
        has_unattributed_speech=has_unattributed_speech(words),
    )

    logger.debug(
        "built transcript",
        words=len(words),
        segments=len(segments),
        speakers=transcript.number_of_speakers,
        unattributed=transcript.has_unattributed_speech,
    )
    return transcript


__all__ = [
    "build_segments",
    "build_transcript",
    "ends_sentence",
    "smooth_speaker_flips",
]
