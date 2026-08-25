"""Speaker identification and counting (SPEC §5.4).

Two requirements from the brief meet here: speakers are reported as
`SPEAKER_01`, `SPEAKER_02`, …, and the number of speakers is **never assumed in
advance** — it is derived from what diarization actually found.

Labels are assigned **by order of first appearance**, not by the provider's own
numbering. Deepgram's integer labels are cluster identifiers with no guaranteed
ordering, so a re-run that happened to number the same voices differently would
otherwise produce a different-looking transcript for identical audio. Ordering by
first appearance makes the output depend only on the audio, and has the pleasant
side effect that `SPEAKER_01` is whoever speaks first — which is what a reader
expects.

Unattributed speech is labelled rather than guessed. If diarization returns no
speaker for some words, assigning them to `SPEAKER_01` would assert a single
speaker we did not measure; the brief's prohibition on invented output applies to
quiet assumptions as much as to fabricated text.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.models import Word

#: Label for speech diarization could not attribute. Deliberately outside the
#: `SPEAKER_NN` sequence so it can never be mistaken for an identified speaker.
UNKNOWN_SPEAKER = "SPEAKER_UNKNOWN"


def format_speaker_label(position: int) -> str:
    """Render a zero-based position as `SPEAKER_01`, `SPEAKER_02`, …

    Two digits below 100 keeps labels sortable as strings; beyond that the label
    widens rather than wrapping, since a truncated label would collide.
    """
    if position < 0:
        raise ValueError("speaker position must not be negative")
    return f"SPEAKER_{position + 1:02d}"


def build_speaker_map(words: Sequence[Word]) -> dict[int, str]:
    """Map provider speaker ids to public labels, ordered by first appearance.

    Deterministic for a given word stream: the same audio always yields the same
    labels, regardless of how the provider numbered its clusters.
    """
    mapping: dict[int, str] = {}
    for item in words:
        if item.speaker is None or item.speaker in mapping:
            continue
        mapping[item.speaker] = format_speaker_label(len(mapping))
    return mapping


def label_for(speaker_id: int | None, mapping: dict[int, str]) -> str:
    """Resolve one provider id to its public label."""
    if speaker_id is None:
        return UNKNOWN_SPEAKER
    return mapping.get(speaker_id, UNKNOWN_SPEAKER)


def count_speakers(words: Sequence[Word]) -> int:
    """Number of distinct identified speakers.

    Counts only attributed speech. Unattributed words do not contribute, because
    we cannot tell whether they belong to a speaker already counted or to another
    one entirely.
    """
    return len(build_speaker_map(words))


def has_unattributed_speech(words: Sequence[Word]) -> bool:
    """Whether any word lacks speaker attribution."""
    return any(item.speaker is None for item in words)


__all__ = [
    "UNKNOWN_SPEAKER",
    "build_speaker_map",
    "count_speakers",
    "format_speaker_label",
    "has_unattributed_speech",
    "label_for",
]
