"""Transcript validation before analysis (SPEC §5.5).

These checks guard a boundary between our own stages, so a failure here means
*our* segmentation is wrong, not that the input was bad. That shapes the design:
validation **reports** rather than raises, and the pipeline logs what it finds.
Silently discarding malformed segments would hide a bug in exactly the component
the brief asks us to explain.

The one check that is not about our correctness is emptiness. A transcript with
no segments is the legitimate outcome for silent audio, and is what the pipeline
turns into `NO_SPEECH_DETECTED` — a successful response with nothing invented to
fill it.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel

from app.core.models import Segment

from .speakers import UNKNOWN_SPEAKER


class IssueKind(StrEnum):
    """Categories of transcript defect."""

    EMPTY_TEXT = "empty_text"
    NON_POSITIVE_DURATION = "non_positive_duration"
    OUT_OF_ORDER = "out_of_order"
    OVERLAPPING = "overlapping"
    INVALID_SPEAKER_LABEL = "invalid_speaker_label"
    NEGATIVE_TIMESTAMP = "negative_timestamp"


class ValidationIssue(BaseModel):
    """One defect, located by segment index."""

    kind: IssueKind
    index: int
    detail: str


def _is_valid_label(label: str) -> bool:
    if label == UNKNOWN_SPEAKER:
        return True
    if not label.startswith("SPEAKER_"):
        return False
    suffix = label.removeprefix("SPEAKER_")
    return suffix.isdigit() and len(suffix) >= 2 and int(suffix) >= 1


def validate_segments(segments: Sequence[Segment]) -> list[ValidationIssue]:
    """Check segments against the §5.5 invariants.

    Returns every issue found rather than stopping at the first, so a single run
    surfaces the full picture instead of revealing defects one at a time.
    """
    issues: list[ValidationIssue] = []

    for index, segment in enumerate(segments):
        if not segment.text.strip():
            issues.append(
                ValidationIssue(
                    kind=IssueKind.EMPTY_TEXT,
                    index=index,
                    detail="segment carries no text",
                )
            )

        if segment.start < 0 or segment.end < 0:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.NEGATIVE_TIMESTAMP,
                    index=index,
                    detail=f"start={segment.start}, end={segment.end}",
                )
            )

        if segment.end <= segment.start:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.NON_POSITIVE_DURATION,
                    index=index,
                    detail=f"start={segment.start} is not before end={segment.end}",
                )
            )

        if not _is_valid_label(segment.speaker):
            issues.append(
                ValidationIssue(
                    kind=IssueKind.INVALID_SPEAKER_LABEL,
                    index=index,
                    detail=f"unexpected speaker label {segment.speaker!r}",
                )
            )

        if index > 0:
            previous = segments[index - 1]
            if segment.start < previous.start:
                issues.append(
                    ValidationIssue(
                        kind=IssueKind.OUT_OF_ORDER,
                        index=index,
                        detail=f"starts at {segment.start} before previous {previous.start}",
                    )
                )
            elif segment.start < previous.end:
                # Distinct from out-of-order: ordering holds, but the segments
                # claim overlapping time. Our own construction cannot produce
                # this, so it would indicate a boundary bug.
                issues.append(
                    ValidationIssue(
                        kind=IssueKind.OVERLAPPING,
                        index=index,
                        detail=f"starts at {segment.start} before previous ends at {previous.end}",
                    )
                )

    return issues


def is_empty_transcript(segments: Sequence[Segment]) -> bool:
    """Whether the transcript contains no usable speech.

    True for both no segments at all and segments carrying only whitespace. This
    is the `NO_SPEECH_DETECTED` condition — a correct answer about silent audio,
    not a failure.
    """
    return not any(segment.text.strip() for segment in segments)


__all__ = [
    "IssueKind",
    "ValidationIssue",
    "is_empty_transcript",
    "validate_segments",
]
