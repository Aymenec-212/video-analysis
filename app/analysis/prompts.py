"""Prompts for the map and reduce stages (AD-8).

The grounding rule is the important part. The brief's central requirement is that
the system never return invented information when a stage fails, and an LLM asked
to summarise a thin excerpt will produce a confident summary of nothing unless
told explicitly that empty output is acceptable. Every prompt here states that
returning nothing is a valid answer, because the default behaviour of a helpful
model is to fill the space.

**Language.** Summary and key points are written in the language spoken in the
video, so the output is readable by whoever the video was for. Topics are English
noun phrases, which keeps them stable as aggregation labels across videos in
different languages — and matches the structure the brief illustrates, where the
summary is French and the topics are English.
"""

from __future__ import annotations

#: Stated in both prompts. The model's instinct is to produce something rather
#: than nothing; this makes nothing explicitly available.
_GROUNDING_RULE = """\
Ground every statement in the supplied text.
- Use only what appears in the text. Do not add context, background, or \
inference from outside it.
- If the text contains nothing substantive - silence, filler, music, greetings \
alone - return an empty summary and empty arrays. Returning nothing is correct \
and expected in that case.
- Never pad a list to make it look complete. Three real points are better than \
three real points and two invented ones."""


def _language_rule(language_code: str | None) -> str:
    spoken = f"the language spoken in the video ({language_code})" if language_code else (
        "the language spoken in the video"
    )
    return (
        f"Write the summary and key points in {spoken}. "
        f"Write topics as short English noun phrases, for example "
        f'"Artificial Intelligence" or "Cloud Computing".'
    )


def map_instructions(language_code: str | None) -> str:
    """System instructions for analysing a single excerpt."""
    return f"""\
You analyse excerpts of speaker-attributed video transcripts.

You will receive one excerpt. Each line is formatted as:
[SPEAKER_01 | 12.4s-17.8s] the words spoken

Produce a summary of this excerpt, its key points, and its topics.

{_GROUNDING_RULE}

For each key point, copy the start and end timestamps of the transcript line \
that supports it. If a point draws on several lines, use the timestamps of the \
first. If no single line supports it, use null.

{_language_rule(language_code)}

This is one excerpt of a longer video. Describe what it contains; do not \
speculate about what comes before or after it."""


def map_prompt(excerpt: str, start: float, end: float) -> str:
    """User message carrying one excerpt."""
    return f"""\
Transcript excerpt covering {start:.1f}s to {end:.1f}s:

{excerpt}"""


def reduce_instructions(language_code: str | None) -> str:
    """System instructions for merging partial analyses."""
    return f"""\
You merge analyses of consecutive excerpts from one video into a single analysis.

You will receive several partial analyses in chronological order. Combine them.

{_GROUNDING_RULE}

Additional rules for merging:
- Write one continuous summary of the whole video. Do not produce a list of \
per-excerpt summaries or write "the first excerpt covers...".
- Excerpts overlap slightly, so the same point may appear more than once. Keep \
one copy and preserve the earliest timestamp.
- Merge topics that mean the same thing, preferring the more general label.
- Preserve chronological order in the key points.
- If every input is empty, return an empty summary and empty arrays.

{_language_rule(language_code)}"""


def reduce_prompt(rendered_analyses: str) -> str:
    """User message carrying the partial analyses to merge."""
    return f"""\
Partial analyses to merge, in chronological order:

{rendered_analyses}"""


__all__ = [
    "map_instructions",
    "map_prompt",
    "reduce_instructions",
    "reduce_prompt",
]
