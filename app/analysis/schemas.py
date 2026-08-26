"""Structured output schemas for the analysis stage (AD-8).

These models are handed to OpenAI Structured Outputs, which constrains generation
to the supplied JSON Schema. That removes an entire class of failure rather than
retrying around it: the model cannot omit a required key, emit a wrong type, or
return prose where an array belongs.

Field descriptions are load-bearing, not documentation. They travel in the schema
and steer the model directly, which is more reliable than restating the same
instructions in prose and hoping they survive a long prompt.

**Strict mode makes every field required.** So nullable fields are declared
`X | None` rather than omitted — the model must emit `null` explicitly, which
forces a decision instead of allowing a silent omission we would have to
interpret.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class KeyPoint(BaseModel):
    """A single substantive point, anchored to where it was said.

    The timestamps are an anti-fabrication device before they are a feature.
    Requiring the model to cite the moment supporting each point makes an
    invented claim harder to produce than a grounded one, and gives us a cheap
    way to detect drift: a point whose evidence lies outside the excerpt it came
    from was not read off the transcript.
    """

    text: str = Field(
        description="The point itself, as a single self-contained sentence."
    )
    start: float | None = Field(
        description=(
            "Start timestamp in seconds of the transcript line supporting this "
            "point, copied from the excerpt. Null if no single line supports it."
        )
    )
    end: float | None = Field(
        description="End timestamp in seconds of the supporting line, or null."
    )


class ChunkAnalysis(BaseModel):
    """Analysis of one excerpt of a transcript (the map stage)."""

    summary: str = Field(
        description=(
            "What is discussed in THIS EXCERPT ONLY, in two or three sentences. "
            "Empty string if the excerpt contains nothing substantive."
        )
    )
    key_points: list[KeyPoint] = Field(
        description=(
            "Substantive points made in this excerpt. Empty array if there are "
            "none. Do not pad the list to make it look complete."
        )
    )
    topics: list[str] = Field(
        description=(
            "Subjects discussed in this excerpt, as short English noun phrases "
            "such as 'Artificial Intelligence' or 'Cloud Computing'. Empty array "
            "if none are identifiable."
        )
    )


class ReducedAnalysis(BaseModel):
    """Analysis merged across excerpts (the reduce stage)."""

    summary: str = Field(
        description=(
            "A single coherent summary of everything covered, written as one "
            "continuous passage rather than a list of per-excerpt summaries."
        )
    )
    key_points: list[KeyPoint] = Field(
        description=(
            "Merged key points with duplicates removed. Where two inputs make "
            "the same point, keep one and preserve the earliest timestamp."
        )
    )
    topics: list[str] = Field(
        description=(
            "Merged topics, deduplicated, as short English noun phrases. Prefer "
            "the more general label when two overlap."
        )
    )


class Analysis(BaseModel):
    """The finished analysis as the API reports it.

    `summary` is nullable and the lists may be empty: when the analysis stage
    fails, the response keeps the transcript and reports exactly nothing here
    rather than inventing plausible content (AD-9).

    `key_points` is a list of strings, matching the structure the brief
    illustrates. Evidence timestamps did their work upstream — grounding
    generation and driving deduplication — and are not part of the public
    contract.
    """

    summary: str | None = None
    key_points: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)

    #: How many excerpts the transcript was split into.
    chunk_count: int = 0

    #: Excerpts whose analysis failed after retries. Non-zero means the summary
    #: is built from partial coverage, which the response reports rather than
    #: hides.
    failed_chunks: int = 0

    @property
    def is_complete(self) -> bool:
        return self.summary is not None and self.failed_chunks == 0


__all__ = ["Analysis", "ChunkAnalysis", "KeyPoint", "ReducedAnalysis"]
