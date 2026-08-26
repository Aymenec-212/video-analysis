"""Transcript chunking for the analysis stage (AD-7).

The brief requires that the system not depend on sending an entire long
transcript in a single prompt. Chunking happens **here**, at the LLM layer, and
deliberately not at the audio layer (AD-5): splitting audio before diarization
would break speaker attribution, since speaker labels are not comparable across
independently diarized chunks.

Two rules govern the split.

**Cut only on segment boundaries.** A chunk that ends mid-utterance hands the
model half a sentence and invites it to guess at the rest. Segments are already
the natural unit — one speaker, one continuous stretch — so they are what chunks
are made of.

**Carry a small overlap.** A point made across a boundary would otherwise be
visible to neither neighbour in full. Overlap costs a duplicate key point, which
the reduce stage deduplicates; losing the point entirely is not recoverable.

Segments are rendered with speaker and timestamps so the model can attribute
statements and cite evidence for each key point.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, Field

from app.core.config import AnalysisConfig
from app.core.logging import get_logger
from app.core.models import Segment

logger = get_logger(__name__)

#: Fallback estimate when the exact vocabulary is unavailable. Deliberately below
#: the usual ~4 characters per token so the estimate runs high: over-counting
#: yields slightly smaller chunks, while under-counting risks exceeding the
#: budget the chunking exists to enforce.
CHARS_PER_TOKEN_ESTIMATE = 3.5


class TokenCounter:
    """Counts tokens, preferring exact counts and degrading safely.

    tiktoken downloads its vocabulary from the network on first use. That is fine
    on a normal machine and fatal in a sandbox, an air-gapped runner, or a test
    suite that promises to work offline — and the failure would surface as an
    HTTP error from deep inside chunking, far from anything that looks like a
    network call.

    So the exact encoding is attempted once and, if unavailable, a character
    heuristic takes over for the process lifetime. Chunk sizes shift slightly;
    nothing breaks. The budget exists to keep prompts bounded, not to bill by the
    token, and a ten-percent error against a budget with orders of magnitude of
    context headroom changes nothing that matters.
    """

    def __init__(self, encoding_name: str) -> None:
        self._encoding_name = encoding_name
        self._encoding: object | None = None
        self._resolved = False

    @property
    def is_exact(self) -> bool:
        """Whether counts come from the real vocabulary."""
        self._resolve()
        return self._encoding is not None

    def _resolve(self) -> None:
        if self._resolved:
            return
        self._resolved = True
        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding(self._encoding_name)
        except Exception as exc:  # noqa: BLE001 - any failure means fall back
            logger.warning(
                "exact token vocabulary unavailable; estimating from length",
                encoding=self._encoding_name,
                reason=str(exc)[:120],
            )
            self._encoding = None

    def count(self, text: str) -> int:
        """Token count for `text`."""
        if not text:
            return 0

        self._resolve()
        if self._encoding is not None:
            return len(self._encoding.encode(text))  # type: ignore[attr-defined]

        return max(1, math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE))


def render_segment(segment: Segment) -> str:
    """Render one segment for the model.

    Speaker and timestamps are included so the model can attribute statements and
    quote a timestamp as evidence for each key point.
    """
    return f"[{segment.speaker} | {segment.start:.1f}s-{segment.end:.1f}s] {segment.text}"


class Chunk(BaseModel):
    """A contiguous run of segments sized to fit one prompt."""

    index: int
    segments: list[Segment] = Field(default_factory=list)
    token_count: int = 0

    @property
    def start(self) -> float:
        return self.segments[0].start if self.segments else 0.0

    @property
    def end(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    def render(self) -> str:
        return "\n".join(render_segment(s) for s in self.segments)


def _make_chunk(index: int, segments: list[Segment], counter: TokenCounter) -> Chunk:
    return Chunk(
        index=index,
        segments=list(segments),
        token_count=sum(counter.count(render_segment(s)) for s in segments),
    )


def build_chunks(
    segments: Sequence[Segment],
    config: AnalysisConfig | None = None,
    counter: TokenCounter | None = None,
) -> list[Chunk]:
    """Split segments into token-budgeted, segment-aligned, overlapping chunks."""
    settings = config or AnalysisConfig()
    tokens = counter or TokenCounter(settings.encoding_name)

    if not segments:
        return []

    budget = settings.chunk_token_budget
    overlap_size = max(0, settings.chunk_overlap_segments)

    chunks: list[Chunk] = []
    current: list[Segment] = []
    current_tokens = 0

    for segment in segments:
        cost = tokens.count(render_segment(segment))

        if current and current_tokens + cost > budget:
            chunks.append(_make_chunk(len(chunks), current, tokens))

            # Overlap is dropped when it would not leave room for new content,
            # which also prevents a segment larger than the budget from being
            # carried forward indefinitely.
            carried = current[-overlap_size:] if overlap_size else []
            carried_tokens = sum(tokens.count(render_segment(s)) for s in carried)
            if carried_tokens >= budget:
                carried, carried_tokens = [], 0

            current = list(carried)
            current_tokens = carried_tokens

        current.append(segment)
        current_tokens += cost

        if cost > budget:
            # Segments are never split, so an oversized one becomes its own
            # chunk. With the segmentation ceilings in place this is rare;
            # logging it means we would learn if that assumption stopped holding.
            logger.warning(
                "segment exceeds the chunk budget on its own",
                tokens=cost,
                budget=budget,
                start=segment.start,
            )

    if current:
        chunks.append(_make_chunk(len(chunks), current, tokens))

    logger.debug(
        "built chunks",
        chunks=len(chunks),
        segments=len(segments),
        exact_tokens=tokens.is_exact,
    )
    return chunks


__all__ = [
    "CHARS_PER_TOKEN_ESTIMATE",
    "Chunk",
    "TokenCounter",
    "build_chunks",
    "render_segment",
]
