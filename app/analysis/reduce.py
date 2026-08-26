"""The reduce stage: fold chunk analyses into one (AD-7).

**Folding is hierarchical, and that is the whole point.** A single reduce call
over every chunk analysis would put the entire set into one prompt — recreating
precisely the problem the brief asks us to solve, just one level up. Thirty
chunk summaries is a long prompt; three hundred is an impossible one. Folding in
batches until a single result remains means the strategy holds for a transcript
of any length, because the prompt size is bounded by `reduce_batch_size` rather
than by the video.

A single analysis is returned directly rather than sent through the model. There
is nothing to merge, and re-summarising a summary only loses detail and spends a
call.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.concurrency import gather_bounded, retry_async
from app.core.config import AnalysisConfig
from app.core.logging import get_logger

from .llm import StructuredLLM
from .prompts import reduce_instructions, reduce_prompt
from .schemas import Analysis, ChunkAnalysis, KeyPoint, ReducedAnalysis

logger = get_logger(__name__)


def render_analysis(analysis: ChunkAnalysis | ReducedAnalysis, index: int) -> str:
    """Render one partial analysis as input to the next fold."""
    lines = [f"--- Analysis {index + 1} ---", f"Summary: {analysis.summary or '(none)'}"]

    if analysis.key_points:
        lines.append("Key points:")
        for point in analysis.key_points:
            stamp = (
                f" [{point.start:.1f}s-{point.end:.1f}s]"
                if point.start is not None and point.end is not None
                else ""
            )
            lines.append(f"  - {point.text}{stamp}")

    if analysis.topics:
        lines.append(f"Topics: {', '.join(analysis.topics)}")

    return "\n".join(lines)


def _to_reduced(analysis: ChunkAnalysis | ReducedAnalysis) -> ReducedAnalysis:
    return ReducedAnalysis(
        summary=analysis.summary,
        key_points=list(analysis.key_points),
        topics=list(analysis.topics),
    )


def _batched(items: Sequence[ReducedAnalysis], size: int) -> list[list[ReducedAnalysis]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


async def _fold_batch(
    llm: StructuredLLM,
    batch: Sequence[ReducedAnalysis],
    config: AnalysisConfig,
    language_code: str | None,
) -> ReducedAnalysis:
    """Merge one batch through the model."""
    if len(batch) == 1:
        return batch[0]

    rendered = "\n\n".join(render_analysis(a, i) for i, a in enumerate(batch))

    async def attempt() -> ReducedAnalysis:
        return await llm.parse(
            model=config.reduce_model,
            instructions=reduce_instructions(language_code),
            prompt=reduce_prompt(rendered),
            schema=ReducedAnalysis,
        )

    return await retry_async(
        attempt,
        attempts=max(1, config.max_retries),
        base_sec=config.backoff_base_sec,
    )


async def reduce_analyses(
    llm: StructuredLLM,
    analyses: Sequence[ChunkAnalysis | ReducedAnalysis],
    config: AnalysisConfig,
    language_code: str | None = None,
) -> ReducedAnalysis:
    """Fold analyses in batches until one remains."""
    if not analyses:
        return ReducedAnalysis(summary="", key_points=[], topics=[])

    current = [_to_reduced(a) for a in analyses]
    rounds = 0

    while len(current) > 1:
        batches = _batched(current, max(2, config.reduce_batch_size))
        current = [
            result
            for result in await gather_bounded(
                config.map_concurrency,
                [_fold_batch(llm, batch, config, language_code) for batch in batches],
            )
            if isinstance(result, ReducedAnalysis)
        ]
        rounds += 1

        if not current:
            # Every fold in this round failed; there is nothing left to merge.
            logger.error("reduce round produced no results", round=rounds)
            break

        logger.debug("reduce round complete", round=rounds, remaining=len(current))

    if not current:
        return ReducedAnalysis(summary="", key_points=[], topics=[])

    logger.info("reduce complete", rounds=rounds, inputs=len(analyses))
    return current[0]


def _dedupe_preserving_order(values: Sequence[str]) -> list[str]:
    """Remove case-insensitive duplicates, keeping first occurrence and casing."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        marker = cleaned.casefold()
        if cleaned and marker not in seen:
            seen.add(marker)
            result.append(cleaned)
    return result


def _order_key_points(points: Sequence[KeyPoint]) -> list[KeyPoint]:
    """Order key points by evidence timestamp, keeping unanchored ones last.

    Chronological order makes the list readable as a walkthrough of the video.
    Points without evidence keep their relative order at the end rather than
    being assigned a position they did not earn.
    """
    anchored = [p for p in points if p.start is not None]
    unanchored = [p for p in points if p.start is None]
    anchored.sort(key=lambda p: p.start or 0.0)
    return anchored + unanchored


def finalise(
    reduced: ReducedAnalysis, *, chunk_count: int, failed_chunks: int
) -> Analysis:
    """Convert the merged analysis into the public contract.

    Deduplication runs again here rather than trusting the model to have done it.
    Overlapping chunks make near-duplicates likely, and a deterministic pass is
    cheaper and more reliable than another round of asking.

    An empty summary becomes `None`: the analysis produced nothing, and saying so
    is more honest than returning an empty string that reads like a summary which
    happened to be blank.
    """
    summary = (reduced.summary or "").strip()
    ordered = _order_key_points(reduced.key_points)

    return Analysis(
        summary=summary or None,
        key_points=_dedupe_preserving_order([p.text for p in ordered]),
        topics=_dedupe_preserving_order(reduced.topics),
        chunk_count=chunk_count,
        failed_chunks=failed_chunks,
    )


__all__ = ["finalise", "reduce_analyses", "render_analysis"]
