"""The map stage: analyse each chunk concurrently (AD-7).

Chunks are independent, so they run in parallel under a semaphore. The bound
matters — firing every chunk of a long transcript at once collects rate limits
rather than throughput.

**Partial failure is tolerated and reported.** If one chunk of thirty fails after
retries, analysing the other twenty-nine and saying so is better than discarding
all of it. The count travels with the result, so a summary built from incomplete
coverage is never presented as complete. Only when *every* chunk fails does the
stage raise, because at that point there is nothing to summarise and producing
one anyway would be exactly the fabrication the brief forbids.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from app.core.concurrency import gather_bounded, retry_async
from app.core.config import AnalysisConfig
from app.core.errors import AnalysisFailedError
from app.core.logging import get_logger

from .chunking import Chunk
from .llm import StructuredLLM
from .prompts import map_instructions, map_prompt
from .schemas import ChunkAnalysis

logger = get_logger(__name__)


class MapOutcome(BaseModel):
    """Results of the map stage, including what did not succeed."""

    analyses: list[ChunkAnalysis] = Field(default_factory=list)
    failed_chunks: int = 0
    total_chunks: int = 0

    @property
    def is_complete(self) -> bool:
        return self.failed_chunks == 0


async def analyze_chunk(
    llm: StructuredLLM,
    chunk: Chunk,
    config: AnalysisConfig,
    language_code: str | None,
) -> ChunkAnalysis:
    """Analyse one chunk, retrying transient failures."""

    async def attempt() -> ChunkAnalysis:
        return await llm.parse(
            model=config.map_model,
            instructions=map_instructions(language_code),
            prompt=map_prompt(chunk.render(), chunk.start, chunk.end),
            schema=ChunkAnalysis,
        )

    return await retry_async(
        attempt,
        attempts=max(1, config.max_retries),
        base_sec=config.backoff_base_sec,
    )


async def map_chunks(
    llm: StructuredLLM,
    chunks: Sequence[Chunk],
    config: AnalysisConfig,
    language_code: str | None = None,
) -> MapOutcome:
    """Analyse every chunk concurrently, bounded by `map_concurrency`.

    Raises `AnalysisFailedError` only when no chunk succeeded.
    """
    if not chunks:
        return MapOutcome()

    results = await gather_bounded(
        config.map_concurrency,
        [analyze_chunk(llm, chunk, config, language_code) for chunk in chunks],
        return_exceptions=True,
    )

    analyses: list[ChunkAnalysis] = []
    failures: list[BaseException] = []

    for chunk, result in zip(chunks, results, strict=True):
        if isinstance(result, ChunkAnalysis):
            analyses.append(result)
        else:
            failures.append(result)
            logger.warning(
                "chunk analysis failed",
                chunk_index=chunk.index,
                error=str(result)[:200],
            )

    if not analyses:
        raise AnalysisFailedError(
            "Every excerpt failed to analyse.",
            detail={"chunks": len(chunks)},
            cause=failures[0] if failures else None,
        )

    outcome = MapOutcome(
        analyses=analyses,
        failed_chunks=len(failures),
        total_chunks=len(chunks),
    )
    logger.info(
        "map stage complete",
        analysed=len(analyses),
        failed=outcome.failed_chunks,
        concurrency=config.map_concurrency,
    )
    return outcome


__all__ = ["MapOutcome", "analyze_chunk", "map_chunks"]
