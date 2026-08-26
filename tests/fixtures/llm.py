"""A `StructuredLLM` test double.

Satisfies the Protocol by shape, so the analysis stages are exercised offline with
no SDK, no network, and no keys — the AD-2 promise applied to the LLM layer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel

from app.analysis.schemas import ChunkAnalysis, KeyPoint, ReducedAnalysis


class FakeLLM:
    """Returns scripted responses and records what it was asked."""

    def __init__(
        self,
        responses: Sequence[BaseModel | Exception] | None = None,
        *,
        handler: Callable[[str, str, type[BaseModel]], BaseModel] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def models_used(self) -> list[str]:
        return [call["model"] for call in self.calls]

    async def parse(
        self,
        *,
        model: str,
        instructions: str,
        prompt: str,
        schema: type[BaseModel],
    ) -> Any:
        self.calls.append(
            {
                "model": model,
                "instructions": instructions,
                "prompt": prompt,
                "schema": schema,
            }
        )

        if self._handler is not None:
            return self._handler(instructions, prompt, schema)

        if not self._responses:
            return _default_for(schema)

        # Scripted responses are consumed in order; the last one repeats so a
        # test does not have to enumerate every call it does not care about.
        response = (
            self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        )
        if isinstance(response, Exception):
            raise response
        return response


def _default_for(schema: type[BaseModel]) -> BaseModel:
    if schema is ChunkAnalysis:
        return chunk_analysis("A summary.")
    return ReducedAnalysis(summary="A merged summary.", key_points=[], topics=[])


def chunk_analysis(
    summary: str = "A summary.",
    points: Sequence[tuple[str, float | None]] = (),
    topics: Sequence[str] = (),
) -> ChunkAnalysis:
    return ChunkAnalysis(
        summary=summary,
        key_points=[
            KeyPoint(text=text, start=start, end=None if start is None else start + 1.0)
            for text, start in points
        ],
        topics=list(topics),
    )


def reduced_analysis(
    summary: str = "A merged summary.",
    points: Sequence[tuple[str, float | None]] = (),
    topics: Sequence[str] = (),
) -> ReducedAnalysis:
    return ReducedAnalysis(
        summary=summary,
        key_points=[
            KeyPoint(text=text, start=start, end=None if start is None else start + 1.0)
            for text, start in points
        ],
        topics=list(topics),
    )
