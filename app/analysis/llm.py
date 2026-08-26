"""Structured LLM access.

One Protocol with one method, and an OpenAI implementation behind it.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from app.core.errors import AnalysisFailedError
from app.core.logging import get_logger

logger = get_logger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredLLM(Protocol):
    """Generates a response conforming to a Pydantic schema."""

    async def parse(
        self,
        *,
        model: str,
        instructions: str,
        prompt: str,
        schema: type[SchemaT],
    ) -> SchemaT:
        """Return an instance of `schema`, or raise `AnalysisFailedError`."""
        ...


class OpenAIStructuredLLM:
    """`StructuredLLM` over OpenAI Structured Outputs."""

    def __init__(self, api_key: str, timeout_sec: float = 120.0) -> None:
        # Imported here so the module can be imported, and the rest of the
        # analysis package tested, without the SDK being importable.
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout_sec)

    async def parse(
        self,
        *,
        model: str,
        instructions: str,
        prompt: str,
        schema: type[SchemaT],
    ) -> SchemaT:
        response = await self._client.responses.parse(
            model=model,
            instructions=instructions,
            input=prompt,
            text_format=schema,
        )

        parsed = response.output_parsed
        if parsed is None:
            # Structured Outputs can still decline to produce content — a refusal
            # or a length cutoff. Treating that as a failure is correct: the
            # alternative is passing an empty object off as an analysis.
            raise AnalysisFailedError(
                "The language model returned no parsable output.",
                detail={"model": model},
            )
        return parsed

    async def aclose(self) -> None:
        await self._client.close()


__all__ = ["OpenAIStructuredLLM", "StructuredLLM"]
