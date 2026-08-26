"""Request dependencies.

Clients are built **lazily, on first use** rather than at startup. That is the
whole design here, and it exists to keep one promise: a clone with no API keys
must still import, start, serve `/health`, and pass its tests (AD-2). Building
the speech and language clients eagerly would make a missing key a startup crash,
so an evaluator without credentials could not see the service run at all.

Built once and cached, so the underlying HTTP connection pools are reused across
requests instead of being rebuilt per call.

Tests inject fakes by constructing `Dependencies` with them supplied, which is
why the fields are plain attributes rather than private.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import Request

from app.analysis.llm import OpenAIStructuredLLM, StructuredLLM
from app.core.config import Settings, get_settings
from app.stt.base import TranscriptionBackend
from app.stt.cache import ResponseCache
from app.stt.deepgram import DeepgramBackend


@dataclass
class Dependencies:
    """Shared, lazily constructed services."""

    settings: Settings = field(default_factory=get_settings)
    backend: TranscriptionBackend | None = None
    llm: StructuredLLM | None = None

    def get_backend(self) -> TranscriptionBackend:
        """The speech backend, built on first use.

        Raises `ConfigurationError` when no key is configured — at request time,
        never at startup.
        """
        if self.backend is None:
            self.backend = DeepgramBackend(
                api_key=self.settings.require_deepgram_key(),
                config=self.settings.deepgram,
                cache=ResponseCache(self.settings.cache),
            )
        return self.backend

    def get_llm(self) -> StructuredLLM:
        """The language model client, built on first use.

        A missing key raises `ConfigurationError`, which the pipeline treats as
        an analysis failure rather than a fatal one: the transcript is still real
        work and is still returned.
        """
        if self.llm is None:
            self.llm = OpenAIStructuredLLM(
                api_key=self.settings.require_openai_key(),
                timeout_sec=self.settings.analysis.timeout_sec,
            )
        return self.llm

    async def aclose(self) -> None:
        """Release whatever was actually constructed."""
        for service in (self.backend, self.llm):
            closer = getattr(service, "aclose", None)
            if closer is not None:
                await closer()


def get_dependencies(request: Request) -> Dependencies:
    """Resolve dependencies from application state."""
    dependencies = getattr(request.app.state, "dependencies", None)
    if dependencies is None:  # pragma: no cover - lifespan always sets this
        dependencies = Dependencies()
        request.app.state.dependencies = dependencies
    return dependencies


__all__ = ["Dependencies", "get_dependencies"]
