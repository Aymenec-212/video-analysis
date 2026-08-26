"""Tests for the map and reduce stages (AD-7, AD-8)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.analysis.chunking import Chunk
from app.analysis.map import map_chunks
from app.analysis.reduce import finalise, reduce_analyses, render_analysis
from app.analysis.schemas import ChunkAnalysis, KeyPoint, ReducedAnalysis
from app.core.config import AnalysisConfig
from app.core.errors import AnalysisFailedError
from app.core.models import Segment
from tests.fixtures.llm import FakeLLM, chunk_analysis, reduced_analysis


def chunk(index: int = 0, text: str = "Bonjour.") -> Chunk:
    return Chunk(
        index=index,
        segments=[
            Segment(start=float(index), end=index + 2.0, speaker="SPEAKER_01", text=text)
        ],
        token_count=10,
    )


FAST = AnalysisConfig(max_retries=2, backoff_base_sec=0.001)


class TestMapStage:
    async def test_every_chunk_is_analysed(self) -> None:
        llm = FakeLLM([chunk_analysis("Résumé.")])
        outcome = await map_chunks(llm, [chunk(0), chunk(1), chunk(2)], FAST)

        assert llm.call_count == 3
        assert len(outcome.analyses) == 3
        assert outcome.is_complete is True

    async def test_empty_chunk_list_makes_no_calls(self) -> None:
        llm = FakeLLM()
        outcome = await map_chunks(llm, [], FAST)

        assert llm.call_count == 0
        assert outcome.analyses == []

    async def test_map_uses_the_configured_model(self) -> None:
        llm = FakeLLM([chunk_analysis()])
        await map_chunks(llm, [chunk(0)], AnalysisConfig(map_model="gpt-5-mini"))

        assert llm.models_used() == ["gpt-5-mini"]

    async def test_prompt_carries_the_transcript_excerpt(self) -> None:
        llm = FakeLLM([chunk_analysis()])
        await map_chunks(llm, [chunk(0)], FAST)

        assert "Bonjour." in llm.calls[0]["prompt"]
        assert "SPEAKER_01" in llm.calls[0]["prompt"]

    async def test_detected_language_reaches_the_instructions(self) -> None:
        llm = FakeLLM([chunk_analysis()])
        await map_chunks(llm, [chunk(0)], FAST, language_code="fr")

        assert "fr" in llm.calls[0]["instructions"]

    async def test_concurrency_is_bounded(self) -> None:
        """Firing every chunk at once collects rate limits, not throughput."""
        import asyncio

        live = {"now": 0, "peak": 0}

        async def handler(instructions: str, prompt: str, schema: type[BaseModel]) -> BaseModel:
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
            await asyncio.sleep(0.01)
            live["now"] -= 1
            return chunk_analysis()

        class SlowLLM(FakeLLM):
            async def parse(self, **kwargs: object) -> BaseModel:  # type: ignore[override]
                self.calls.append(dict(kwargs))
                return await handler("", "", ChunkAnalysis)

        llm = SlowLLM()
        await map_chunks(
            llm, [chunk(i) for i in range(12)], AnalysisConfig(map_concurrency=3)
        )

        assert live["peak"] <= 3


class TestMapFailureHandling:
    async def test_partial_failure_keeps_the_successes(self) -> None:
        """Twenty-nine analysed chunks beat discarding all thirty.

        The failure is keyed on chunk content rather than call order: chunks run
        concurrently, so a counter-based double would fail whichever chunk
        happened to be scheduled first.
        """

        class SelectiveLLM(FakeLLM):
            async def parse(self, **kwargs: object) -> BaseModel:  # type: ignore[override]
                self.calls.append(dict(kwargs))
                if "POISON" in str(kwargs.get("prompt", "")):
                    raise RuntimeError("upstream error")
                return chunk_analysis()

        outcome = await map_chunks(
            SelectiveLLM(),
            [chunk(0, "Bonjour."), chunk(1, "POISON"), chunk(2, "Merci.")],
            FAST,
        )

        assert len(outcome.analyses) == 2
        assert outcome.failed_chunks == 1
        assert outcome.total_chunks == 3
        assert outcome.is_complete is False

    async def test_total_failure_raises_rather_than_inventing(self) -> None:
        """With nothing analysed, a summary could only be fabricated."""
        llm = FakeLLM([RuntimeError("upstream down")])

        with pytest.raises(AnalysisFailedError):
            await map_chunks(llm, [chunk(0), chunk(1)], FAST)

    async def test_transient_failure_is_retried(self) -> None:
        calls = {"n": 0}

        class FlakyLLM(FakeLLM):
            async def parse(self, **kwargs: object) -> BaseModel:  # type: ignore[override]
                self.calls.append(dict(kwargs))
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("transient")
                return chunk_analysis()

        outcome = await map_chunks(FlakyLLM(), [chunk(0)], FAST)

        assert calls["n"] == 2
        assert len(outcome.analyses) == 1


class TestHierarchicalReduce:
    async def test_single_analysis_skips_the_model(self) -> None:
        """Nothing to merge; re-summarising only loses detail and spends a call."""
        llm = FakeLLM()
        result = await reduce_analyses(llm, [chunk_analysis("Seul résumé.")], FAST)

        assert llm.call_count == 0
        assert result.summary == "Seul résumé."

    async def test_empty_input_returns_an_empty_analysis(self) -> None:
        llm = FakeLLM()
        result = await reduce_analyses(llm, [], FAST)

        assert llm.call_count == 0
        assert result.summary == ""

    async def test_small_set_folds_in_one_round(self) -> None:
        llm = FakeLLM([reduced_analysis("Fusionné.")])
        result = await reduce_analyses(
            llm, [chunk_analysis() for _ in range(3)], AnalysisConfig(reduce_batch_size=8)
        )

        assert llm.call_count == 1
        assert result.summary == "Fusionné."

    async def test_large_set_folds_in_multiple_rounds(self) -> None:
        """The property that makes the strategy hold for any length.

        Twenty analyses at batch size four: round one is five folds producing
        five results; round two batches those as [4, 1], and the single-item
        batch returns without a call, so one fold; round three merges the
        remaining two. Six calls, never one prompt containing everything.
        """
        llm = FakeLLM([reduced_analysis("Fusionné.")])
        result = await reduce_analyses(
            llm, [chunk_analysis() for _ in range(20)], AnalysisConfig(reduce_batch_size=4)
        )

        assert llm.call_count == 5 + 1 + 1
        assert result.summary == "Fusionné."

    async def test_no_single_prompt_contains_every_analysis(self) -> None:
        """The exact failure hierarchical folding exists to prevent."""
        llm = FakeLLM([reduced_analysis()])
        analyses = [chunk_analysis(f"Résumé numéro {i}.") for i in range(20)]

        await reduce_analyses(llm, analyses, AnalysisConfig(reduce_batch_size=4))

        for call in llm.calls:
            present = sum(1 for i in range(20) if f"Résumé numéro {i}." in call["prompt"])
            assert present <= 4

    async def test_reduce_uses_the_configured_model(self) -> None:
        llm = FakeLLM([reduced_analysis()])
        await reduce_analyses(
            llm,
            [chunk_analysis(), chunk_analysis()],
            AnalysisConfig(reduce_model="gpt-5"),
        )

        assert llm.models_used() == ["gpt-5"]


class TestRendering:
    def test_rendered_analysis_carries_points_and_timestamps(self) -> None:
        rendered = render_analysis(
            chunk_analysis("Résumé.", [("Un point", 12.4)], ["AI"]), 0
        )

        assert "Résumé." in rendered
        assert "Un point" in rendered
        assert "12.4s" in rendered
        assert "AI" in rendered

    def test_empty_analysis_renders_without_fabrication(self) -> None:
        rendered = render_analysis(ChunkAnalysis(summary="", key_points=[], topics=[]), 0)
        assert "(none)" in rendered


class TestFinalise:
    def test_produces_the_public_contract(self) -> None:
        reduced = ReducedAnalysis(
            summary="Cette vidéo présente un projet.",
            key_points=[KeyPoint(text="Présentation du projet", start=1.0, end=2.0)],
            topics=["Artificial Intelligence"],
        )
        analysis = finalise(reduced, chunk_count=3, failed_chunks=0)

        assert analysis.summary == "Cette vidéo présente un projet."
        assert analysis.key_points == ["Présentation du projet"]
        assert analysis.topics == ["Artificial Intelligence"]
        assert analysis.is_complete is True

    def test_duplicate_topics_are_removed_case_insensitively(self) -> None:
        """Overlapping chunks make near-duplicates likely."""
        reduced = ReducedAnalysis(
            summary="s",
            key_points=[],
            topics=["Cloud Computing", "cloud computing", "  Cloud Computing  ", "AI"],
        )
        analysis = finalise(reduced, chunk_count=1, failed_chunks=0)

        assert analysis.topics == ["Cloud Computing", "AI"]

    def test_duplicate_key_points_are_removed(self) -> None:
        reduced = ReducedAnalysis(
            summary="s",
            key_points=[
                KeyPoint(text="Un point", start=1.0, end=2.0),
                KeyPoint(text="un point", start=5.0, end=6.0),
            ],
            topics=[],
        )
        analysis = finalise(reduced, chunk_count=1, failed_chunks=0)

        assert analysis.key_points == ["Un point"]

    def test_key_points_are_ordered_chronologically(self) -> None:
        reduced = ReducedAnalysis(
            summary="s",
            key_points=[
                KeyPoint(text="Troisième", start=30.0, end=31.0),
                KeyPoint(text="Premier", start=1.0, end=2.0),
                KeyPoint(text="Deuxième", start=15.0, end=16.0),
            ],
            topics=[],
        )
        analysis = finalise(reduced, chunk_count=1, failed_chunks=0)

        assert analysis.key_points == ["Premier", "Deuxième", "Troisième"]

    def test_unanchored_points_keep_their_order_at_the_end(self) -> None:
        """They did not earn a position, so they are not assigned one."""
        reduced = ReducedAnalysis(
            summary="s",
            key_points=[
                KeyPoint(text="Sans horodatage", start=None, end=None),
                KeyPoint(text="Avec horodatage", start=5.0, end=6.0),
            ],
            topics=[],
        )
        analysis = finalise(reduced, chunk_count=1, failed_chunks=0)

        assert analysis.key_points == ["Avec horodatage", "Sans horodatage"]

    def test_empty_summary_becomes_null(self) -> None:
        """Null says the analysis produced nothing; "" reads like a blank summary."""
        analysis = finalise(
            ReducedAnalysis(summary="   ", key_points=[], topics=[]),
            chunk_count=1,
            failed_chunks=0,
        )
        assert analysis.summary is None

    def test_partial_coverage_is_reported_not_hidden(self) -> None:
        analysis = finalise(
            reduced_analysis("Résumé partiel."), chunk_count=10, failed_chunks=3
        )

        assert analysis.failed_chunks == 3
        assert analysis.is_complete is False
