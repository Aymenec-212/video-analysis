"""Tests for transcript chunking (AD-7)."""

from __future__ import annotations

from app.analysis.chunking import (
    Chunk,
    TokenCounter,
    build_chunks,
    render_segment,
)
from app.core.config import AnalysisConfig
from app.core.models import Segment


class FixedCounter:
    """Deterministic counter: one token per word.

    Keeps the boundary assertions exact and independent of whichever vocabulary
    the environment can reach.
    """

    is_exact = True

    def count(self, text: str) -> int:
        return len(text.split())


def seg(start: float, end: float, speaker: str = "SPEAKER_01", words: int = 10) -> Segment:
    return Segment(start=start, end=end, speaker=speaker, text=" ".join(["mot"] * words))


class TestTokenCounter:
    def test_empty_text_is_zero(self) -> None:
        assert TokenCounter("o200k_base").count("") == 0

    def test_counts_are_positive_and_scale_with_length(self) -> None:
        counter = TokenCounter("o200k_base")
        short = counter.count("Bonjour")
        long = counter.count("Bonjour et bienvenue dans cette émission de télévision.")

        assert short >= 1
        assert long > short

    def test_falls_back_when_the_vocabulary_is_unreachable(self) -> None:
        """tiktoken downloads its vocabulary on first use.

        An unresolvable encoding must degrade to an estimate rather than raising
        an HTTP error from inside chunking.
        """
        counter = TokenCounter("definitely-not-a-real-encoding")

        assert counter.count("Bonjour et bienvenue") >= 1
        assert counter.is_exact is False

    def test_fallback_estimate_runs_high(self) -> None:
        """Over-counting shrinks chunks; under-counting overruns the budget."""
        counter = TokenCounter("definitely-not-a-real-encoding")
        text = "a" * 100

        assert counter.count(text) >= 100 / 4


class TestRendering:
    def test_segment_carries_speaker_and_timestamps(self) -> None:
        """The model needs both to attribute statements and cite evidence."""
        rendered = render_segment(
            Segment(start=12.4, end=17.8, speaker="SPEAKER_02", text="Merci.")
        )

        assert "SPEAKER_02" in rendered
        assert "12.4s" in rendered
        assert "17.8s" in rendered
        assert "Merci." in rendered


class TestBudget:
    def test_chunks_stay_within_the_budget(self) -> None:
        segments = [seg(float(i) * 5, i * 5 + 4, words=10) for i in range(40)]
        chunks = build_chunks(
            segments,
            AnalysisConfig(chunk_token_budget=50, chunk_overlap_segments=0),
            TokenCounter("x"),  # replaced below
        )
        # Recompute with the deterministic counter for an exact assertion.
        chunks = build_chunks(
            segments,
            AnalysisConfig(chunk_token_budget=50, chunk_overlap_segments=0),
            FixedCounter(),  # type: ignore[arg-type]
        )

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.token_count <= 50

    def test_short_transcript_is_a_single_chunk(self) -> None:
        chunks = build_chunks(
            [seg(0.0, 4.0), seg(5.0, 9.0)],
            AnalysisConfig(chunk_token_budget=2000),
            FixedCounter(),  # type: ignore[arg-type]
        )
        assert len(chunks) == 1

    def test_empty_input_yields_no_chunks(self) -> None:
        assert build_chunks([], AnalysisConfig()) == []


class TestSegmentAlignment:
    def test_segments_are_never_split(self) -> None:
        """A chunk ending mid-utterance invites the model to guess the rest."""
        segments = [seg(float(i) * 5, i * 5 + 4, words=10) for i in range(20)]
        chunks = build_chunks(
            segments,
            AnalysisConfig(chunk_token_budget=45, chunk_overlap_segments=0),
            FixedCounter(),  # type: ignore[arg-type]
        )

        rebuilt = [s for chunk in chunks for s in chunk.segments]
        assert rebuilt == segments

    def test_oversized_segment_becomes_its_own_chunk(self) -> None:
        """Never split, so an oversized segment stands alone rather than breaking."""
        segments = [seg(0.0, 4.0, words=5), seg(5.0, 60.0, words=500), seg(61.0, 65.0, words=5)]
        chunks = build_chunks(
            segments,
            AnalysisConfig(chunk_token_budget=50, chunk_overlap_segments=0),
            FixedCounter(),  # type: ignore[arg-type]
        )

        assert any(len(c.segments) == 1 and c.token_count > 50 for c in chunks)
        rebuilt = [s for chunk in chunks for s in chunk.segments]
        assert rebuilt == segments


class TestOverlap:
    def test_overlap_repeats_trailing_segments(self) -> None:
        """A point spanning a boundary would otherwise be lost to both sides."""
        segments = [seg(float(i) * 5, i * 5 + 4, words=10) for i in range(12)]
        chunks = build_chunks(
            segments,
            AnalysisConfig(chunk_token_budget=50, chunk_overlap_segments=1),
            FixedCounter(),  # type: ignore[arg-type]
        )

        assert len(chunks) > 1
        assert chunks[1].segments[0] == chunks[0].segments[-1]

    def test_zero_overlap_produces_a_clean_partition(self) -> None:
        segments = [seg(float(i) * 5, i * 5 + 4, words=10) for i in range(12)]
        chunks = build_chunks(
            segments,
            AnalysisConfig(chunk_token_budget=50, chunk_overlap_segments=0),
            FixedCounter(),  # type: ignore[arg-type]
        )

        rebuilt = [s for chunk in chunks for s in chunk.segments]
        assert len(rebuilt) == len(segments)

    def test_oversized_overlap_is_dropped(self) -> None:
        """Carrying an oversized segment forward would repeat it indefinitely."""
        segments = [seg(0.0, 4.0, words=5), seg(5.0, 60.0, words=200), seg(61.0, 65.0, words=5)]
        chunks = build_chunks(
            segments,
            AnalysisConfig(chunk_token_budget=50, chunk_overlap_segments=1),
            FixedCounter(),  # type: ignore[arg-type]
        )

        occurrences = sum(
            1 for chunk in chunks for s in chunk.segments if s.start == 5.0
        )
        assert occurrences == 1


class TestChunkMetadata:
    def test_chunks_are_indexed_in_order(self) -> None:
        segments = [seg(float(i) * 5, i * 5 + 4, words=10) for i in range(20)]
        chunks = build_chunks(
            segments,
            AnalysisConfig(chunk_token_budget=45, chunk_overlap_segments=0),
            FixedCounter(),  # type: ignore[arg-type]
        )

        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_time_range_spans_the_contained_segments(self) -> None:
        chunk = Chunk(index=0, segments=[seg(3.0, 7.0), seg(8.0, 12.0)])

        assert chunk.start == 3.0
        assert chunk.end == 12.0

    def test_rendered_chunk_contains_every_segment(self) -> None:
        chunk = Chunk(
            index=0,
            segments=[
                Segment(start=0.0, end=2.0, speaker="SPEAKER_01", text="Bonjour."),
                Segment(start=3.0, end=5.0, speaker="SPEAKER_02", text="Merci."),
            ],
        )
        rendered = chunk.render()

        assert "Bonjour." in rendered
        assert "Merci." in rendered
        assert rendered.count("\n") == 1
