"""Tests for media probing (SPEC 5.1, 5.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.audio.probe import MediaProbe, probe_media
from app.core.errors import ErrorCode, UnreadableMediaError
from tests.fixtures.media import ffmpeg_required

pytestmark = ffmpeg_required


class TestValidMedia:
    async def test_reports_duration_from_the_container(
        self, media_corpus: dict[str, Path]
    ) -> None:
        probe = await probe_media(media_corpus["tone"])
        assert probe.duration == pytest.approx(3.0, abs=0.15)

    async def test_reports_audio_stream_parameters(
        self, media_corpus: dict[str, Path]
    ) -> None:
        probe = await probe_media(media_corpus["stereo"])
        stream = probe.audio_streams[0]

        assert stream.channels == 2
        assert stream.sample_rate == 48_000
        assert probe.has_audio is True

    async def test_detects_both_streams_in_a_video(
        self, media_corpus: dict[str, Path]
    ) -> None:
        probe = await probe_media(media_corpus["with_audio"])

        assert probe.has_audio is True
        assert probe.has_video is True
        assert probe.video_stream_count == 1

    async def test_reports_file_size(self, media_corpus: dict[str, Path]) -> None:
        probe = await probe_media(media_corpus["tone"])
        assert probe.size_bytes > 0


class TestSilenceIsValid:
    """Silence is a transcription finding, not an ingestion failure.

    The taxonomy places NO_SPEECH_DETECTED in the transcription stage, so a
    silent file must pass ingestion cleanly and reach the speech backend.
    """

    async def test_silent_audio_probes_successfully(
        self, media_corpus: dict[str, Path]
    ) -> None:
        probe = await probe_media(media_corpus["silent"])

        assert probe.has_audio is True
        assert probe.duration == pytest.approx(2.0, abs=0.15)


class TestAudiolessMedia:
    """Probes successfully but carries nothing to transcribe.

    Detecting this here is what lets the pipeline raise NO_AUDIO_STREAM instead
    of surfacing FFmpeg's opaque downstream failure as UNREADABLE_MEDIA.
    """

    async def test_video_without_audio_still_probes(
        self, media_corpus: dict[str, Path]
    ) -> None:
        probe = await probe_media(media_corpus["no_audio"])

        assert probe.has_video is True
        assert probe.has_audio is False
        assert probe.audio_streams == []


class TestUnreadableMedia:
    async def test_corrupt_bytes_are_rejected(
        self, media_corpus: dict[str, Path]
    ) -> None:
        with pytest.raises(UnreadableMediaError) as exc_info:
            await probe_media(media_corpus["corrupt"])

        assert exc_info.value.code is ErrorCode.UNREADABLE_MEDIA

    async def test_text_file_with_media_extension_is_rejected(
        self, media_corpus: dict[str, Path]
    ) -> None:
        """The reason extension-based validation would be worthless."""
        with pytest.raises(UnreadableMediaError):
            await probe_media(media_corpus["fake"])

    async def test_empty_file_is_rejected(self, media_corpus: dict[str, Path]) -> None:
        with pytest.raises(UnreadableMediaError):
            await probe_media(media_corpus["empty"])

    async def test_missing_path_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(UnreadableMediaError):
            await probe_media(tmp_path / "nowhere.mp4")

    async def test_failure_detail_carries_the_ffprobe_message(
        self, media_corpus: dict[str, Path]
    ) -> None:
        """Diagnosis without needing to reproduce the failure locally."""
        with pytest.raises(UnreadableMediaError) as exc_info:
            await probe_media(media_corpus["fake"])

        assert exc_info.value.detail.get("ffprobe_error")

    async def test_unreadable_media_is_fatal(
        self, media_corpus: dict[str, Path]
    ) -> None:
        with pytest.raises(UnreadableMediaError) as exc_info:
            await probe_media(media_corpus["corrupt"])

        assert exc_info.value.is_fatal is True
        assert exc_info.value.http_status == 422


class TestDurationCoercion:
    """ffprobe emits numbers as strings and sometimes as 'N/A'."""

    def test_absent_duration_is_none_not_zero(self) -> None:
        assert MediaProbe().duration is None

    def test_has_audio_is_false_without_streams(self) -> None:
        assert MediaProbe().has_audio is False
