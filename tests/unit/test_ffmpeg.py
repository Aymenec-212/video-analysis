"""Tests for audio normalisation (SPEC 5.2, AD-2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.audio.ffmpeg import build_ffmpeg_command, compute_sha256, normalize_audio
from app.audio.probe import probe_media
from app.core.config import AudioConfig
from app.core.errors import NoAudioStreamError, UnreadableMediaError
from tests.fixtures.media import ffmpeg_required

pytestmark = ffmpeg_required


class TestCommandConstruction:
    """Flags are asserted directly so reproducibility cannot regress silently."""

    def test_reproducibility_flags_are_always_present(self) -> None:
        """Without these the output hash changes on every FFmpeg upgrade.

        Measured: default FLAC output embeds the encoder version string, which
        would invalidate every committed AD-2 fixture on a different machine.
        """
        command = [str(a) for a in build_ffmpeg_command(
            Path("in.mp4"), Path("out.flac"), AudioConfig()
        )]

        assert "-bitexact" in command
        assert "-map_metadata" in command
        assert command[command.index("-map_metadata") + 1] == "-1"

    def test_video_is_discarded(self) -> None:
        command = [str(a) for a in build_ffmpeg_command(
            Path("in.mp4"), Path("out.flac"), AudioConfig()
        )]
        assert "-vn" in command

    def test_stdin_is_never_read(self) -> None:
        """Without -nostdin, FFmpeg can block on input it will never receive."""
        command = [str(a) for a in build_ffmpeg_command(
            Path("in.mp4"), Path("out.flac"), AudioConfig()
        )]
        assert "-nostdin" in command

    def test_target_format_comes_from_configuration(self) -> None:
        command = [str(a) for a in build_ffmpeg_command(
            Path("in.mp4"), Path("out.flac"), AudioConfig()
        )]

        assert command[command.index("-ac") + 1] == "1"
        assert command[command.index("-ar") + 1] == "16000"

    def test_denoising_absent_by_default(self) -> None:
        """AD-4 reasoning: the default is decided by measurement, not assumption."""
        command = [str(a) for a in build_ffmpeg_command(
            Path("in.mp4"), Path("out.flac"), AudioConfig()
        )]
        assert "afftdn" not in command

    def test_denoising_appears_only_when_enabled(self) -> None:
        command = [str(a) for a in build_ffmpeg_command(
            Path("in.mp4"), Path("out.flac"), AudioConfig(denoise=True)
        )]
        assert "afftdn" in command


class TestNormalisation:
    async def test_produces_mono_16khz_flac(
        self, media_corpus: dict[str, Path], tmp_path: Path
    ) -> None:
        audio = await normalize_audio(media_corpus["stereo"], tmp_path, AudioConfig())
        probe = await probe_media(audio.path)

        assert probe.audio_streams[0].channels == 1
        assert probe.audio_streams[0].sample_rate == 16_000
        assert probe.audio_streams[0].codec_name == "flac"

    async def test_extracts_audio_from_video(
        self, media_corpus: dict[str, Path], tmp_path: Path
    ) -> None:
        audio = await normalize_audio(media_corpus["with_audio"], tmp_path, AudioConfig())
        probe = await probe_media(audio.path)

        assert probe.has_audio is True
        assert probe.has_video is False, "video must be dropped, not carried through"

    async def test_reports_measured_parameters(
        self, media_corpus: dict[str, Path], tmp_path: Path
    ) -> None:
        """Values are re-probed from the output rather than echoed from config."""
        audio = await normalize_audio(media_corpus["tone"], tmp_path, AudioConfig())

        assert audio.sample_rate == 16_000
        assert audio.channels == 1
        assert audio.size_bytes > 0
        assert audio.duration == pytest.approx(3.0, abs=0.2)

    async def test_silent_audio_normalises_successfully(
        self, media_corpus: dict[str, Path], tmp_path: Path
    ) -> None:
        """Silence must reach the speech backend to be reported as no-speech."""
        audio = await normalize_audio(media_corpus["silent"], tmp_path, AudioConfig())
        assert audio.size_bytes > 0


class TestCacheKeyReproducibility:
    """AD-2 depends on the same audio always hashing to the same value."""

    async def test_identical_input_yields_identical_hash(
        self, media_corpus: dict[str, Path], tmp_path: Path
    ) -> None:
        first = await normalize_audio(media_corpus["tone"], tmp_path / "a", AudioConfig())
        second = await normalize_audio(media_corpus["tone"], tmp_path / "b", AudioConfig())

        assert first.sha256 == second.sha256

    async def test_output_carries_no_encoder_version_stamp(
        self, media_corpus: dict[str, Path], tmp_path: Path
    ) -> None:
        """The concrete reason the hash is portable across FFmpeg versions."""
        audio = await normalize_audio(media_corpus["tone"], tmp_path, AudioConfig())
        raw = audio.path.read_bytes()

        assert b"Lavf" not in raw, "libavformat version leaked into the output"
        assert b"encoder=" not in raw

    async def test_different_audio_yields_different_hash(
        self, media_corpus: dict[str, Path], tmp_path: Path
    ) -> None:
        first = await normalize_audio(media_corpus["tone"], tmp_path / "a", AudioConfig())
        second = await normalize_audio(media_corpus["stereo"], tmp_path / "b", AudioConfig())

        assert first.sha256 != second.sha256

    def test_hash_matches_a_plain_file_digest(self, tmp_path: Path) -> None:
        import hashlib

        target = tmp_path / "sample.bin"
        payload = b"deterministic content" * 1000
        target.write_bytes(payload)

        assert compute_sha256(target) == hashlib.sha256(payload).hexdigest()


class TestFailureModes:
    async def test_audioless_media_raises_no_audio_stream(
        self, media_corpus: dict[str, Path], tmp_path: Path
    ) -> None:
        """Not UNREADABLE_MEDIA — the file is perfectly readable, just silent.

        Detected before FFmpeg runs, which is the only way to get the right code:
        FFmpeg's own failure here is an opaque exit with no usable signal.
        """
        with pytest.raises(NoAudioStreamError) as exc_info:
            await normalize_audio(media_corpus["no_audio"], tmp_path, AudioConfig())

        assert exc_info.value.http_status == 422

    async def test_corrupt_media_raises_unreadable(
        self, media_corpus: dict[str, Path], tmp_path: Path
    ) -> None:
        with pytest.raises(UnreadableMediaError):
            await normalize_audio(media_corpus["corrupt"], tmp_path, AudioConfig())

    async def test_missing_source_raises_unreadable(self, tmp_path: Path) -> None:
        with pytest.raises(UnreadableMediaError):
            await normalize_audio(tmp_path / "absent.mp4", tmp_path, AudioConfig())
