"""Real media fixtures, generated with FFmpeg at session scope.

These tests deliberately exercise the actual FFmpeg and ffprobe binaries rather
than mocking them. Mocked probe output would only confirm my assumptions about
ffprobe's behaviour, which is precisely the thing worth testing — the audioless
case, for instance, behaves in a way that is easy to guess wrong.

Generating the corpus needs no network, so this keeps the AD-2 promise: the suite
requires zero API keys. It does require FFmpeg, which is a hard prerequisite of
the system anyway; tests skip with a clear reason if it is absent.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ffmpeg_required = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and ffprobe are required to generate media fixtures",
)


def _run(args: list[str]) -> None:
    subprocess.run(args, check=True, capture_output=True)


@pytest.fixture(scope="session")
def media_corpus(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A small set of media covering the cases ingestion must distinguish.

    Session-scoped because encoding costs a second or two and the files are read
    only. Anything that writes must copy first.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is not installed")

    root = tmp_path_factory.mktemp("media")

    tone = root / "tone.wav"
    _run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
          "sine=frequency=440:duration=3", "-ac", "2", "-ar", "44100", str(tone)])

    stereo = root / "stereo.wav"
    _run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
          "sine=frequency=330:duration=2", "-ac", "2", "-ar", "48000", str(stereo)])

    # Valid audio containing no sound. Must ingest cleanly: silence is a
    # transcription-stage finding, not an ingestion failure.
    silent = root / "silent.wav"
    _run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
          "anullsrc=r=44100:cl=mono", "-t", "2", str(silent)])

    with_audio = root / "with_audio.mp4"
    _run(["ffmpeg", "-v", "error", "-y",
          "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
          "-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=10",
          "-map", "1:v", "-map", "0:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-c:a", "aac", str(with_audio)])

    # Probes successfully but has no audio track — the case FFmpeg reports badly.
    no_audio = root / "no_audio.mp4"
    _run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
          "testsrc=duration=2:size=160x120:rate=10", "-c:v", "libx264",
          "-pix_fmt", "yuv420p", "-an", str(no_audio)])

    # Random bytes wearing a media extension.
    corrupt = root / "corrupt.mp4"
    corrupt.write_bytes(bytes(range(256)) * 8)

    # Plain text wearing a media extension: the case that proves extension-based
    # validation would be worthless.
    fake = root / "fake.mp4"
    fake.write_text("this is not media")

    empty = root / "empty.mp4"
    empty.write_bytes(b"")

    return {
        "tone": tone,
        "stereo": stereo,
        "silent": silent,
        "with_audio": with_audio,
        "no_audio": no_audio,
        "corrupt": corrupt,
        "fake": fake,
        "empty": empty,
    }
