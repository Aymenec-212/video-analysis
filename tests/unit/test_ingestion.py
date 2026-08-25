"""Tests for ingestion (SPEC 5.1).

URL validation and upload handling are tested without network access. The
yt-dlp download path itself is integration-marked, since it requires a live
source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import IngestionConfig
from app.core.errors import (
    ErrorCode,
    InvalidURLError,
    MediaTooLargeError,
    UnreadableMediaError,
)
from app.ingestion.file_source import load_local_file, save_upload
from app.ingestion.metadata import UPLOAD_SOURCE, MediaSource, SourceKind
from app.ingestion.url_source import validate_url
from tests.fixtures.media import ffmpeg_required


class TestSchemeAllowlist:
    def test_https_is_accepted(self) -> None:
        url = "https://example.com/video"
        assert validate_url(url, IngestionConfig(block_private_addresses=False)) == url

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "ftp://example.com/video.mp4",
            "gopher://example.com/",
            "data:text/plain;base64,SGVsbG8=",
            "javascript:alert(1)",
        ],
    )
    def test_non_http_schemes_are_rejected(self, url: str) -> None:
        with pytest.raises(InvalidURLError) as exc_info:
            validate_url(url, IngestionConfig())

        assert exc_info.value.code is ErrorCode.INVALID_URL

    @pytest.mark.parametrize("url", ["", "   ", "not a url", "https://"])
    def test_malformed_input_is_rejected(self, url: str) -> None:
        with pytest.raises(InvalidURLError):
            validate_url(url, IngestionConfig())


class TestPrivateAddressBlocking:
    """A caller-supplied URL handed to an HTTP client is an SSRF vector."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/video",
            "http://localhost/video",
            "http://0.0.0.0/video",
            "http://10.0.0.5/video",
            "http://192.168.1.1/admin",
            "http://172.16.0.1/",
            "http://[::1]/video",
        ],
    )
    def test_private_and_loopback_targets_are_blocked(self, url: str) -> None:
        with pytest.raises(InvalidURLError):
            validate_url(url, IngestionConfig(block_private_addresses=True))

    def test_cloud_metadata_endpoint_is_blocked(self) -> None:
        """169.254.169.254 returns instance credentials on major cloud providers."""
        with pytest.raises(InvalidURLError):
            validate_url("http://169.254.169.254/latest/meta-data/", IngestionConfig())

    def test_error_message_does_not_reveal_why(self) -> None:
        """Distinguishing 'blocked' from 'does not resolve' discloses internal hosts."""
        with pytest.raises(InvalidURLError) as exc_info:
            validate_url("http://192.168.1.1/", IngestionConfig())

        assert "not publicly routable" in exc_info.value.message

    def test_blocking_can_be_disabled_for_local_testing(self) -> None:
        url = "http://127.0.0.1:8080/fixture.mp4"
        assert validate_url(url, IngestionConfig(block_private_addresses=False)) == url


class _FakeReader:
    """Minimal stand-in for a streaming upload."""

    def __init__(self, payload: bytes, chunk_size: int = 1024) -> None:
        self._payload = payload
        self._chunk_size = chunk_size
        self._offset = 0

    async def read(self, size: int = -1) -> bytes:
        limit = self._chunk_size if size < 0 else min(size, self._chunk_size)
        chunk = self._payload[self._offset : self._offset + limit]
        self._offset += len(chunk)
        return chunk


class TestUploadStreaming:
    async def test_writes_the_full_payload(self, tmp_path: Path) -> None:
        payload = b"x" * 5000
        destination = await save_upload(
            _FakeReader(payload), tmp_path / "upload.bin", max_bytes=10_000
        )

        assert destination.read_bytes() == payload

    async def test_cap_is_enforced_during_the_write(self, tmp_path: Path) -> None:
        """Buffering to measure size would make a large upload a memory attack."""
        with pytest.raises(MediaTooLargeError):
            await save_upload(
                _FakeReader(b"x" * 50_000), tmp_path / "upload.bin", max_bytes=1_000
            )

    async def test_partial_file_is_removed_on_breach(self, tmp_path: Path) -> None:
        destination = tmp_path / "upload.bin"
        with pytest.raises(MediaTooLargeError):
            await save_upload(_FakeReader(b"x" * 50_000), destination, max_bytes=1_000)

        assert not destination.exists(), "aborted upload left bytes on disk"

    async def test_empty_upload_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(UnreadableMediaError):
            await save_upload(_FakeReader(b""), tmp_path / "upload.bin", max_bytes=1_000)


@ffmpeg_required
class TestLocalFileIngestion:
    async def test_builds_media_source_from_a_real_file(
        self, media_corpus: dict[str, Path]
    ) -> None:
        source = await load_local_file(media_corpus["with_audio"], IngestionConfig())

        assert source.kind is SourceKind.UPLOAD
        assert source.source == UPLOAD_SOURCE
        assert source.duration == pytest.approx(2.0, abs=0.2)
        assert source.size_bytes > 0

    async def test_duration_from_ffprobe_is_marked_verified(
        self, media_corpus: dict[str, Path]
    ) -> None:
        """SPEC §5.1 makes ffprobe the authority for duration."""
        source = await load_local_file(media_corpus["tone"], IngestionConfig())
        assert source.duration_verified is True

    async def test_description_is_null_not_invented(
        self, media_corpus: dict[str, Path]
    ) -> None:
        """An uploaded file carries no description; inferring one would fabricate."""
        source = await load_local_file(media_corpus["tone"], IngestionConfig())
        assert source.description is None

    async def test_title_comes_from_the_filename_stem(
        self, media_corpus: dict[str, Path]
    ) -> None:
        source = await load_local_file(media_corpus["tone"], IngestionConfig())
        assert source.title == "tone"

    async def test_supplied_filename_overrides_the_temp_path(
        self, media_corpus: dict[str, Path]
    ) -> None:
        source = await load_local_file(
            media_corpus["tone"], IngestionConfig(), original_filename="Interview AI.mp4"
        )
        assert source.title == "Interview AI"

    async def test_directory_traversal_in_filename_is_stripped(
        self, media_corpus: dict[str, Path]
    ) -> None:
        """A path in a supplied filename is an attack, not information."""
        source = await load_local_file(
            media_corpus["tone"], IngestionConfig(), original_filename="../../etc/passwd.mp4"
        )

        assert "/" not in source.title
        assert source.title == "passwd"

    async def test_validation_is_by_probe_not_extension(
        self, media_corpus: dict[str, Path]
    ) -> None:
        """A text file named .mp4 must be rejected."""
        with pytest.raises(UnreadableMediaError):
            await load_local_file(media_corpus["fake"], IngestionConfig())

    async def test_silent_media_is_accepted(
        self, media_corpus: dict[str, Path]
    ) -> None:
        """No-speech is a transcription outcome, not an ingestion rejection."""
        source = await load_local_file(media_corpus["silent"], IngestionConfig())
        assert source.duration == pytest.approx(2.0, abs=0.2)

    async def test_oversize_file_is_rejected(
        self, media_corpus: dict[str, Path]
    ) -> None:
        with pytest.raises(MediaTooLargeError):
            await load_local_file(media_corpus["tone"], IngestionConfig(max_bytes=10))

    async def test_missing_file_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(UnreadableMediaError):
            await load_local_file(tmp_path / "absent.mp4", IngestionConfig())


class TestMediaSourceContract:
    def test_upload_defaults_are_explicit(self) -> None:
        source = MediaSource(
            kind=SourceKind.UPLOAD, path=Path("/tmp/x.mp4"), title="x", source=UPLOAD_SOURCE
        )

        assert source.duration is None
        assert source.duration_verified is False
        assert source.description is None
        assert source.original_url is None
