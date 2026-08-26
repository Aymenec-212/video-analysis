"""API tests (SPEC §6.2, §6.3, AD-9).

Driven through `httpx.ASGITransport`, so real routing, real content-type
dispatch, real serialisation — no network. Dependencies are injected on
`app.state`, which is why the lifespan only creates them when absent.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.api.dependencies import Dependencies
from app.api.routes import create_app
from app.core.config import AnalysisConfig, IngestionConfig, Settings
from app.core.errors import (
    AnalysisFailedError,
    SourceUnavailableError,
    STTFailedError,
)
from tests.fixtures.backend import FakeBackend, transcription_result
from tests.fixtures.llm import FakeLLM, chunk_analysis, reduced_analysis
from tests.fixtures.media import ffmpeg_required


def build_client(
    backend: Any = None,
    llm: Any = None,
    settings: Settings | None = None,
) -> httpx.AsyncClient:
    """An ASGI client with dependencies injected."""
    app = create_app()
    app.state.dependencies = Dependencies(
        settings=settings
        or Settings(analysis=AnalysisConfig(max_retries=1, backoff_base_sec=0.001)),
        backend=backend if backend is not None else FakeBackend(),
        llm=llm if llm is not None else FakeLLM(),
    )
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


def working_llm() -> FakeLLM:
    def handler(instructions: str, prompt: str, schema: type) -> Any:
        from app.analysis.schemas import ChunkAnalysis

        # A short transcript is a single chunk, and reduce_analyses returns a
        # lone analysis directly rather than re-summarising a summary. Both
        # branches therefore produce the same content, so assertions hold
        # whichever path a given fixture takes.
        if schema is ChunkAnalysis:
            return chunk_analysis(
                "Cette vidéo présente une interview.",
                [("Présentation du projet", 0.5)],
                ["Interview", "Artificial Intelligence"],
            )
        return reduced_analysis(
            "Cette vidéo présente une interview.",
            [("Présentation du projet", 0.5)],
            ["Interview", "Artificial Intelligence"],
        )

    return FakeLLM(handler=handler)


class TestHealth:
    async def test_reports_ok(self) -> None:
        async with build_client() as client:
            response = await client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    async def test_reachable_without_any_credentials(self) -> None:
        """A fresh clone must start and be inspectable before keys are set."""
        async with build_client(settings=Settings()) as client:
            body = (await client.get("/health")).json()

        assert body["deepgram_configured"] is False
        assert body["openai_configured"] is False

    async def test_reports_configuration_without_exposing_it(self) -> None:
        settings = Settings(deepgram_api_key="dg-secret", openai_api_key="sk-secret")
        async with build_client(settings=settings) as client:
            response = await client.get("/health")

        assert response.json()["deepgram_configured"] is True
        assert "dg-secret" not in response.text
        assert "sk-secret" not in response.text


class TestUrlSubmission:
    async def test_json_body_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_ingestion(monkeypatch)

        async with build_client(llm=working_llm()) as client:
            response = await client.post(
                "/analyze-video", json={"url": "https://example.com/video"}
            )

        assert response.status_code == 200
        assert response.json()["status"] == "success"

    async def test_missing_url_is_rejected(self) -> None:
        async with build_client() as client:
            response = await client.post("/analyze-video", json={})

        assert response.status_code == 400
        assert response.json()["errors"][0]["code"] == "INVALID_REQUEST"

    async def test_malformed_json_is_rejected(self) -> None:
        async with build_client() as client:
            response = await client.post(
                "/analyze-video",
                content=b"{not json",
                headers={"content-type": "application/json"},
            )

        assert response.status_code == 400

    async def test_invalid_scheme_is_rejected(self) -> None:
        async with build_client() as client:
            response = await client.post(
                "/analyze-video", json={"url": "file:///etc/passwd"}
            )

        assert response.status_code == 400
        assert response.json()["errors"][0]["code"] == "INVALID_URL"


@ffmpeg_required
class TestFileUpload:
    async def test_multipart_upload_is_accepted(
        self, media_corpus: dict[str, Path]
    ) -> None:
        async with build_client(llm=working_llm()) as client:
            with media_corpus["with_audio"].open("rb") as handle:
                response = await client.post(
                    "/analyze-video", files={"file": ("interview.mp4", handle, "video/mp4")}
                )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["source"] == "upload"
        assert body["title"] == "interview"

    async def test_video_field_name_is_also_accepted(
        self, media_corpus: dict[str, Path]
    ) -> None:
        async with build_client(llm=working_llm()) as client:
            with media_corpus["with_audio"].open("rb") as handle:
                response = await client.post(
                    "/analyze-video", files={"video": ("clip.mp4", handle, "video/mp4")}
                )

        assert response.status_code == 200

    async def test_missing_file_part_is_rejected(self) -> None:
        """A multipart body carrying only plain fields, no file."""
        async with build_client() as client:
            response = await client.post(
                "/analyze-video", files={"note": (None, "no file here")}
            )

        assert response.status_code == 400
        assert response.json()["errors"][0]["code"] == "INVALID_REQUEST"

    async def test_non_media_upload_is_rejected_by_probe(
        self, media_corpus: dict[str, Path]
    ) -> None:
        """A text file named .mp4 — extension proves nothing."""
        async with build_client() as client:
            with media_corpus["fake"].open("rb") as handle:
                response = await client.post(
                    "/analyze-video", files={"file": ("movie.mp4", handle, "video/mp4")}
                )

        assert response.status_code == 422
        assert response.json()["errors"][0]["code"] == "UNREADABLE_MEDIA"

    async def test_audioless_upload_is_rejected(
        self, media_corpus: dict[str, Path]
    ) -> None:
        async with build_client() as client:
            with media_corpus["no_audio"].open("rb") as handle:
                response = await client.post(
                    "/analyze-video", files={"file": ("silent.mp4", handle, "video/mp4")}
                )

        assert response.status_code == 422
        assert response.json()["errors"][0]["code"] == "NO_AUDIO_STREAM"

    async def test_oversize_upload_is_rejected(
        self, media_corpus: dict[str, Path]
    ) -> None:
        settings = Settings(ingestion=IngestionConfig(max_bytes=32))
        async with build_client(settings=settings) as client:
            with media_corpus["with_audio"].open("rb") as handle:
                response = await client.post(
                    "/analyze-video", files={"file": ("big.mp4", handle, "video/mp4")}
                )

        assert response.status_code == 413


class TestContentTypeDispatch:
    async def test_unsupported_content_type_is_rejected(self) -> None:
        async with build_client() as client:
            response = await client.post(
                "/analyze-video",
                content=b"raw bytes",
                headers={"content-type": "text/plain"},
            )

        assert response.status_code == 415
        assert response.json()["errors"][0]["code"] == "UNSUPPORTED_CONTENT_TYPE"

    async def test_json_charset_suffix_is_tolerated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`application/json; charset=utf-8` is the same content type."""
        _stub_ingestion(monkeypatch)

        async with build_client(llm=working_llm()) as client:
            response = await client.post(
                "/analyze-video",
                content=b'{"url": "https://example.com/v"}',
                headers={"content-type": "application/json; charset=utf-8"},
            )

        assert response.status_code == 200


@ffmpeg_required
class TestSuccessContract:
    """The response shape the brief illustrates (SPEC §6.2)."""

    async def test_contains_every_required_field(
        self, media_corpus: dict[str, Path]
    ) -> None:
        body = await _upload(media_corpus["with_audio"], llm=working_llm())

        for field in (
            "status", "title", "duration", "source", "number_of_speakers",
            "transcript", "summary", "key_points", "topics",
            "stages", "errors", "degraded", "provenance",
        ):
            assert field in body, field

    async def test_transcript_segments_match_the_brief(
        self, media_corpus: dict[str, Path]
    ) -> None:
        body = await _upload(media_corpus["with_audio"], llm=working_llm())
        segment = body["transcript"][0]

        assert segment["start"] == 0.5
        assert segment["end"] == 2.0
        assert segment["speaker"] == "SPEAKER_01"
        assert segment["text"] == "Bonjour et bienvenue."

    async def test_speakers_are_counted_not_assumed(
        self, media_corpus: dict[str, Path]
    ) -> None:
        body = await _upload(media_corpus["with_audio"], llm=working_llm())

        assert body["number_of_speakers"] == 2
        assert {s["speaker"] for s in body["transcript"]} == {"SPEAKER_01", "SPEAKER_02"}

    async def test_analysis_fields_are_populated(
        self, media_corpus: dict[str, Path]
    ) -> None:
        body = await _upload(media_corpus["with_audio"], llm=working_llm())

        assert body["summary"] == "Cette vidéo présente une interview."
        assert body["key_points"] == ["Présentation du projet"]
        assert "Interview" in body["topics"]

    async def test_language_carries_the_confidence_caveat(
        self, media_corpus: dict[str, Path]
    ) -> None:
        body = await _upload(media_corpus["with_audio"], llm=working_llm())

        assert body["language"]["code"] == "fr"
        assert body["language"]["confidence_is_meaningful"] is True

    async def test_every_stage_reports_its_outcome(
        self, media_corpus: dict[str, Path]
    ) -> None:
        body = await _upload(media_corpus["with_audio"], llm=working_llm())

        assert body["stages"] == {
            "ingestion": "ok", "audio": "ok", "transcription": "ok",
            "diarization": "ok", "analysis": "ok",
        }
        assert body["errors"] == []
        assert body["degraded"] is False

    async def test_provenance_records_what_ran(
        self, media_corpus: dict[str, Path]
    ) -> None:
        body = await _upload(media_corpus["with_audio"], llm=working_llm())

        assert body["provenance"]["resolved_model"] == "nova-3-general"
        assert body["provenance"]["diarizer_arch"] == "v2"

    async def test_no_filesystem_paths_leak(
        self, media_corpus: dict[str, Path]
    ) -> None:
        """Internal paths are not the caller's business."""
        async with build_client(llm=working_llm()) as client:
            with media_corpus["with_audio"].open("rb") as handle:
                response = await client.post(
                    "/analyze-video", files={"file": ("x.mp4", handle, "video/mp4")}
                )

        assert "/tmp" not in response.text
        assert "video-analysis-" not in response.text


@ffmpeg_required
class TestNoSpeechPath:
    """Silence is an answer, not a failure (SPEC §6.3)."""

    async def test_returns_200_with_an_empty_transcript(
        self, media_corpus: dict[str, Path]
    ) -> None:
        backend = FakeBackend(transcription_result(words=[], language=None))
        body = await _upload(media_corpus["silent"], backend=backend, expect=200)

        assert body["status"] == "no_speech"
        assert body["transcript"] == []
        assert body["number_of_speakers"] == 0

    async def test_invents_nothing(self, media_corpus: dict[str, Path]) -> None:
        """The whole point: no summary of audio that contained no speech."""
        backend = FakeBackend(transcription_result(words=[], language=None))
        body = await _upload(media_corpus["silent"], backend=backend, expect=200)

        assert body["summary"] is None
        assert body["key_points"] == []
        assert body["topics"] == []

    async def test_reports_the_reason(self, media_corpus: dict[str, Path]) -> None:
        backend = FakeBackend(transcription_result(words=[], language=None))
        body = await _upload(media_corpus["silent"], backend=backend, expect=200)

        assert body["errors"][0]["code"] == "NO_SPEECH_DETECTED"
        assert body["stages"]["analysis"] == "skipped"

    async def test_analysis_is_never_invoked(
        self, media_corpus: dict[str, Path]
    ) -> None:
        backend = FakeBackend(transcription_result(words=[], language=None))
        llm = FakeLLM()
        await _upload(media_corpus["silent"], backend=backend, llm=llm, expect=200)

        assert llm.call_count == 0


@ffmpeg_required
class TestPartialSuccessPath:
    """AD-9: a failed analysis never destroys a successful transcription."""

    async def test_returns_200_with_the_transcript_intact(
        self, media_corpus: dict[str, Path]
    ) -> None:
        llm = FakeLLM([AnalysisFailedError("model unavailable")])
        body = await _upload(media_corpus["with_audio"], llm=llm, expect=200)

        assert body["status"] == "partial_success"
        assert len(body["transcript"]) == 2
        assert body["number_of_speakers"] == 2

    async def test_analysis_fields_are_null_not_invented(
        self, media_corpus: dict[str, Path]
    ) -> None:
        llm = FakeLLM([AnalysisFailedError("model unavailable")])
        body = await _upload(media_corpus["with_audio"], llm=llm, expect=200)

        assert body["summary"] is None
        assert body["key_points"] == []
        assert body["topics"] == []

    async def test_summary_key_is_present_and_explicitly_null(
        self, media_corpus: dict[str, Path]
    ) -> None:
        """Omitting the key would leave a caller unable to tell null from absent."""
        llm = FakeLLM([AnalysisFailedError("model unavailable")])
        body = await _upload(media_corpus["with_audio"], llm=llm, expect=200)

        assert "summary" in body
        assert body["summary"] is None

    async def test_failure_is_reported_not_hidden(
        self, media_corpus: dict[str, Path]
    ) -> None:
        llm = FakeLLM([AnalysisFailedError("model unavailable")])
        body = await _upload(media_corpus["with_audio"], llm=llm, expect=200)

        assert body["errors"][0]["code"] == "ANALYSIS_FAILED"
        assert body["stages"]["analysis"] == "failed"
        assert body["stages"]["transcription"] == "ok"
        assert body["degraded"] is True

    async def test_unexpected_errors_also_degrade(
        self, media_corpus: dict[str, Path]
    ) -> None:
        """An unanticipated bug in analysis must not become a 500."""
        llm = FakeLLM([RuntimeError("something nobody predicted")])
        body = await _upload(media_corpus["with_audio"], llm=llm, expect=200)

        assert body["status"] == "partial_success"
        assert len(body["transcript"]) == 2

    async def test_missing_openai_key_degrades_rather_than_fails(
        self, media_corpus: dict[str, Path]
    ) -> None:
        """The bug wiring found: resolving the client eagerly made this a 500."""
        app = create_app()
        app.state.dependencies = Dependencies(
            settings=Settings(openai_api_key=None), backend=FakeBackend(), llm=None
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            with media_corpus["with_audio"].open("rb") as handle:
                response = await client.post(
                    "/analyze-video", files={"file": ("x.mp4", handle, "video/mp4")}
                )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "partial_success"
        assert len(body["transcript"]) == 2
        assert body["summary"] is None


@ffmpeg_required
class TestFatalFailures:
    """Stages before a transcript exists have no partial result to preserve."""

    async def test_transcription_failure_returns_502(
        self, media_corpus: dict[str, Path]
    ) -> None:
        backend = FakeBackend(error=STTFailedError("upstream down"))
        async with build_client(backend=backend) as client:
            with media_corpus["with_audio"].open("rb") as handle:
                response = await client.post(
                    "/analyze-video", files={"file": ("x.mp4", handle, "video/mp4")}
                )

        assert response.status_code == 502
        assert response.json()["errors"][0]["code"] == "STT_FAILED"

    async def test_unavailable_source_returns_422(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def failing(*args: object, **kwargs: object) -> None:
            raise SourceUnavailableError("video is private")

        monkeypatch.setattr("app.pipeline.fetch_from_url", failing)

        async with build_client() as client:
            response = await client.post(
                "/analyze-video", json={"url": "https://example.com/private"}
            )

        assert response.status_code == 422
        assert response.json()["errors"][0]["code"] == "SOURCE_UNAVAILABLE"

    async def test_missing_deepgram_key_is_fatal(
        self, media_corpus: dict[str, Path]
    ) -> None:
        """Without transcription there is no partial result worth returning."""
        app = create_app()
        app.state.dependencies = Dependencies(
            settings=Settings(deepgram_api_key=None), backend=None, llm=FakeLLM()
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            with media_corpus["with_audio"].open("rb") as handle:
                response = await client.post(
                    "/analyze-video", files={"file": ("x.mp4", handle, "video/mp4")}
                )

        assert response.status_code == 500
        assert response.json()["errors"][0]["code"] == "CONFIGURATION_ERROR"

    async def test_error_bodies_share_one_shape(self) -> None:
        """A caller parses `errors[]` regardless of outcome."""
        async with build_client() as client:
            response = await client.post("/analyze-video", json={"url": "not-a-url"})

        body = response.json()
        assert body["status"] == "error"
        assert {"stage", "code", "message", "detail"} <= set(body["errors"][0])


@ffmpeg_required
class TestConcurrentRequests:
    async def test_concurrent_uploads_do_not_interfere(
        self, media_corpus: dict[str, Path]
    ) -> None:
        """Each request owns its workspace; results must not cross."""
        async with build_client(llm=working_llm()) as client:

            async def submit(name: str) -> dict[str, Any]:
                content = media_corpus["with_audio"].read_bytes()
                response = await client.post(
                    "/analyze-video",
                    files={"file": (f"{name}.mp4", content, "video/mp4")},
                )
                assert response.status_code == 200
                return dict(response.json())

            bodies = await asyncio.gather(*(submit(f"video{i}") for i in range(5)))

        assert [b["title"] for b in bodies] == [f"video{i}" for i in range(5)]
        assert all(b["status"] == "success" for b in bodies)
        assert all(len(b["transcript"]) == 2 for b in bodies)

    async def test_workspaces_are_removed_after_each_request(
        self, media_corpus: dict[str, Path]
    ) -> None:
        """Failure paths leak scratch space just as readily as success paths."""
        import tempfile

        root = Path(tempfile.gettempdir())
        before = set(root.glob("video-analysis-*"))

        async with build_client(llm=working_llm()) as client:
            content = media_corpus["with_audio"].read_bytes()
            await client.post(
                "/analyze-video", files={"file": ("a.mp4", content, "video/mp4")}
            )
            with media_corpus["fake"].open("rb") as handle:  # fatal path
                await client.post(
                    "/analyze-video", files={"file": ("b.mp4", handle, "video/mp4")}
                )

        assert set(root.glob("video-analysis-*")) == before


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_ingestion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace URL fetching with a locally generated media file."""
    import subprocess

    from app.ingestion.metadata import MediaSource, SourceKind

    async def fake_fetch(url: str, work_dir: Path, config: object) -> MediaSource:
        work_dir.mkdir(parents=True, exist_ok=True)
        path = work_dir / "downloaded.mp4"
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                "-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=10",
                "-map", "1:v", "-map", "0:a", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
            ],
            check=True,
            capture_output=True,
        )
        return MediaSource(
            kind=SourceKind.URL,
            path=path,
            title="Interview AI",
            duration=2.0,
            duration_verified=True,
            description="A test video.",
            source="Youtube",
            size_bytes=path.stat().st_size,
            original_url=url,
        )

    monkeypatch.setattr("app.pipeline.fetch_from_url", fake_fetch)


async def _upload(
    path: Path,
    *,
    backend: Any = None,
    llm: Any = None,
    expect: int = 200,
) -> dict[str, Any]:
    async with build_client(backend=backend, llm=llm) as client:
        with path.open("rb") as handle:
            response = await client.post(
                "/analyze-video", files={"file": (path.name, handle, "video/mp4")}
            )

    assert response.status_code == expect, response.text
    return dict(response.json())
