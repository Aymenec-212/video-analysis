"""HTTP surface: `POST /analyze-video` and `GET /health`.

The brief specifies one endpoint accepting *either* a JSON body with a URL *or* a
video file sent directly. That is two content types on one route, which FastAPI's
declarative parameters cannot express, so the body is dispatched on
`Content-Type` explicitly. The alternative — two endpoints — would satisfy the
framework and not the brief.

This layer stays thin on purpose. It decodes the request, hands a `MediaRequest`
to the pipeline, and serialises the result. Every decision about what constitutes
failure lives in `app/pipeline.py`, so the failure contract is testable without
an HTTP client.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

from app.api.dependencies import Dependencies, get_dependencies
from app.api.schemas import (
    AnalyzeUrlRequest,
    AnalyzeVideoResponse,
    ErrorResponse,
    HealthResponse,
)
from app.core.config import get_settings
from app.core.errors import (
    InvalidRequestError,
    PipelineError,
    UnsupportedContentTypeError,
)
from app.core.logging import configure_logging, get_logger
from app.ingestion.file_source import save_upload
from app.pipeline import MediaRequest, analyze_media, temporary_workspace

logger = get_logger(__name__)

API_VERSION = "0.1.0"

#: Form field carrying the upload. `video` is accepted as a courtesy, since it is
#: the obvious guess and rejecting it would be a pointless papercut.
UPLOAD_FIELDS = ("file", "video")

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness and configuration visibility.

    Deliberately reachable with no credentials configured, so a fresh clone can
    be started and inspected before any key is set.
    """
    settings = get_dependencies(request).settings
    return HealthResponse(
        version=API_VERSION,
        deepgram_configured=settings.deepgram_api_key is not None,
        openai_configured=settings.openai_api_key is not None,
    )


@router.post(
    "/analyze-video",
    response_model=AnalyzeVideoResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
async def analyze_video(request: Request) -> AnalyzeVideoResponse:
    """Analyse a video supplied as a URL or as an uploaded file."""
    dependencies = get_dependencies(request)
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()

    with temporary_workspace() as work_dir:
        if content_type == "application/json":
            media_request = await _read_url_request(request)
        elif content_type == "multipart/form-data":
            media_request = await _read_upload_request(request, dependencies, work_dir)
        else:
            raise UnsupportedContentTypeError(detail={"received": content_type or "none"})

        return await analyze_media(
            media_request,
            dependencies.settings,
            dependencies.get_backend(),
            # Passed unresolved: a missing OpenAI key must degrade to
            # partial_success with the transcript intact, not fail the request.
            dependencies.get_llm,
            work_dir,
        )


async def _read_url_request(request: Request) -> MediaRequest:
    try:
        payload = await request.json()
    except ValueError as exc:
        raise InvalidRequestError("The request body is not valid JSON.", cause=exc) from exc

    if not isinstance(payload, dict) or not payload.get("url"):
        raise InvalidRequestError("Provide a non-empty `url` field.")

    return MediaRequest(url=AnalyzeUrlRequest.model_validate(payload).url)


async def _read_upload_request(
    request: Request, dependencies: Dependencies, work_dir: Path
) -> MediaRequest:
    form = await request.form()
    upload = next(
        (
            value
            for field in UPLOAD_FIELDS
            if isinstance(value := form.get(field), UploadFile)
        ),
        None,
    )
    if upload is None:
        raise InvalidRequestError(
            "Attach the video as a `file` part in the multipart body.",
            detail={"accepted_fields": list(UPLOAD_FIELDS)},
        )

    # Streamed to disk with the size cap enforced during the write, so a large
    # upload cannot be buffered into memory to be measured.
    destination = await save_upload(
        upload,
        work_dir / "upload.bin",
        max_bytes=dependencies.settings.ingestion.max_bytes,
    )
    return MediaRequest(path=destination, filename=upload.filename)


async def pipeline_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Render a fatal `PipelineError` in the shared error shape.

    Only fatal errors reach here. Non-fatal ones are reported inside a `200`
    response by the pipeline, which is the point of AD-9.
    """
    if not isinstance(exc, PipelineError):  # pragma: no cover - registered per type
        raise exc
    logger.warning(
        "request failed", code=exc.code.value, stage=exc.stage.value, path=request.url.path
    )
    return JSONResponse(
        status_code=exc.http_status,
        content=ErrorResponse(errors=[exc.to_entry()]).model_dump(mode="json"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging, share dependencies, release clients on shutdown."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    if not hasattr(app.state, "dependencies"):
        app.state.dependencies = Dependencies(settings=settings)

    logger.info(
        "service started",
        version=API_VERSION,
        deepgram_configured=settings.deepgram_api_key is not None,
        openai_configured=settings.openai_api_key is not None,
    )
    try:
        yield
    finally:
        await app.state.dependencies.aclose()


def create_app() -> FastAPI:
    """Build the application.

    A factory rather than a module-level singleton so tests can construct an
    isolated instance with injected dependencies.
    """
    application = FastAPI(
        title="AI Video Transcription & Analysis",
        version=API_VERSION,
        lifespan=lifespan,
    )
    application.include_router(router)
    application.add_exception_handler(PipelineError, pipeline_error_handler)
    return application


app = create_app()

__all__ = ["API_VERSION", "app", "create_app", "router"]
