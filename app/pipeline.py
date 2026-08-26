"""Stage orchestration and the failure contract (AD-9).


**Ingestion, audio and transcription are fatal.** If we cannot fetch the media,
decode it, or transcribe it, there is no partial result worth returning — the
request produced nothing, and the error is the honest answer.

**Everything after a successful transcription is not.** Once a transcript exists
it is real work, and no later failure may discard it. A failed analysis returns
`200` with the transcript intact, `summary: null`, empty arrays, and an entry in
`errors[]`. That is the brief's central requirement made structural:

    "Le système ne devra pas retourner des informations inventées lorsqu'une
     étape du pipeline échoue."

The analysis stage therefore catches broadly, including exceptions it does not
anticipate. That is normally poor practice; here it is the point. An unhandled
error in summarisation must not turn a successful transcription into a `500`.
The exception is logged in full so a real bug is still visible to us, while the
caller keeps what succeeded.

Silence is treated as an answer, not a failure. Audio with no speech yields
`no_speech` with an empty transcript — correct, and materially different from
inventing a summary of nothing.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from app.analysis.chunking import build_chunks
from app.analysis.llm import StructuredLLM
from app.analysis.map import map_chunks
from app.analysis.reduce import finalise, reduce_analyses
from app.analysis.schemas import Analysis
from app.api.schemas import (
    AnalysisStatus,
    AnalyzeVideoResponse,
    LanguageInfo,
    Provenance,
    TranscriptSegment,
)
from app.audio.ffmpeg import normalize_audio
from app.core.config import Settings
from app.core.errors import (
    ErrorCode,
    ErrorEntry,
    NoSpeechDetectedError,
    PipelineError,
    Stage,
    StageStatus,
)
from app.core.logging import get_logger
from app.core.models import Transcript, TranscriptionResult
from app.ingestion.file_source import load_local_file
from app.ingestion.metadata import MediaSource
from app.ingestion.url_source import fetch_from_url
from app.stt.base import TranscriptionBackend
from app.transcript.segmentation import build_transcript
from app.transcript.validation import is_empty_transcript, validate_segments

logger = get_logger(__name__)


@contextmanager
def temporary_workspace(prefix: str = "video-analysis-") -> Iterator[Path]:
    """A per-request scratch directory, removed however the request ends.

    Downloads and extracted audio can run to tens of megabytes. Cleanup lives in
    `finally` because the failure paths are exactly the ones that would otherwise
    accumulate: an abandoned download after an unreadable-media error leaks just
    as surely as one after a success.
    """
    path = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@dataclass(slots=True)
class MediaRequest:
    """What to analyse: exactly one of a URL or a local file."""

    url: str | None = None
    path: Path | None = None
    filename: str | None = None


async def _ingest(
    request: MediaRequest, settings: Settings, work_dir: Path
) -> MediaSource:
    if request.url is not None:
        return await fetch_from_url(request.url, work_dir, settings.ingestion)
    if request.path is not None:
        return await load_local_file(
            request.path, settings.ingestion, original_filename=request.filename
        )
    raise ValueError("MediaRequest requires either a url or a path")


async def _analyse(
    transcript: Transcript,
    speech: TranscriptionResult,
    llm: StructuredLLM,
    settings: Settings,
) -> Analysis:
    """Chunk, map, reduce."""
    chunks = build_chunks(transcript.segments, settings.analysis)
    outcome = await map_chunks(
        llm, chunks, settings.analysis, language_code=speech.language.code
    )
    reduced = await reduce_analyses(
        llm, outcome.analyses, settings.analysis, language_code=speech.language.code
    )
    return finalise(
        reduced, chunk_count=len(chunks), failed_chunks=outcome.failed_chunks
    )


def _to_segments(transcript: Transcript) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            start=s.start,
            end=s.end,
            speaker=s.speaker,
            text=s.text,
            speaker_confidence=s.speaker_confidence,
        )
        for s in transcript.segments
    ]


async def analyze_media(
    request: MediaRequest,
    settings: Settings,
    backend: TranscriptionBackend,
    llm_factory: Callable[[], StructuredLLM],
    work_dir: Path,
) -> AnalyzeVideoResponse:
    """Run the pipeline and build the response.

    Raises `PipelineError` only for fatal stages. Everything reachable after a
    successful transcription is reported inside a `200` response.

    The language model arrives as a **factory**, not an instance. Constructing it
    requires a key, and a missing key must degrade to `partial_success` with the
    transcript intact — not kill a request whose transcription already succeeded.
    Resolving it eagerly at the transport layer would make that impossible,
    because the failure would happen before this function is entered.
    """
    stages: dict[Stage, StageStatus] = dict.fromkeys(Stage, StageStatus.SKIPPED)
    errors: list[ErrorEntry] = []
    log = logger.bind(source="url" if request.url else "upload")

    # --- Fatal stages -----------------------------------------------------
    media = await _ingest(request, settings, work_dir)
    stages[Stage.INGESTION] = StageStatus.OK

    audio = await normalize_audio(media.path, work_dir / "audio", settings.audio)
    stages[Stage.AUDIO] = StageStatus.OK

    speech = await backend.transcribe(audio)
    stages[Stage.TRANSCRIPTION] = StageStatus.OK

    # --- Segmentation -----------------------------------------------------
    transcript = build_transcript(speech.words, settings.segmentation)
    issues = validate_segments(transcript.segments)
    if issues:
        # Our own construction produced these, so they indicate a bug in
        # segmentation rather than bad input. Logged loudly; the transcript is
        # still returned, because discarding it would hide the defect entirely.
        log.error(
            "segment validation found defects",
            issue_count=len(issues),
            kinds=sorted({i.kind.value for i in issues}),
        )
    stages[Stage.DIARIZATION] = (
        StageStatus.DEGRADED
        if issues or transcript.has_unattributed_speech
        else StageStatus.OK
    )

    provenance = Provenance(
        resolved_model=speech.provenance.resolved_model,
        diarizer_arch=speech.provenance.diarizer_arch,
        transcription_cached=speech.provenance.from_cache,
    )
    language = LanguageInfo(
        code=speech.language.code,
        confidence=speech.language.confidence,
        confidence_is_meaningful=speech.language.confidence_is_meaningful,
    )

    def build(
        status: AnalysisStatus, analysis: Analysis | None, degraded: bool
    ) -> AnalyzeVideoResponse:
        return AnalyzeVideoResponse(
            status=status,
            title=media.title,
            duration=media.duration,
            source=media.source,
            description=media.description,
            language=language,
            number_of_speakers=transcript.number_of_speakers,
            transcript=_to_segments(transcript),
            summary=analysis.summary if analysis else None,
            key_points=analysis.key_points if analysis else [],
            topics=analysis.topics if analysis else [],
            stages={stage.value: status_.value for stage, status_ in stages.items()},
            errors=errors,
            degraded=degraded,
            provenance=provenance,
        )

    # --- No speech --------------------------------------------------------
    if is_empty_transcript(transcript.segments):
        # A correct answer about silent audio. Running analysis on an empty
        # transcript could only produce invented content.
        errors.append(NoSpeechDetectedError().to_entry())
        log.info("no speech detected", duration=media.duration)
        return build(AnalysisStatus.NO_SPEECH, None, degraded=False)

    # --- Analysis (never fatal) -------------------------------------------
    provenance.map_model = settings.analysis.map_model
    provenance.reduce_model = settings.analysis.reduce_model

    try:
        # Constructed here, inside the guard: a missing key is an analysis
        # failure, not a fatal one.
        analysis = await _analyse(transcript, speech, llm_factory(), settings)
    except PipelineError as exc:
        stages[Stage.ANALYSIS] = StageStatus.FAILED
        errors.append(exc.to_entry())
        log.warning("analysis failed; transcript preserved", code=exc.code.value)
        return build(AnalysisStatus.PARTIAL_SUCCESS, None, degraded=True)
    except Exception as exc:  # noqa: BLE001 - see module docstring
        stages[Stage.ANALYSIS] = StageStatus.FAILED
        errors.append(
            ErrorEntry(
                stage=Stage.ANALYSIS,
                code=ErrorCode.ANALYSIS_FAILED,
                message="Transcript analysis failed; the transcript is unaffected.",
                detail={"type": type(exc).__name__},
            )
        )
        log.exception("unexpected error during analysis; transcript preserved")
        return build(AnalysisStatus.PARTIAL_SUCCESS, None, degraded=True)

    partial_coverage = analysis.failed_chunks > 0
    stages[Stage.ANALYSIS] = (
        StageStatus.DEGRADED if partial_coverage else StageStatus.OK
    )
    if partial_coverage:
        log.warning("analysis built from partial coverage", failed=analysis.failed_chunks)

    degraded = (
        partial_coverage
        or transcript.has_unattributed_speech
        or stages[Stage.DIARIZATION] is StageStatus.DEGRADED
    )
    log.info(
        "pipeline complete",
        speakers=transcript.number_of_speakers,
        segments=len(transcript.segments),
        degraded=degraded,
    )
    return build(AnalysisStatus.SUCCESS, analysis, degraded=degraded)


__all__ = ["MediaRequest", "analyze_media", "temporary_workspace"]
