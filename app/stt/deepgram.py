"""Deepgram speech backend (SPEC 5.3, AD-11).

Calls `POST /v1/listen` directly with httpx. Verified against Deepgram's
documentation on 2026-08-25; the response paths this module reads are recorded
in SPEC §3 and should be re-checked before changing them.

**`504` is not retried, and that is the important decision here.** Deepgram's
ten-minute limit is on *processing* time, not audio duration, so a request that
times out will time out again — retrying spends another ten minutes to reach the
same answer while the caller waits. A `504` is the signal that the chunking
fallback (AD-6) is needed, not a transient fault. Retrying it would convert one
slow failure into three.

Two response schemas must both parse (SPEC §3.3). In single-language mode the
channel carries `detected_language` and `language_confidence`. Under
`language=multi` each word instead carries its own `language` and the alternative
gains a `languages` array. A parser that assumed one shape would fail silently on
the other, dropping language information rather than erroring.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx

from app.audio.ffmpeg import NormalizedAudio
from app.core.concurrency import backoff_delays
from app.core.config import DeepgramConfig
from app.core.errors import ConfigurationError, MediaTooLargeError, STTFailedError
from app.core.logging import get_logger
from app.core.models import (
    DetectedLanguage,
    TranscriptionProvenance,
    TranscriptionResult,
    Word,
)
from app.stt.cache import ResponseCache, build_cache_key

logger = get_logger(__name__)

BACKEND_NAME = "deepgram"
LISTEN_URL = "https://api.deepgram.com/v1/listen"

#: Processing-time timeout. Not retryable — see the module docstring — and the
#: trigger for the AD-6 chunking fallback.
PROCESSING_TIMEOUT_STATUS = 504

#: Transient upstream conditions worth another attempt. `504` is deliberately
#: absent; `429` is present because rate limits clear with time.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503})

#: The 35 languages Deepgram's detector covers. Outside this set the reported
#: `language_confidence` is not meaningful (SPEC §3.2), so results are flagged
#: rather than left to be thresholded naively.
DETECTABLE_LANGUAGES = frozenset(
    {
        "bg", "ca", "cs", "da", "de", "de-CH", "el", "en", "es", "et", "fi",
        "fr", "hi", "hu", "id", "it", "ja", "ko", "lt", "lv", "ms", "nl",
        "nl-BE", "no", "pl", "pt", "ro", "ru", "sk", "sv", "th", "tr", "uk",
        "vi", "zh",
    }
)

_CONTENT_TYPES = {
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
}


def content_type_for(path: Path) -> str:
    """Map an audio file to its request Content-Type."""
    return _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


class DeepgramBackend:
    """Speech-to-text and diarization via Deepgram's pre-recorded API.

    Satisfies the `TranscriptionBackend` Protocol structurally.
    """

    def __init__(
        self,
        api_key: str,
        config: DeepgramConfig,
        cache: ResponseCache | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._config = config
        self._cache = cache
        # Injectable so tests can drive the full client — retries, status
        # handling, parsing — through httpx.MockTransport with no network.
        self._client = client
        self._owns_client = client is None

    @property
    def name(self) -> str:
        return BACKEND_NAME

    async def transcribe(self, audio: NormalizedAudio) -> TranscriptionResult:
        """Transcribe normalised audio, using the cache when available."""
        params = self._config.to_query_params()
        key = build_cache_key(audio.sha256, params)
        log = logger.bind(cache_key=key, audio_sha256=audio.sha256[:12])

        if self._cache is not None:
            cached = await self._cache.get(key)
            if cached is not None:
                log.info("transcription served from cache")
                return parse_response(cached, from_cache=True)

        payload = await self._request(audio, params, log)

        if self._cache is not None:
            await self._cache.put(key, payload)

        result = parse_response(payload, from_cache=False)
        log.info(
            "transcription complete",
            words=len(result.words),
            speakers=len(result.speaker_ids),
            resolved_model=result.provenance.resolved_model,
            detected_language=result.language.code,
        )
        return result

    async def _request(
        self, audio: NormalizedAudio, params: dict[str, Any], log: Any
    ) -> dict[str, Any]:
        """POST the audio, retrying only genuinely transient failures."""
        # Read once and reuse across attempts: a streamed file handle would be
        # consumed by the first attempt and arrive empty on a retry.
        body = await asyncio.to_thread(audio.path.read_bytes)

        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": content_type_for(audio.path),
        }
        client = self._ensure_client()
        attempts = max(1, self._config.max_retries)
        delays = list(backoff_delays(attempts, self._config.backoff_base_sec))
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                response = await client.post(
                    LISTEN_URL, params=params, headers=headers, content=body
                )
            except httpx.TimeoutException as exc:
                last_error = exc
                log.warning("request timed out", attempt=attempt + 1)
            except httpx.HTTPError as exc:
                last_error = exc
                log.warning("transport error", attempt=attempt + 1, error=str(exc))
            else:
                if response.status_code == httpx.codes.OK:
                    return self._decode(response)

                error = self._classify(response)
                if response.status_code not in RETRYABLE_STATUSES:
                    raise error

                last_error = error
                log.warning(
                    "retryable upstream status",
                    attempt=attempt + 1,
                    status=response.status_code,
                )

            if attempt < len(delays):
                await asyncio.sleep(delays[attempt])

        raise STTFailedError(
            "Speech-to-text failed after retries.",
            detail={"attempts": attempts},
            cause=last_error,
        )

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._config.timeout_sec, connect=30.0)
            )
        return self._client

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise STTFailedError(
                "The transcription service returned a response that is not JSON.",
                cause=exc,
            ) from exc
        if not isinstance(payload, dict):
            raise STTFailedError("The transcription service returned an unexpected payload.")
        return payload

    @staticmethod
    def _classify(response: httpx.Response) -> Exception:
        """Map an HTTP status onto the taxonomy."""
        status = response.status_code
        body = response.text[:300]

        if status in (401, 403):
            # Our misconfiguration, not the caller's request. The message stays
            # generic so an API response never hints at credential contents.
            return ConfigurationError(
                "The transcription service rejected our credentials.",
                detail={"status": status},
            )

        if status == 413:
            return MediaTooLargeError(
                "The audio exceeds the transcription service's size limit.",
                detail={"status": status},
            )

        if status == PROCESSING_TIMEOUT_STATUS:
            return STTFailedError(
                "The transcription service exceeded its processing time limit.",
                detail={
                    "status": status,
                    # Consumed by the AD-6 fallback: this is a signal to split
                    # the audio, not a fault to retry.
                    "chunking_fallback_applicable": True,
                },
            )

        if status == 400:
            # Includes rejected parameter combinations, e.g. `diarize` sent
            # alongside `diarize_model`. The body is retained because it names
            # the offending parameter.
            return STTFailedError(
                "The transcription service rejected the request.",
                detail={"status": status, "response": body},
            )

        return STTFailedError(
            "The transcription service returned an error.",
            detail={"status": status, "response": body},
        )

    async def aclose(self) -> None:
        """Close the HTTP client if this instance created it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> DeepgramBackend:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_words(raw_words: list[dict[str, Any]]) -> list[Word]:
    """Convert provider words into domain words.

    `punctuated_word` is preferred when present: with `smart_format=true` it
    carries the casing and punctuation that make a transcript readable, while
    `word` is the bare token.

    Malformed entries are skipped rather than defaulted. A word without timing
    cannot be placed in a segment, and inventing `start=0.0` would put text at a
    moment where it was never spoken.
    """
    words: list[Word] = []
    for raw in raw_words:
        text = raw.get("punctuated_word") or raw.get("word")
        start = raw.get("start")
        end = raw.get("end")
        if not text or not isinstance(start, int | float) or not isinstance(end, int | float):
            continue

        words.append(
            Word(
                text=str(text),
                start=float(start),
                end=float(end),
                confidence=_as_float(raw.get("confidence")),
                speaker=_as_int(raw.get("speaker")),
                speaker_confidence=_as_float(raw.get("speaker_confidence")),
                language=raw.get("language"),
            )
        )
    return words


def _parse_language(channel: dict[str, Any], alternative: dict[str, Any]) -> DetectedLanguage:
    """Resolve language across both response schemas (SPEC §3.3).

    Single-language mode reports on the channel. Code-switching mode reports a
    `languages` array on the alternative instead, with no single confidence —
    which is honest, since there is no single dominant language to be confident
    about.
    """
    languages = [str(code) for code in (alternative.get("languages") or [])]
    code = channel.get("detected_language") or (languages[0] if languages else None)
    confidence = _as_float(channel.get("language_confidence"))

    return DetectedLanguage(
        code=code,
        confidence=confidence,
        # Outside the detector's supported set the score is not interpretable.
        confidence_is_meaningful=code in DETECTABLE_LANGUAGES if code else False,
        languages=languages,
    )


def parse_response(payload: dict[str, Any], *, from_cache: bool) -> TranscriptionResult:
    """Convert a raw Deepgram response into the domain contract.

    A response with no channels, no alternatives, or no words yields an **empty**
    result rather than an error. That is the correct outcome for silence; the
    pipeline decides whether it means `NO_SPEECH_DETECTED`.

    A response missing `results` entirely is different — that is a malformed
    payload, and treating it as "no speech" would report silence for what was
    actually a failure.
    """
    if "results" not in payload:
        raise STTFailedError(
            "The transcription response is missing its results section.",
            detail={"keys": sorted(payload)[:10]},
        )

    metadata = payload.get("metadata") or {}
    results = payload.get("results") or {}
    channels = results.get("channels") or []
    channel: dict[str, Any] = channels[0] if channels else {}
    alternatives = channel.get("alternatives") or []
    alternative: dict[str, Any] = alternatives[0] if alternatives else {}

    utterances = results.get("utterances")
    diarize_info = metadata.get("diarize_info") or {}

    provenance = TranscriptionProvenance(
        backend=BACKEND_NAME,
        request_id=metadata.get("request_id"),
        resolved_model=_resolve_model_name(metadata),
        diarizer_arch=diarize_info.get("arch"),
        diarizer_model_uuid=diarize_info.get("model_uuid"),
        from_cache=from_cache,
    )

    return TranscriptionResult(
        words=_parse_words(alternative.get("words") or []),
        language=_parse_language(channel, alternative),
        provenance=provenance,
        audio_duration=_as_float(metadata.get("duration")),
        utterance_count=len(utterances) if isinstance(utterances, list) else None,
    )


def _resolve_model_name(metadata: dict[str, Any]) -> str | None:
    """Extract the model that actually ran.

    Recorded because Deepgram silently downgrades along
    Nova-3 → Nova-2 → Nova-1 → Enhanced → Base when a detected language is not
    available on the requested model. Without this, a result produced by a
    different model than we asked for looks identical to one that was not.
    """
    model_info = metadata.get("model_info") or {}
    if isinstance(model_info, dict):
        for entry in model_info.values():
            if isinstance(entry, dict) and entry.get("name"):
                return str(entry["name"])

    models = metadata.get("models")
    if isinstance(models, list) and models:
        return str(models[0])
    return None


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


__all__ = [
    "BACKEND_NAME",
    "DETECTABLE_LANGUAGES",
    "LISTEN_URL",
    "PROCESSING_TIMEOUT_STATUS",
    "RETRYABLE_STATUSES",
    "DeepgramBackend",
    "content_type_for",
    "parse_response",
]
