"""Application configuration (SPEC 5.3).

Two Deepgram parameter combinations are rejected by the API and would otherwise
surface as opaque runtime failures:

1. `diarize` together with `diarize_model` — the request is rejected outright.
2. `detect_language` together with `language` — detection silently overrides the
   fixed language, so sending both means the request does not do what it says.

`DeepgramConfig` handles these differently on purpose. The second is made
*unrepresentable*: `language_mode` selects exactly one language strategy, and
`to_query_params` emits only that strategy's parameters, so no configuration can
produce a conflicting request. The first cannot be designed away, because the
danger lives in the `extra_params` escape hatch — a future maintainer adding
`diarize=true` there without knowing it is deprecated. That case is guarded
explicitly against a reserved-key list.

Verified against Deepgram documentation on 2026-08-25 (SPEC 3).
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.errors import ConfigurationError

# ---------------------------------------------------------------------------
# Speech-to-text
# ---------------------------------------------------------------------------


class LanguageMode(StrEnum):
    """Which language strategy to send to Deepgram.

    Exactly one applies per request. Modelling this as a mode rather than three
    independent flags is what makes the `detect_language` + `language` conflict
    impossible to express.
    """

    #: `detect_language`, optionally restricted to a candidate set.
    DETECT = "detect"
    #: `language=<code>`, a fixed known language.
    FIXED = "fixed"
    #: `language=multi`, Nova-3 code-switching. No Arabic — see SPEC §3.3.
    MULTI = "multi"


#: Query parameters `extra_params` may never set. `diarize` is deprecated and is
#: rejected by Deepgram when combined with `diarize_model`; the rest are owned by
#: dedicated fields and would let the escape hatch contradict validated config.
RESERVED_QUERY_PARAMS = frozenset(
    {"diarize", "diarize_model", "language", "detect_language", "model"}
)


class DeepgramConfig(BaseModel):
    """Transcription request configuration.

    Defaults encode the decisions in SPEC §5.3: Nova-3, the current GA diarizer,
    and language detection restricted to the two languages our test set uses.
    """

    model: str = "nova-3-general"

    #: `diarize_model` both enables diarization and pins the version, so
    #: `diarize=true` is neither needed nor permitted alongside it (SPEC §3.1).
    #: `latest` currently resolves to the v2 batch diarizer.
    diarize_model: Literal["latest", "v1", "v2"] = "latest"

    smart_format: bool = True
    punctuate: bool = True
   
    utterances: bool = True

    language_mode: LanguageMode = LanguageMode.DETECT

    #: Candidate set for DETECT mode. Restricting detection removes a whole class
    #: of misdetection (SPEC 3.2). Empty means unrestricted across all 35
    #: supported languages.
    detect_candidates: tuple[str, ...] = ("en", "fr")

    #: Language code for FIXED mode. Ignored in every other mode.
    language: str = "fr"

    #: Escape hatch for parameters without a dedicated field. Guarded against
    #: RESERVED_QUERY_PARAMS.
    extra_params: dict[str, str] = Field(default_factory=dict)

    #: Deepgram returns 504 past 10 minutes of *processing* time.
    #  Our client timeout sits just above it.
    timeout_sec: float = 660.0
    max_retries: int = 3
    backoff_base_sec: float = 1.0

    @field_validator("detect_candidates")
    @classmethod
    def _normalise_candidates(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Lowercase, strip, and de-duplicate while preserving order."""
        seen: list[str] = []
        for raw in value:
            code = raw.strip().lower()
            if not code:
                continue
            if code not in seen:
                seen.append(code)
        return tuple(seen)

    @model_validator(mode="after")
    def _guard_reserved_params(self) -> DeepgramConfig:
        """Guard 1 (SPEC §5.3): reject `diarize` and other owned keys in extras."""
        offending = sorted(RESERVED_QUERY_PARAMS & {k.lower() for k in self.extra_params})
        if offending:
            raise ValueError(
                f"extra_params may not set reserved Deepgram parameters: "
                f"{', '.join(offending)}. `diarize` is deprecated and is rejected "
                f"when sent alongside `diarize_model`; the others are controlled by "
                f"dedicated configuration fields."
            )
        return self

    @model_validator(mode="after")
    def _guard_fixed_language(self) -> DeepgramConfig:
        """FIXED mode without a language code would emit `language=`."""
        if self.language_mode is LanguageMode.FIXED and not self.language.strip():
            raise ValueError("language_mode=fixed requires a non-empty `language` code.")
        return self

    def to_query_params(self) -> dict[str, Any]:
        """Build the `/v1/listen` query parameters.

        Guard 2 (SPEC §5.3) is satisfied structurally: exactly one branch of the
        language strategy runs, so `detect_language` and `language` can never
        both appear in the result.

        A list value means a repeated query parameter, which is how Deepgram
        expects a restricted detection set (`detect_language=en&detect_language=fr`).
        """
        params: dict[str, Any] = {
            "model": self.model,
            "diarize_model": self.diarize_model,
            "smart_format": self.smart_format,
            "punctuate": self.punctuate,
            "utterances": self.utterances,
        }

        if self.language_mode is LanguageMode.DETECT:
            params["detect_language"] = (
                list(self.detect_candidates) if self.detect_candidates else True
            )
        elif self.language_mode is LanguageMode.FIXED:
            params["language"] = self.language
        else:  # LanguageMode.MULTI
            params["language"] = "multi"

        params.update(self.extra_params)
        return params


# ---------------------------------------------------------------------------
# Ingestion, audio, transcript, analysis
# ---------------------------------------------------------------------------


class IngestionConfig(BaseModel):
    """URL and upload intake limits (SPEC §5.1)."""

    #: Generous headroom over the brief's 10-15 minute test videos, while still
    #: bounding what a single request can cost us.
    max_duration_sec: float = 1800.0
    max_bytes: int = 500 * 1024 * 1024

    allowed_schemes: tuple[str, ...] = ("http", "https")

    #: Refuse loopback, link-local, and RFC1918 targets. A URL parameter that
    #: reaches an HTTP client is an SSRF vector by default.
    block_private_addresses: bool = True

    socket_timeout_sec: float = 30.0


class AudioConfig(BaseModel):
    """FFmpeg normalisation target (SPEC 5.2)."""

    sample_rate: int = 16_000
    channels: int = 1
    audio_format: Literal["flac", "wav"] = "flac"

    #: Off by default
    denoise: bool = False


class SegmentationConfig(BaseModel):
    """Word-stream to segment construction (AD-3, AD-4)."""

    #: Cut a segment when the silence between two words exceeds this.
    pause_threshold_sec: float = 0.7
    #: Upper bound on a segment, applied at a sentence boundary.
    max_segment_sec: float = 30.0

    smoothing_enabled: bool = False
    smoothing_max_turn_sec: float = 1.0
    smoothing_min_confidence: float = 0.5


class AnalysisConfig(BaseModel):
    """LLM map-reduce configuration (AD-7, AD-8)."""

    #: SPEC 11 open question 3: pin exact model IDs at implementation time.
    map_model: str = "gpt-5-mini"
    reduce_model: str = "gpt-5"

    #: Chunks are cut on segment boundaries, never mid-utterance (AD-7).
    chunk_token_budget: int = 2000
    chunk_overlap_segments: int = 1
    encoding_name: str = "cl100k_base"

    #: Bounded concurrency for the map stage.
    map_concurrency: int = 5

    reduce_batch_size: int = 8

    max_retries: int = 3
    backoff_base_sec: float = 1.0
    timeout_sec: float = 120.0


class CacheConfig(BaseModel):
    """Response cache keyed by SHA-256 of normalised audio (AD-2).

    Protects the API credit during development, makes reruns instant, and turns
    cached responses into the fixtures that let the unit suite run with no keys.
    """

    enabled: bool = True
    directory: Path = Path(".cache/stt")


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Root configuration, populated from environment and `.env`.

    Secrets sit at the top level so they read from the conventional flat names
    (`DEEPGRAM_API_KEY`, `OPENAI_API_KEY`). Behavioural configuration is nested
    and uses the `__` delimiter, e.g. `SEGMENTATION__PAUSE_THRESHOLD_SEC=0.9`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    deepgram_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    log_level: str = "INFO"
    log_json: bool = False

    deepgram: DeepgramConfig = Field(default_factory=DeepgramConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    def require_deepgram_key(self) -> str:
        """Return the Deepgram key or log the failure.

        Called at the point of use rather than at startup, so that importing the
        application, running the unit suite, and reading cached responses all
        work without any key present (AD-2).
        """
        if self.deepgram_api_key is None:
            raise ConfigurationError(
                "DEEPGRAM_API_KEY is not set. Copy .env.example to .env and add "
                "your key, or run against cached fixtures."
            )
        return self.deepgram_api_key.get_secret_value()

    def require_openai_key(self) -> str:
        """Return the OpenAI key or log the failure.

        A missing key shows as ANALYSIS_FAILED: the
        transcript is still passed on and returned (AD-9).
        """
        if self.openai_api_key is None:
            raise ConfigurationError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.openai_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached so configuration is parsed and validated once. Tests that need a
    different configuration construct `Settings(...)` directly or call
    `get_settings.cache_clear()`.
    """
    return Settings()


__all__ = [
    "RESERVED_QUERY_PARAMS",
    "AnalysisConfig",
    "AudioConfig",
    "CacheConfig",
    "DeepgramConfig",
    "IngestionConfig",
    "LanguageMode",
    "SegmentationConfig",
    "Settings",
    "get_settings",
]
