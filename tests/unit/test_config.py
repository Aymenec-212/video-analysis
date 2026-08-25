"""Tests for configuration and the SPEC 5.3 validation guards.

These guards exist because two Deepgram parameter combinations fail in ways that
are hard to diagnose from the outside: one is rejected outright, the other
silently does something other than what the caller asked for.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import (
    DeepgramConfig,
    LanguageMode,
    Settings,
    get_settings,
)
from app.core.errors import ConfigurationError


class TestLanguageStrategyIsExclusive:
    """Guard 2 (SPEC §5.3), enforced structurally rather than by validation.

    `detect_language` overrides `language`, so a request carrying both does not
    do what it appears to. Because `to_query_params` branches on `language_mode`,
    no configuration can produce that request.
    """

    @pytest.mark.parametrize("mode", list(LanguageMode))
    def test_detect_language_and_language_are_never_both_emitted(
        self, mode: LanguageMode
    ) -> None:
        params = DeepgramConfig(language_mode=mode).to_query_params()
        assert not ("detect_language" in params and "language" in params)

    def test_detect_mode_emits_restricted_candidate_set(self) -> None:
        """A repeated parameter: detect_language=en&detect_language=fr."""
        params = DeepgramConfig(
            language_mode=LanguageMode.DETECT, detect_candidates=("en", "fr")
        ).to_query_params()

        assert params["detect_language"] == ["en", "fr"]
        assert "language" not in params

    def test_empty_candidate_set_means_unrestricted_detection(self) -> None:
        params = DeepgramConfig(
            language_mode=LanguageMode.DETECT, detect_candidates=()
        ).to_query_params()
        assert params["detect_language"] is True

    def test_fixed_mode_emits_language_only(self) -> None:
        params = DeepgramConfig(
            language_mode=LanguageMode.FIXED, language="fr"
        ).to_query_params()

        assert params["language"] == "fr"
        assert "detect_language" not in params

    def test_multi_mode_emits_language_multi(self) -> None:
        """Nova-3 code-switching. No Arabic in the supported set — SPEC §3.3."""
        params = DeepgramConfig(language_mode=LanguageMode.MULTI).to_query_params()

        assert params["language"] == "multi"
        assert "detect_language" not in params

    def test_fixed_mode_rejects_blank_language(self) -> None:
        with pytest.raises(ValidationError, match="requires a non-empty"):
            DeepgramConfig(language_mode=LanguageMode.FIXED, language="   ")


class TestReservedParameterGuard:
    """Guard 1 (SPEC §5.3): `diarize` alongside `diarize_model` is rejected by
    Deepgram, and the escape hatch is where that mistake would actually be made.
    """

    def test_deprecated_diarize_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="diarize"):
            DeepgramConfig(extra_params={"diarize": "true"})

    @pytest.mark.parametrize(
        "param", ["diarize", "diarize_model", "language", "detect_language", "model"]
    )
    def test_all_owned_parameters_are_rejected_in_extras(self, param: str) -> None:
        with pytest.raises(ValidationError):
            DeepgramConfig(extra_params={param: "x"})

    def test_guard_is_case_insensitive(self) -> None:
        with pytest.raises(ValidationError):
            DeepgramConfig(extra_params={"Diarize": "true"})

    def test_unreserved_extras_pass_through(self) -> None:
        params = DeepgramConfig(extra_params={"filler_words": "false"}).to_query_params()
        assert params["filler_words"] == "false"


class TestTranscriptionDefaults:
    """Defaults encode SPEC §5.3 and the facts verified in §3."""

    def test_diarization_is_enabled_via_diarize_model(self) -> None:
        params = DeepgramConfig().to_query_params()

        assert params["diarize_model"] == "latest"
        assert "diarize" not in params, "the deprecated parameter must never be sent"

    def test_defaults_match_the_specification(self) -> None:
        params = DeepgramConfig().to_query_params()

        assert params["model"] == "nova-3-general"
        assert params["smart_format"] is True
        assert params["punctuate"] is True
        assert params["utterances"] is True
        assert params["detect_language"] == ["en", "fr"]

    def test_diarize_model_only_accepts_documented_versions(self) -> None:
        with pytest.raises(ValidationError):
            DeepgramConfig(diarize_model="v3")

    def test_candidates_are_normalised_and_deduplicated(self) -> None:
        config = DeepgramConfig(detect_candidates=("EN", " fr ", "en", ""))
        assert config.detect_candidates == ("en", "fr")

    def test_client_timeout_exceeds_the_server_processing_limit(self) -> None:
        """Deepgram returns 504 past 10 minutes of processing (SPEC §3.4).

        A shorter local timeout would mask that with a client-side error and lose
        the signal our chunking fallback triggers on (AD-6).
        """
        assert DeepgramConfig().timeout_sec > 600.0


class TestSegmentationDefaults:
    def test_smoothing_ships_disabled(self) -> None:
        """AD-4: enabled only if measurement shows it is needed."""
        assert Settings().segmentation.smoothing_enabled is False

    def test_thresholds_match_the_specification(self) -> None:
        segmentation = Settings().segmentation
        assert segmentation.pause_threshold_sec == 0.7
        assert segmentation.max_segment_sec == 30.0


class TestAudioDefaults:
    def test_normalisation_target_is_mono_16khz(self) -> None:
        audio = Settings().audio
        assert audio.sample_rate == 16_000
        assert audio.channels == 1
        assert audio.audio_format == "flac"

    def test_denoising_is_off_pending_measurement(self) -> None:
        assert Settings().audio.denoise is False


class TestSecrets:
    def test_missing_deepgram_key_raises_a_clear_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="DEEPGRAM_API_KEY"):
            Settings(deepgram_api_key=None).require_deepgram_key()

    def test_missing_openai_key_raises_a_clear_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            Settings(openai_api_key=None).require_openai_key()

    def test_keys_are_not_exposed_by_repr(self) -> None:
        """Settings objects reach logs and tracebacks; secrets must not ride along."""
        settings = Settings(deepgram_api_key="dg-super-secret")
        assert "dg-super-secret" not in repr(settings)

    def test_key_is_retrievable_when_present(self) -> None:
        settings = Settings(deepgram_api_key="dg-super-secret")
        assert settings.require_deepgram_key() == "dg-super-secret"


class TestSettingsConstruction:
    def test_constructs_with_no_environment(self) -> None:
        """AD-2: importing and testing the app must not require any key."""
        settings = Settings()
        assert settings.deepgram_api_key is None
        assert settings.deepgram.model == "nova-3-general"

    def test_nested_configuration_reads_from_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEGMENTATION__PAUSE_THRESHOLD_SEC", "1.25")
        assert Settings().segmentation.pause_threshold_sec == 1.25

    def test_secrets_read_from_flat_environment_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPGRAM_API_KEY", "from-env")
        assert Settings().require_deepgram_key() == "from-env"

    def test_get_settings_is_cached(self) -> None:
        get_settings.cache_clear()
        assert get_settings() is get_settings()
        get_settings.cache_clear()
