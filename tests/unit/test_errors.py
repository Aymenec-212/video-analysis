"""Tests for the error taxonomy (SPEC §6.3, AD-9)."""

from __future__ import annotations

import pytest

from app.core.errors import (
    HTTP_STATUS,
    AnalysisFailedError,
    ErrorCode,
    ErrorEntry,
    InvalidURLError,
    NoSpeechDetectedError,
    PipelineError,
    Stage,
    STTFailedError,
)


class TestTaxonomyCompleteness:
    def test_every_error_code_has_an_http_status(self) -> None:
        """A code without a status would crash on `.http_status` at raise time."""
        missing = set(ErrorCode) - set(HTTP_STATUS)
        assert not missing, f"ErrorCode members missing from HTTP_STATUS: {missing}"

    def test_no_orphan_status_mappings(self) -> None:
        assert not set(HTTP_STATUS) - set(ErrorCode)

    def test_every_concrete_error_maps_to_a_taxonomy_code(self) -> None:
        for subclass in _all_subclasses(PipelineError):
            assert subclass.code in HTTP_STATUS


class TestFatalVersusNonFatal:
    """The core of AD-9: which failures still return real work."""

    def test_no_speech_detected_is_not_fatal(self) -> None:
        error = NoSpeechDetectedError()
        assert error.http_status == 200
        assert error.is_fatal is False

    def test_analysis_failure_is_not_fatal(self) -> None:
        """A failed LLM stage must never discard a successful transcript."""
        error = AnalysisFailedError()
        assert error.http_status == 200
        assert error.is_fatal is False

    def test_transcription_failure_is_fatal(self) -> None:
        """Without a transcript there is nothing downstream worth returning."""
        error = STTFailedError()
        assert error.http_status == 502
        assert error.is_fatal is True

    def test_exactly_two_codes_are_non_fatal(self) -> None:
        """Pins the AD-9 contract: adding a third 200 must be a deliberate act."""
        non_fatal = {code for code, status in HTTP_STATUS.items() if status == 200}
        assert non_fatal == {ErrorCode.NO_SPEECH_DETECTED, ErrorCode.ANALYSIS_FAILED}

    def test_is_fatal_is_derived_from_status(self) -> None:
        for subclass in _all_subclasses(PipelineError):
            error = subclass()
            assert error.is_fatal == (error.http_status >= 400)


class TestSerialisation:
    def test_to_entry_produces_the_response_shape(self) -> None:
        error = InvalidURLError("Scheme not allowed", detail={"scheme": "file"})
        entry = error.to_entry()

        assert isinstance(entry, ErrorEntry)
        assert entry.stage is Stage.INGESTION
        assert entry.code is ErrorCode.INVALID_URL
        assert entry.message == "Scheme not allowed"
        assert entry.detail == {"scheme": "file"}

    def test_default_message_used_when_none_supplied(self) -> None:
        assert NoSpeechDetectedError().message == NoSpeechDetectedError.default_message

    def test_detail_defaults_to_empty_dict_not_none(self) -> None:
        """`errors[]` entries always carry a detail object, so callers need no None check."""
        assert STTFailedError().to_entry().detail == {}

    def test_cause_is_retained_for_logging(self) -> None:
        original = ValueError("connection reset")
        error = STTFailedError("upstream failed", cause=original)
        assert error.cause is original


class TestSubclassContract:
    def test_subclass_without_code_or_stage_is_rejected_at_definition(self) -> None:
        """Enforced at import time rather than when first raised in production."""
        with pytest.raises(TypeError, match="must define both"):

            class Incomplete(PipelineError):
                pass

    def test_error_codes_are_stable_strings(self) -> None:
        """Values are part of the public contract; renaming breaks callers."""
        assert ErrorCode.NO_SPEECH_DETECTED.value == "NO_SPEECH_DETECTED"
        assert ErrorCode.ANALYSIS_FAILED.value == "ANALYSIS_FAILED"

    def test_is_a_standard_exception(self) -> None:
        with pytest.raises(PipelineError):
            raise InvalidURLError()


def _all_subclasses(cls: type) -> list[type]:
    found: list[type] = []
    for subclass in cls.__subclasses__():
        found.append(subclass)
        found.extend(_all_subclasses(subclass))
    return found
