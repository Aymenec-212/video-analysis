"""Tests for structured logging, primarily the redaction guarantee."""

from __future__ import annotations

import json
import logging

import pytest

from app.core.logging import HumanFormatter, JSONFormatter, get_logger


def _record(**context: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="pipeline stage complete",
        args=(),
        exc_info=None,
    )
    for key, value in context.items():
        setattr(record, key, value)
    return record


class TestJSONFormatter:
    def test_emits_a_single_json_object(self) -> None:
        payload = json.loads(JSONFormatter().format(_record()))

        assert payload["level"] == "INFO"
        assert payload["logger"] == "app.test"
        assert payload["message"] == "pipeline stage complete"

    def test_structured_context_is_preserved_as_fields(self) -> None:
        payload = json.loads(JSONFormatter().format(_record(stage="audio", duration=12.5)))

        assert payload["stage"] == "audio"
        assert payload["duration"] == 12.5

    def test_non_serialisable_values_do_not_break_formatting(self) -> None:
        """A logging call must never be the thing that takes down a request."""
        payload = json.loads(JSONFormatter().format(_record(path=object())))
        assert "path" in payload


class TestRedaction:
    """Credentials reaching logs is a routine way keys leak."""

    def test_api_keys_are_redacted(self) -> None:
        payload = json.loads(JSONFormatter().format(_record(deepgram_api_key="dg-secret")))

        assert "dg-secret" not in json.dumps(payload)
        assert payload["deepgram_api_key"] == "***redacted***"

    def test_redaction_covers_common_secret_names(self) -> None:
        record = _record(
            auth_token="t", client_secret="s", user_password="p", authorization="a"
        )
        rendered = JSONFormatter().format(record)

        for value in ("\"t\"", "\"s\"", "\"p\"", "\"a\""):
            assert value not in rendered

    def test_redaction_applies_to_human_output_too(self) -> None:
        rendered = HumanFormatter().format(_record(openai_api_key="sk-live"))
        assert "sk-live" not in rendered

    def test_ordinary_fields_are_not_redacted(self) -> None:
        payload = json.loads(JSONFormatter().format(_record(video_id="abc123")))
        assert payload["video_id"] == "abc123"


class TestHumanFormatter:
    def test_message_and_context_both_appear(self) -> None:
        rendered = HumanFormatter().format(_record(stage="audio"))

        assert "pipeline stage complete" in rendered
        assert "stage=audio" in rendered

    def test_renders_without_context(self) -> None:
        assert "pipeline stage complete" in HumanFormatter().format(_record())


class TestBoundLogger:
    def test_bind_returns_a_new_logger(self) -> None:
        """Mutation would leak context between concurrent pipeline runs."""
        base = get_logger("app.test")
        bound = base.bind(video_id="abc")

        assert bound is not base
        assert base.context == {}
        assert bound.context == {"video_id": "abc"}

    def test_bindings_accumulate(self) -> None:
        logger = get_logger("app.test").bind(video_id="abc").bind(stage="audio")
        assert logger.context == {"video_id": "abc", "stage": "audio"}

    def test_context_reaches_the_emitted_record(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            get_logger("app.test").bind(video_id="abc").info("working")

        assert caplog.records[-1].video_id == "abc"  # type: ignore[attr-defined]

    def test_per_call_context_overrides_bound_context(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            get_logger("app.test").bind(stage="audio").info("working", extra={"stage": "stt"})

        assert caplog.records[-1].stage == "stt"  # type: ignore[attr-defined]
