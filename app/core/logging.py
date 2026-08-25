"""Structured logging.

Two output modes: human-readable for development and Colab notebooks, JSON for
anywhere logs are collected. Both carry the same structured context, so a field
bound during development is still there in production.

Context travels via the `extra` keyword rather than string interpolation, which
keeps values machine-readable. `get_logger(...).bind(video_id=...)` attaches
context once and every subsequent record carries it.

Implemented on the standard library rather than a logging framework: one less
dependency, and the behaviour we need here is a formatter and an adapter.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import MutableMapping
from typing import Any

#: Attributes present on every LogRecord. Anything outside this set was passed
#: by the caller as structured context and belongs in the output.
_STANDARD_RECORD_FIELDS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "taskName", "thread", "threadName",
    }
)

#: Substrings that mark a context value as sensitive. API keys reaching logs is a
#: routine way credentials leak, so redaction is on by default rather than
#: something each call site has to remember.
_SENSITIVE_KEY_MARKERS = ("key", "token", "secret", "password", "authorization")

_REDACTED = "***redacted***"


def _extract_context(record: logging.LogRecord) -> dict[str, Any]:
    """Pull caller-supplied structured fields off a record, redacting secrets."""
    context: dict[str, Any] = {}
    for key, value in record.__dict__.items():
        if key in _STANDARD_RECORD_FIELDS or key.startswith("_"):
            continue
        lowered = key.lower()
        if any(marker in lowered for marker in _SENSITIVE_KEY_MARKERS):
            context[key] = _REDACTED
        else:
            context[key] = value
    return context


class JSONFormatter(logging.Formatter):
    """One JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_extract_context(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    """Readable single-line output with context appended as key=value pairs."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s %(name)s │ %(message)s",
            datefmt="%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        context = _extract_context(record)
        if context:
            rendered = " ".join(f"{k}={v}" for k, v in context.items())
            base = f"{base} │ {rendered}"
        return base


class BoundLogger(logging.LoggerAdapter):  # type: ignore[type-arg]
    """A logger carrying persistent structured context."""

    @property
    def context(self) -> dict[str, Any]:
        """Bound context as a plain dict.

        `LoggerAdapter.extra` is typed as optional by the standard library, so
        every read normalises it here rather than at each call site.
        """
        return dict(self.extra) if self.extra else {}

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        call_extra = kwargs.get("extra") or {}
        # Per-call context wins, so a caller can override a bound value.
        kwargs["extra"] = {**self.context, **call_extra}
        return msg, kwargs

    def bind(self, **context: Any) -> BoundLogger:
        """Return a new logger with additional context attached.

        Returns a new instance rather than mutating, so context cannot leak
        between concurrent pipeline runs sharing a module-level logger.
        """
        return BoundLogger(self.logger, {**self.context, **context})


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Install the root handler. Idempotent — safe to call more than once."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter() if json_output else HumanFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())

    # These libraries are chatty at INFO and drown out pipeline logs.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str, **context: Any) -> BoundLogger:
    """Return a logger, optionally pre-bound with context."""
    return BoundLogger(logging.getLogger(name), context)


__all__ = ["BoundLogger", "HumanFormatter", "JSONFormatter", "configure_logging", "get_logger"]