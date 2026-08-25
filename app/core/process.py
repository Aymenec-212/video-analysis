"""Guarded subprocess execution.

Both ffprobe and FFmpeg are invoked as external processes, and both are handed
paths that ultimately derive from user input. Two rules are enforced here rather
than at each call site:

*Never a shell.* Commands are argument lists, so a filename containing shell
metacharacters is an ordinary filename rather than an injection.

*Always a timeout.* FFmpeg on malformed input can block indefinitely. An
unbounded subprocess in an API request handler is a stalled worker, so every
invocation carries a deadline and a timeout is reported as a distinct outcome.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Outcome of a completed subprocess."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def stderr_summary(self, limit: int = 400) -> str:
        """First lines of stderr, for error messages and logs.

        FFmpeg is verbose even at `-v error`, and the useful diagnosis is at the
        top. Truncated so a failure detail cannot balloon an API response.
        """
        text = self.stderr.strip()
        return text[:limit] + "…" if len(text) > limit else text


class CommandTimeout(Exception):
    """A subprocess exceeded its deadline and was killed.

    Deliberately not a `PipelineError`: the meaning of a timeout depends on the
    stage, so each caller maps it to the right taxonomy entry rather than
    inheriting a generic one.
    """

    def __init__(self, command: str, timeout_sec: float) -> None:
        self.command = command
        self.timeout_sec = timeout_sec
        super().__init__(f"`{command}` exceeded {timeout_sec:.0f}s and was terminated")


def require_executable(name: str) -> str:
    """Resolve an executable or fail with actionable guidance.

    Missing FFmpeg is the most common first-run failure, and the default
    `FileNotFoundError` does not say what to install.
    """
    resolved = shutil.which(name)
    if resolved is None:
        raise ConfigurationError(
            f"`{name}` was not found on PATH. Install FFmpeg "
            f"(`apt-get install ffmpeg` or `brew install ffmpeg`); "
            f"Google Colab already provides it.",
            detail={"executable": name},
        )
    return resolved


async def run_command(
    args: list[str | Path],
    *,
    timeout_sec: float,
) -> CommandResult:
    """Run a command to completion, capturing output.

    A non-zero exit is returned as a result rather than raised: callers
    distinguish expected failures (unreadable media) from bugs, and that
    judgement belongs to them.

    On timeout the process is killed and awaited before raising, so no orphan
    FFmpeg process survives the request that started it.
    """
    argv = [str(arg) for arg in args]
    require_executable(argv[0])

    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_sec)
    except TimeoutError:
        process.kill()
        # Reap the killed process; skipping this leaves a zombie and an
        # "unawaited process" warning under asyncio.
        await process.wait()
        raise CommandTimeout(argv[0], timeout_sec) from None

    return CommandResult(
        returncode=process.returncode if process.returncode is not None else -1,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
    )


__all__ = ["CommandResult", "CommandTimeout", "require_executable", "run_command"]
