"""Concurrency and retry helpers.

Both external services this pipeline calls — the speech API and the LLM — fail in
the same two ways: transient faults worth retrying, and rate limits that need
backing off. The delay schedule lives here so both use the same one.

**Full jitter is not optional.** Under concurrency, plain exponential backoff
makes every retrying request wake at the same moment and collide again, turning
one rate-limit response into a synchronised herd. Randomising each delay across
the whole window spreads them out. The cost is that delays are no longer exactly
predictable, which is why tests assert bounds rather than values.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Coroutine, Iterator, Sequence
from typing import Any, TypeVar

T = TypeVar("T")

#: Ceiling on any single delay. Without it, exponential growth on a long retry
#: budget produces waits longer than the request timeout they are protecting.
MAX_DELAY_SEC = 30.0


def backoff_delays(
    attempts: int,
    base_sec: float,
    *,
    jitter: bool = True,
    max_delay_sec: float = MAX_DELAY_SEC,
) -> Iterator[float]:
    """Yield the delay to wait *before* each retry.

    Yields `attempts - 1` values: the first attempt is immediate, and a delay
    only precedes a retry. With `attempts=3, base_sec=1.0` the un-jittered
    schedule is 1s then 2s.

    With jitter each delay is drawn uniformly from `[0, computed]` — "full
    jitter", which spreads a herd more effectively than adding a small random
    offset to a fixed delay.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if base_sec < 0:
        raise ValueError("base_sec must not be negative")

    for attempt in range(attempts - 1):
        capped = min(base_sec * (2**attempt), max_delay_sec)
        yield random.uniform(0.0, capped) if jitter else capped


__all__ = ["MAX_DELAY_SEC", "backoff_delays", "gather_bounded", "retry_async"]


async def gather_bounded(
    limit: int,
    coroutines: Sequence[Coroutine[Any, Any, T]],
    *,
    return_exceptions: bool = False,
) -> list[T | BaseException]:
    """Await coroutines with at most `limit` running at once.

    A plain `asyncio.gather` over every chunk of a long transcript would fire
    dozens of simultaneous LLM requests and collect rate limits for the trouble.
    The semaphore bounds in-flight work while still overlapping it.

    Coroutines are cheap until awaited, so wrapping them here does not start
    anything early.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")

    semaphore = asyncio.Semaphore(limit)

    async def guarded(coroutine: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coroutine

    return list(
        await asyncio.gather(
            *(guarded(c) for c in coroutines), return_exceptions=return_exceptions
        )
    )


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base_sec: float,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Run `operation`, retrying transient failures with jittered backoff.

    Takes a factory rather than a coroutine because a coroutine can only be
    awaited once — retrying would raise `RuntimeError` instead of retrying.

    Exceptions outside `retry_on` propagate immediately. A malformed request does
    not become valid by being sent again, and retrying it wastes the caller's
    time to reach the same answer.
    """
    delays = list(backoff_delays(attempts, base_sec))
    last: BaseException | None = None

    for attempt in range(attempts):
        try:
            return await operation()
        except retry_on as exc:
            last = exc
            if attempt < len(delays):
                await asyncio.sleep(delays[attempt])

    assert last is not None  # unreachable: the loop either returns or sets `last`
    raise last
