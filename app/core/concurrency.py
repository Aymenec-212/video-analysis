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

import random
from collections.abc import Iterator

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


__all__ = ["MAX_DELAY_SEC", "backoff_delays"]
