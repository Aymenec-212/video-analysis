"""Local cache of raw speech-API responses (AD-2).

One mechanism, three payoffs: it protects the API credit during development,
makes repeated runs instant, and turns real responses into the fixtures that let
the entire unit suite pass with no API keys.

**The key covers the request, not just the audio.** Hashing only the audio would
be wrong: the same file transcribed with `language=fr` and with
`detect_language=en,fr` are different requests with different answers, and a
key blind to that would serve one as the other after a configuration change —
a stale result presented as a fresh one, which is exactly the class of silent
wrongness the brief cares about.

Stored payloads are the **unmodified** API response. A cache entry is therefore
byte-identical to what the wire produced, so a parser test against a fixture is a
test against the real format rather than against our own round-trip.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from app.core.config import CacheConfig
from app.core.logging import get_logger

logger = get_logger(__name__)


def canonicalise_params(params: Mapping[str, Any]) -> str:
    """Render request parameters to a stable string.

    Sorted keys make the encoding independent of dictionary order. List values
    keep their order, since `detect_language=en&detect_language=fr` and its
    reverse express different candidate priorities and should not collide.
    """
    return json.dumps(params, sort_keys=True, default=str, separators=(",", ":"))


def build_cache_key(audio_sha256: str, params: Mapping[str, Any]) -> str:
    """Compose the cache key from audio content and request parameters.

    The readable `{audio}_{params}` shape is intentional: cache files can be
    matched to their audio by eye during debugging, which a single opaque digest
    would prevent.
    """
    param_digest = sha256(canonicalise_params(params).encode("utf-8")).hexdigest()
    return f"{audio_sha256[:16]}_{param_digest[:12]}"


class ResponseCache:
    """Filesystem cache for speech-API responses."""

    def __init__(self, config: CacheConfig) -> None:
        self._config = config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def directory(self) -> Path:
        return self._config.directory

    def path_for(self, key: str) -> Path:
        return self._config.directory / f"{key}.json"

    async def get(self, key: str) -> dict[str, Any] | None:
        """Return a cached response, or None on any miss.

        A malformed entry is treated as a miss rather than an error. A cache is
        an optimisation; letting a corrupt file abort a request would make the
        optimisation less reliable than not having it.
        """
        if not self.enabled:
            return None

        path = self.path_for(key)
        if not path.exists():
            return None

        try:
            payload = await asyncio.to_thread(self._read, path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("discarding unreadable cache entry", cache_key=key, error=str(exc))
            return None

        if not isinstance(payload, dict):
            logger.warning("discarding cache entry of unexpected shape", cache_key=key)
            return None

        logger.debug("cache hit", cache_key=key)
        return payload

    async def put(self, key: str, payload: dict[str, Any]) -> None:
        """Store a response.

        Write failures are logged and swallowed: a full disk should not fail a
        request whose transcription already succeeded.
        """
        if not self.enabled:
            return

        try:
            await asyncio.to_thread(self._write, self.path_for(key), payload)
            logger.debug("cache write", cache_key=key)
        except OSError as exc:
            logger.warning("could not write cache entry", cache_key=key, error=str(exc))

    @staticmethod
    def _read(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _write(path: Path, payload: dict[str, Any]) -> None:
        """Write atomically via a temporary file in the same directory.

        Writing in place would leave a truncated, unparseable file if the process
        died mid-write — and that file would then be read as a miss on every
        subsequent run, silently disabling the cache for that entry. Renaming
        within a directory is atomic on POSIX, so an entry either exists complete
        or does not exist.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        descriptor, temp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp"
        )
        try:
            # fdopen takes ownership, so closing the writer closes the descriptor.
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            os.replace(temp_name, path)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise


__all__ = ["ResponseCache", "build_cache_key", "canonicalise_params"]
