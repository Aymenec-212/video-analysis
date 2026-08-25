"""Shared test fixtures.

The unit suite must produce identical results on a fresh clone, on a developer
machine with a populated `.env`, and in CI (AD-2). `Settings` reads both the
ambient environment and `.env` from the working directory, so without isolation
a developer's own credentials would silently change what the tests exercise —
and the failure would appear as an unrelated assertion error days later.

The fixture below is autouse: isolation is the default, and a test that wants
environment input opts back in explicitly with `monkeypatch.setenv`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from app.core.config import Settings, get_settings

# Media fixtures are defined separately; registering them as a plugin keeps
# conftest focused on environment isolation.
pytest_plugins = ["tests.fixtures.media"]

#: Nested configuration prefixes (`SEGMENTATION__PAUSE_THRESHOLD_SEC`) plus the
#: flat secret and logging names.
_MANAGED_PREFIXES = (
    "DEEPGRAM",
    "OPENAI",
    "INGESTION__",
    "AUDIO__",
    "SEGMENTATION__",
    "ANALYSIS__",
    "CACHE__",
    "LOG_",
)


@pytest.fixture(autouse=True)
def isolated_settings_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip application environment variables and ignore any developer `.env`."""
    for name in list(os.environ):
        if name.upper().startswith(_MANAGED_PREFIXES):
            monkeypatch.delenv(name, raising=False)

    # Restored automatically by monkeypatch; `Settings` resolves its sources per
    # instantiation, so mutating the class config is enough.
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()