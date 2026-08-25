"""The unified media representation (SPEC 5.1).

A URL and an uploaded file are different acquisition problems and identical
downstream problems. `MediaSource` is where they stop differing: once ingestion
returns one, nothing further in the pipeline knows or cares which path produced
it. That is what keeps transcription, segmentation, and analysis free of
source-specific branching.

The brief requires title, duration, description, and source for URL inputs.
Uploads have no description and no remote title, so those are filled from what
the file actually provides rather than invented.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

#: Reported as `source` when the input was uploaded rather than fetched.
UPLOAD_SOURCE = "upload"


class SourceKind(StrEnum):
    """How the media entered the system."""

    URL = "url"
    UPLOAD = "upload"


class MediaSource(BaseModel):
    """A local media file plus the metadata the response contract requires."""

    kind: SourceKind

    #: Local path to the downloaded or uploaded file. Internal only — never
    #: serialised into an API response, which would leak filesystem layout.
    path: Path

    #: Remote title, or the filename stem for uploads.
    title: str

    #: Seconds. Optional because some containers genuinely carry no duration,
    #: and reporting null is preferable to inventing a number.
    duration: float | None = None

    #: True when `duration` came from ffprobe rather than from source-reported
    #: metadata. SPEC 5.1 requires ffprobe as the authority; when a fallback is
    #: used, this records that the value is unverified rather than silently
    #: presenting it as measured.
    duration_verified: bool = False

    #: Source description where one exists. Always None for uploads.
    description: str | None = None

    #: Extractor name (e.g. "Youtube") or `UPLOAD_SOURCE`.
    source: str

    size_bytes: int = 0

    #: Original URL for URL inputs, for provenance in logs and errors.
    original_url: str | None = None


__all__ = ["UPLOAD_SOURCE", "MediaSource", "SourceKind"]
