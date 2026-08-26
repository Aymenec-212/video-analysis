"""Diarization probe: compare configurations on one file.

Motivated by a measured failure. A French TV news package (114s, six speakers)
returned three speaker labels, with four distinct people collapsed into one ID.
The failure is upstream — the log line `words=327 speakers=3` is computed from
raw provider output, before our segmentation runs, and our speaker mapping has no
path that merges two provider IDs.

Deepgram exposes no speaker-count hint and no clustering threshold, so the two
levers we do control are the diarizer version and what we send it. This script
runs the same audio through both and reports what changes.

Usage:

    uv run python evaluation/diarization_probe.py <url-or-path>
    uv run python evaluation/diarization_probe.py <url> --expect 6

Each configuration is one API request against roughly two minutes of audio, so a
full sweep costs a few cents. Responses are cached by audio hash *and request
parameters*, so re-running is free.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio.ffmpeg import normalize_audio  # noqa: E402
from app.core.config import AudioConfig, Settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.core.models import TranscriptionResult  # noqa: E402
from app.ingestion.file_source import load_local_file  # noqa: E402
from app.ingestion.url_source import fetch_from_url  # noqa: E402
from app.pipeline import temporary_workspace  # noqa: E402
from app.stt.cache import ResponseCache  # noqa: E402
from app.stt.deepgram import DeepgramBackend  # noqa: E402
from app.transcript.segmentation import build_transcript  # noqa: E402


@dataclass(frozen=True)
class Probe:
    """One configuration to try."""

    label: str
    sample_rate: int
    channels: int
    diarize_model: str

    def audio_config(self, base: AudioConfig) -> AudioConfig:
        return base.model_copy(
            update={"sample_rate": self.sample_rate, "channels": self.channels}
        )


#: The grid. Two diarizer versions crossed with three audio representations,
#: isolating one variable at a time against the shipped default.
PROBES = [
    Probe("16k mono   · diarize v2", 16_000, 1, "v2"),
    Probe("16k mono   · diarize v1", 16_000, 1, "v1"),
    Probe("16k stereo · diarize v2", 16_000, 2, "v2"),
    Probe("48k mono   · diarize v2", 48_000, 1, "v2"),
    Probe("48k stereo · diarize v2", 48_000, 2, "v2"),
    Probe("48k stereo · diarize v1", 48_000, 2, "v1"),
]


@dataclass
class Observation:
    """What one configuration produced."""

    label: str
    speakers: int
    segments: int
    words: int
    diarizer_arch: str | None
    confidences: list[float]
    talk_time: dict[str, float]
    cached: bool

    @property
    def median_confidence(self) -> float | None:
        return statistics.median(self.confidences) if self.confidences else None

    @property
    def min_confidence(self) -> float | None:
        return min(self.confidences) if self.confidences else None


def _talk_time(result: TranscriptionResult, settings: Settings) -> dict[str, float]:
    """Seconds of speech per public speaker label.

    Under-clustering is most visible here: a merged identity shows up as one
    label holding an implausible share of the runtime.
    """
    transcript = build_transcript(result.words, settings.segmentation)
    totals: dict[str, float] = {}
    for segment in transcript.segments:
        totals[segment.speaker] = totals.get(segment.speaker, 0.0) + segment.duration
    return dict(sorted(totals.items(), key=lambda kv: -kv[1]))


async def run_probe(
    probe: Probe, media_path: Path, settings: Settings, work_dir: Path
) -> Observation:
    audio = await normalize_audio(
        media_path,
        work_dir / probe.label.replace(" ", "").replace("·", "-"),
        probe.audio_config(settings.audio),
    )

    config = settings.deepgram.model_copy(update={"diarize_model": probe.diarize_model})
    backend = DeepgramBackend(
        api_key=settings.require_deepgram_key(),
        config=config,
        cache=ResponseCache(settings.cache),
    )
    try:
        result = await backend.transcribe(audio)
    finally:
        await backend.aclose()

    transcript = build_transcript(result.words, settings.segmentation)
    return Observation(
        label=probe.label,
        speakers=transcript.number_of_speakers,
        segments=len(transcript.segments),
        words=len(result.words),
        diarizer_arch=result.provenance.diarizer_arch,
        confidences=[
            w.speaker_confidence for w in result.words if w.speaker_confidence is not None
        ],
        talk_time=_talk_time(result, settings),
        cached=result.provenance.from_cache,
    )


def report(observations: list[Observation], expected: int | None) -> None:
    print(f"\n{'configuration':<26} {'spk':>4} {'segs':>5} {'arch':>5} "
          f"{'min conf':>9} {'med conf':>9}  {'error':>6}")
    print("-" * 78)

    for obs in observations:
        error = "" if expected is None else f"{obs.speakers - expected:+d}"
        min_c = f"{obs.min_confidence:.3f}" if obs.min_confidence is not None else "-"
        med_c = f"{obs.median_confidence:.3f}" if obs.median_confidence is not None else "-"
        print(
            f"{obs.label:<26} {obs.speakers:>4} {obs.segments:>5} "
            f"{obs.diarizer_arch or '-':>5} {min_c:>9} {med_c:>9}  {error:>6}"
        )

    print("\nspeaking time per label (under-clustering shows as one dominant label):")
    for obs in observations:
        share = "  ".join(f"{k}={v:.0f}s" for k, v in obs.talk_time.items())
        print(f"  {obs.label:<26} {share}")

    if expected is not None:
        best = min(observations, key=lambda o: abs(o.speakers - expected))
        print(f"\nclosest to ground truth ({expected}): {best.label} "
              f"-> {best.speakers} speakers")
        if best.speakers != expected:
            print("  none of the configurations recovered the true count; this is a "
                  "provider limitation to document, not a parameter left untuned.")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="video URL or local file path")
    parser.add_argument(
        "--expect", type=int, default=None, help="ground-truth speaker count"
    )
    args = parser.parse_args()

    configure_logging("WARNING")
    settings = Settings()

    with temporary_workspace(prefix="diarization-probe-") as work_dir:
        if args.source.startswith(("http://", "https://")):
            media = await fetch_from_url(args.source, work_dir / "src", settings.ingestion)
        else:
            media = await load_local_file(Path(args.source), settings.ingestion)

        print(f"source   {media.title}")
        print(f"duration {media.duration}s   from {media.source}")

        observations = []
        for probe in PROBES:
            print(f"  running {probe.label} ...", flush=True)
            observations.append(await run_probe(probe, media.path, settings, work_dir))

        report(observations, args.expect)


if __name__ == "__main__":
    asyncio.run(main())
