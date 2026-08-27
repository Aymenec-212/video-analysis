"""Evaluation suite: run the test corpus against a live API and report.

Exercises the real HTTP surface rather than calling the pipeline in-process,
because that is what an evaluator will run and it is where the differences live:
concurrency limits, upload handling, timeouts, and real provider latency.

URLs are submitted as JSON; local paths are submitted as multipart uploads, so
one manifest covers both ingestion routes. Requests run concurrently to surface
interference between them.

Usage:

    uv run uvicorn app.api.routes:app --port 8000        # in another shell
    uv run python evaluation/run_suite.py evaluation/manifest.json

    # tighter or looser overlap
    uv run python evaluation/run_suite.py manifest.json --concurrency 5

Writes `evaluation/results.md` for the README's results section.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Case:
    """One entry in the manifest."""

    name: str
    source: str
    profile: str = ""
    expected_speakers: int | None = None

    #: "url" or "upload". Inferred from the source when omitted; set it
    #: explicitly for failure cases, since a rejected scheme like `file://` must
    #: still be submitted through the URL branch to exercise the rejection.
    route: str | None = None

    @property
    def is_url(self) -> bool:
        if self.route is not None:
            return self.route == "url"
        return self.source.startswith(("http://", "https://"))


@dataclass
class Outcome:
    """What one submission produced."""

    case: Case
    http_status: int
    elapsed_sec: float
    body: dict[str, Any] = field(default_factory=dict)
    transport_error: str | None = None

    @property
    def status(self) -> str:
        if self.transport_error:
            return "transport-error"
        return str(self.body.get("status") or self.body.get("errors", [{}])[0].get("code", "?"))

    @property
    def speakers(self) -> int | None:
        return self.body.get("number_of_speakers")

    @property
    def speaker_error(self) -> str:
        if self.case.expected_speakers is None or self.speakers is None:
            return ""
        return f"{self.speakers - self.case.expected_speakers:+d}"

    @property
    def segment_count(self) -> int:
        return len(self.body.get("transcript") or [])

    @property
    def has_summary(self) -> bool:
        return bool(self.body.get("summary"))


async def submit(client: httpx.AsyncClient, case: Case, api: str) -> Outcome:
    """Submit one case over HTTP, by URL or by upload."""
    started = time.monotonic()
    try:
        if case.is_url:
            response = await client.post(
                f"{api}/analyze-video", json={"url": case.source}
            )
        else:
            path = Path(case.source)
            if not path.is_absolute():
                path = (REPO_ROOT / path).resolve()
            if not path.exists():
                return Outcome(case, 0, 0.0, transport_error=f"file not found: {path}")
            response = await client.post(
                f"{api}/analyze-video",
                files={"file": (path.name, path.read_bytes(), "video/mp4")},
            )
    except httpx.HTTPError as exc:
        return Outcome(case, 0, time.monotonic() - started, transport_error=str(exc))

    elapsed = time.monotonic() - started
    try:
        body = response.json()
    except ValueError:
        body = {}
    return Outcome(case, response.status_code, elapsed, body=body)


def _confidences(outcome: Outcome) -> list[float]:
    return [
        s["speaker_confidence"]
        for s in (outcome.body.get("transcript") or [])
        if s.get("speaker_confidence") is not None
    ]


def render(outcomes: list[Outcome], wall_sec: float) -> str:
    """Build the results table and commentary."""
    lines: list[str] = [
        "# Evaluation results",
        "",
        f"Generated from `evaluation/run_suite.py`. "
        f"{len(outcomes)} cases, {wall_sec:.1f}s wall clock.",
        "",
        "| Case | Route | HTTP | Status | Lang | Speakers | Err | Segments | Chunks "
        "| Summary | Degraded | Time |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for o in outcomes:
        route = "url" if o.case.is_url else "upload"
        speakers = "-" if o.speakers is None else str(o.speakers)
        degraded = "yes" if o.body.get("degraded") else "no"
        lang = (o.body.get("language") or {}).get("code") or "-"
        chunks = (o.body.get("provenance") or {}).get("chunk_count") or "-"
        lines.append(
            f"| {o.case.name} | {route} | {o.http_status} | {o.status} | {lang} | "
            f"{speakers} | {o.speaker_error or '-'} | {o.segment_count} | {chunks} | "
            f"{'yes' if o.has_summary else 'no'} | {degraded} | {o.elapsed_sec:.1f}s |"
        )

    lines += ["", "## Per-case notes", ""]
    for o in outcomes:
        lines.append(f"### {o.case.name}")
        if o.case.profile:
            lines.append(f"*{o.case.profile}*")
        lines.append("")
        if o.transport_error:
            lines += [f"Transport error: `{o.transport_error}`", ""]
            continue

        lines.append(f"- Route: {'URL' if o.case.is_url else 'file upload'}")
        language = o.body.get("language") or {}
        if language.get("code"):
            meaningful = "" if language.get("confidence_is_meaningful", True) else \
                " (confidence not meaningful for this language)"
            lines.append(
                f"- Language detected: `{language['code']}` "
                f"at {language.get('confidence', 0):.3f}{meaningful}"
            )
        lines.append(f"- HTTP {o.http_status}, status `{o.status}`")
        if o.body.get("title"):
            lines.append(f"- Title: {o.body['title']}")
        if o.body.get("duration"):
            lines.append(f"- Duration: {o.body['duration']:.1f}s")
        if o.speakers is not None:
            expected = (
                f" (expected {o.case.expected_speakers}, error {o.speaker_error})"
                if o.case.expected_speakers is not None
                else ""
            )
            lines.append(f"- Speakers detected: {o.speakers}{expected}")

        scores = _confidences(o)
        if scores:
            lines.append(
                f"- Diarization confidence: min {min(scores):.3f}, "
                f"median {statistics.median(scores):.3f}"
            )

        provenance = o.body.get("provenance") or {}
        if provenance:
            lines.append(
                f"- Provenance: model `{provenance.get('resolved_model')}`, "
                f"diarizer `{provenance.get('diarizer_arch')}`, "
                f"cached `{provenance.get('transcription_cached')}`"
            )
            chunks = provenance.get("chunk_count") or 0
            if chunks:
                failed = provenance.get("failed_chunks") or 0
                note = f", {failed} failed" if failed else ""
                lines.append(
                    f"- Analysis: {chunks} excerpt(s){note} "
                    f"{'- transcript was never sent in one prompt' if chunks > 1 else ''}"
                )

        for reason in o.body.get("degraded_reasons") or []:
            lines.append(f"- Degraded: {reason}")
        for error in o.body.get("errors") or []:
            lines.append(f"- Error `{error.get('code')}`: {error.get('message')}")
        lines.append("")

    return "\n".join(lines)


def summarise(outcomes: list[Outcome], wall_sec: float) -> None:
    print(f"\n{'case':<24} {'route':<7} {'http':>5} {'status':<16} "
          f"{'spk':>4} {'err':>4} {'segs':>5} {'sum':>4} {'degr':>5} {'time':>7}")
    print("-" * 96)
    for o in outcomes:
        print(
            f"{o.case.name:<24} {'url' if o.case.is_url else 'upload':<7} "
            f"{o.http_status:>5} {o.status:<16} "
            f"{('-' if o.speakers is None else o.speakers):>4} "
            f"{o.speaker_error or '-':>4} {o.segment_count:>5} "
            f"{'yes' if o.has_summary else 'no':>4} "
            f"{'yes' if o.body.get('degraded') else 'no':>5} {o.elapsed_sec:>6.1f}s"
        )

    sequential = sum(o.elapsed_sec for o in outcomes)
    floor = max((o.elapsed_sec for o in outcomes), default=0.0)
    if wall_sec > 0:
        print(
            f"\nwall clock {wall_sec:.1f}s vs {sequential:.1f}s sequential "
            f"({sequential / wall_sec:.1f}x overlap)"
        )
        if floor > 0:
            # No batch finishes faster than its slowest member, so raw speedup
            # is capped by workload shape rather than by the server. Efficiency
            # against that floor is the number that says whether requests
            # actually overlapped.
            print(
                f"longest single case {floor:.1f}s -> theoretical floor; "
                f"achieved {floor / wall_sec:.0%} of it"
            )

    ok = sum(1 for o in outcomes if o.http_status == 200)
    print(f"{ok}/{len(outcomes)} returned 200")

    matched = [
        o
        for o in outcomes
        if o.case.expected_speakers is not None and o.speakers is not None
    ]
    if matched:
        exact = sum(1 for o in matched if o.speakers == o.case.expected_speakers)
        print(f"{exact}/{len(matched)} matched the expected speaker count exactly")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="JSON manifest of cases")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="base URL")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", default=None, help="markdown path")
    parser.add_argument("--json", default=None, help="machine-readable path")
    parser.add_argument(
        "--label",
        default=None,
        help="run identifier; defaults to the concurrency level. Output files are "
        "named after it so a sweep does not overwrite itself.",
    )
    args = parser.parse_args()
    label = args.label or f"c{args.concurrency}"

    cases = [Case(**entry) for entry in json.loads(Path(args.manifest).read_text())]
    print(f"{len(cases)} cases, concurrency {args.concurrency}, api {args.api}")

    semaphore = asyncio.Semaphore(args.concurrency)
    # trust_env=False so a proxy configured for outbound traffic does not
    # intercept requests to a local server.
    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:
        try:
            health = await client.get(f"{args.api}/health")
            print(f"health: {health.json()}\n")
        except httpx.HTTPError as exc:
            print(f"cannot reach {args.api}: {exc}")
            sys.exit(1)

        async def guarded(case: Case) -> Outcome:
            async with semaphore:
                print(f"  submitting {case.name} ...", flush=True)
                return await submit(client, case, args.api)

        started = time.monotonic()
        outcomes = list(await asyncio.gather(*(guarded(c) for c in cases)))
        wall = time.monotonic() - started

    summarise(outcomes, wall)

    markdown = Path(args.output or REPO_ROOT / "evaluation" / f"results-{label}.md")
    markdown.write_text(render(outcomes, wall), encoding="utf-8")

    # A sweep is only comparable if each level is machine-readable. Collating
    # four markdown tables by eye is where benchmark mistakes come from.
    payload = Path(args.json or REPO_ROOT / "evaluation" / f"results-{label}.json")
    payload.write_text(
        json.dumps(
            {
                "label": label,
                "concurrency": args.concurrency,
                "wall_sec": round(wall, 2),
                "sequential_sec": round(sum(o.elapsed_sec for o in outcomes), 2),
                "longest_case_sec": round(
                    max((o.elapsed_sec for o in outcomes), default=0.0), 2
                ),
                "cases": [
                    {
                        "name": o.case.name,
                        "route": "url" if o.case.is_url else "upload",
                        "http": o.http_status,
                        "status": o.status,
                        "elapsed_sec": round(o.elapsed_sec, 2),
                        "speakers": o.speakers,
                        "expected_speakers": o.case.expected_speakers,
                        "segments": o.segment_count,
                        "chunks": (o.body.get("provenance") or {}).get("chunk_count"),
                        "language": (o.body.get("language") or {}).get("code"),
                        "degraded": bool(o.body.get("degraded")),
                        "transcription_cached": (o.body.get("provenance") or {}).get(
                            "transcription_cached"
                        ),
                    }
                    for o in outcomes
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {markdown}\nwrote {payload}")


if __name__ == "__main__":
    asyncio.run(main())
