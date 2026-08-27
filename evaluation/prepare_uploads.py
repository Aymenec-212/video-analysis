"""One-off benchmark preparation: download the corpus and write the upload manifest.

The concurrency sweep runs on local files rather than URLs, for two reasons.

Repeatedly downloading the same YouTube videos across a dozen sweep runs invites
the bot-check we already handle as a failure mode — contaminating a performance
measurement with the very error the system is designed to survive. And download
time is network variance we are not trying to measure: the URL route is proven by
the correctness run, so the sweep should not re-prove it.

Reads the URL cases out of `manifest.json` so filenames and expectations cannot
drift from the corpus definition, then writes `uploads.json` pointing at what it
downloaded.

    uv run python evaluation/prepare_uploads.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parent
MEDIA_DIR = EVALUATION_DIR / "media"

#: Modest resolution keeps downloads quick. Only the audio track survives
#: normalisation, so picture quality is irrelevant to every measurement here —
#: but a real video file keeps the upload route representative.
FORMAT = "bestvideo[height<=480]+bestaudio/best"

def download(url: str, destination: Path) -> bool:
    if destination.exists():
        print(f"    already present: {destination.name}")
        return True

    print(f"    downloading -> {destination.name}")
    result = subprocess.run(
        [
            "yt-dlp",
            "-f", FORMAT,
            "--merge-output-format", "mp4",
            "-o", str(destination),
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"    FAILED: {result.stderr.strip()[:200]}")
        return False
    return True


def main() -> None:
    manifest_path = EVALUATION_DIR / "manifest.json"
    cases = json.loads(manifest_path.read_text())
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    upload_cases: list[dict[str, object]] = []
    failures: list[str] = []

    for case in cases:
        source = str(case["source"])
        name = str(case["name"])

        # Failure cases stay as they are: the invalid URL and the missing video
        # have to reach the network layer to be rejected by it.
        if name.startswith("failure-"):
            continue

        # Checked before the scheme test: a placeholder does not start with
        # http, so testing the scheme first would classify it as a local path
        # and emit a manifest pointing at a file that does not exist.
        if "REPLACE_WITH" in source:
            failures.append(f"{name}: manifest still holds a placeholder URL")
            continue

        if not source.startswith(("http://", "https://")):
            # Already a local file — carry it through unchanged.
            upload_cases.append(
                {
                    "name": name,
                    "source": source,
                    "profile": case.get("profile", ""),
                    "expected_speakers": case.get("expected_speakers"),
                }
            )
            continue

        destination = MEDIA_DIR / f"{name}.mp4"
        print(f"  {name}")
        if download(source, destination):
            upload_cases.append(
                {
                    "name": name,
                    "source": f"evaluation/media/{destination.name}",
                    "profile": case.get("profile", ""),
                    "expected_speakers": case.get("expected_speakers"),
                }
            )
        else:
            failures.append(f"{name}: download failed")

    uploads_path = EVALUATION_DIR / "uploads.json"
    uploads_path.write_text(json.dumps(upload_cases, indent=2), encoding="utf-8")

    print(f"\nwrote {uploads_path} with {len(upload_cases)} cases")
    total_mb = sum(
        (MEDIA_DIR / Path(str(c["source"])).name).stat().st_size
        for c in upload_cases
        if (MEDIA_DIR / Path(str(c["source"])).name).exists()
    ) / (1024 * 1024)
    print(f"corpus on disk: {total_mb:.0f} MB")

    if failures:
        print("\nunresolved:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)


if __name__ == "__main__":
    main()