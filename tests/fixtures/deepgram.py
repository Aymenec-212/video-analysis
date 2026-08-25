"""Deepgram response fixtures.

Shaped to the schema documented at developers.deepgram.com and recorded in
SPEC §3, verified 2026-08-25. Once real responses are captured during Day 3
evaluation, cached entries replace these — a cache file is byte-identical to
what the API returned (AD-2), so the same tests then run against genuine output.
"""

from __future__ import annotations

from typing import Any


def word(
    text: str,
    start: float,
    end: float,
    *,
    speaker: int | None = 0,
    confidence: float = 0.99,
    speaker_confidence: float | None = 0.95,
    language: str | None = None,
) -> dict[str, Any]:
    """One word entry as Deepgram emits it.

    `word` and `punctuated_word` both appear: with `smart_format=true` the
    punctuated form is the readable one, and our parser must prefer it.
    """
    entry: dict[str, Any] = {
        "word": text.lower().strip(".,!?"),
        "punctuated_word": text,
        "start": start,
        "end": end,
        "confidence": confidence,
    }
    if speaker is not None:
        entry["speaker"] = speaker
    if speaker_confidence is not None:
        entry["speaker_confidence"] = speaker_confidence
    if language is not None:
        entry["language"] = language
    return entry


def diarized_response(
    words: list[dict[str, Any]] | None = None,
    *,
    detected_language: str | None = "fr",
    language_confidence: float | None = 0.97,
    duration: float = 12.0,
    utterances: int | None = 2,
) -> dict[str, Any]:
    """A standard single-language response with diarization."""
    if words is None:
        words = [
            word("Bonjour", 0.5, 1.1, speaker=0),
            word("et", 1.1, 1.3, speaker=0),
            word("bienvenue.", 1.3, 2.0, speaker=0),
            word("Merci", 2.8, 3.2, speaker=1),
            word("de", 3.2, 3.4, speaker=1),
            word("m'avoir", 3.4, 3.9, speaker=1),
            word("invité.", 3.9, 4.5, speaker=1),
        ]

    channel: dict[str, Any] = {
        "alternatives": [
            {
                "transcript": " ".join(w["punctuated_word"] for w in words),
                "confidence": 0.98,
                "words": words,
            }
        ]
    }
    if detected_language is not None:
        channel["detected_language"] = detected_language
    if language_confidence is not None:
        channel["language_confidence"] = language_confidence

    payload: dict[str, Any] = {
        "metadata": {
            "transaction_key": "deprecated",
            "request_id": "11111111-2222-3333-4444-555555555555",
            "sha256": "abc123",
            "created": "2026-08-25T10:00:00.000Z",
            "duration": duration,
            "channels": 1,
            "models": ["30089e05-99d1-4376-b32e-c263170674af"],
            "model_info": {
                "30089e05-99d1-4376-b32e-c263170674af": {
                    "name": "nova-3-general",
                    "version": "2026-01-01.0",
                    "arch": "nova-3",
                }
            },
            "diarize_info": {
                "model_uuid": "99999999-8888-7777-6666-555555555555",
                "arch": "v2",
            },
        },
        "results": {"channels": [channel]},
    }

    if utterances is not None:
        payload["results"]["utterances"] = [
            {"start": 0.5, "end": 2.0, "speaker": 0, "transcript": "Bonjour et bienvenue."}
        ] * utterances

    return payload


def multilingual_response() -> dict[str, Any]:
    """A `language=multi` response.

    Different schema (SPEC §3.3): words carry their own `language`, and the
    alternative gains a `languages` array. There is no channel-level
    `detected_language` because there is no single dominant language.
    """
    words = [
        word("Hello", 0.4, 0.9, speaker=0, language="en"),
        word("everyone.", 0.9, 1.5, speaker=0, language="en"),
        word("Bonjour", 2.0, 2.6, speaker=1, language="fr"),
        word("à", 2.6, 2.8, speaker=1, language="fr"),
        word("tous.", 2.8, 3.3, speaker=1, language="fr"),
    ]

    return {
        "metadata": {
            "request_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "duration": 3.5,
            "channels": 1,
            "model_info": {
                "uuid-1": {"name": "nova-3-general", "version": "1", "arch": "nova-3"}
            },
            "diarize_info": {"model_uuid": "uuid-2", "arch": "v2"},
        },
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "Hello everyone. Bonjour à tous.",
                            "confidence": 0.96,
                            "languages": ["en", "fr"],
                            "words": words,
                        }
                    ]
                }
            ]
        },
    }


def empty_response(duration: float = 5.0) -> dict[str, Any]:
    """A successful transcription of audio containing no speech.

    Not an error: silence genuinely has no words. Whether this becomes
    `NO_SPEECH_DETECTED` is decided by the pipeline, not by the backend.
    """
    return {
        "metadata": {
            "request_id": "00000000-0000-0000-0000-000000000000",
            "duration": duration,
            "channels": 1,
            "model_info": {"uuid": {"name": "nova-3-general", "version": "1", "arch": "nova-3"}},
        },
        "results": {
            "channels": [{"alternatives": [{"transcript": "", "confidence": 0.0, "words": []}]}]
        },
    }


def downgraded_model_response() -> dict[str, Any]:
    """A response where Deepgram silently fell back to a lesser model.

    Happens when a detected language is unavailable on the requested model
    (Nova-3 → Nova-2 → Nova-1 → Enhanced → Base). Invisible unless the resolved
    model is recorded.
    """
    payload = diarized_response(detected_language="th", language_confidence=0.88)
    payload["metadata"]["model_info"] = {
        "uuid-x": {"name": "enhanced-general", "version": "1", "arch": "enhanced"}
    }
    return payload
