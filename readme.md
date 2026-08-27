# AI Video Transcription & Analysis

Submit a video URL or a video file. Receive a speaker-attributed, timestamped
transcript plus an LLM-generated summary, key points, and topics.

> **Résumé.** Ce système récupère une vidéo (URL ou fichier), extrait et normalise
> la piste audio, produit une transcription horodatée avec détection des
> locuteurs, puis analyse le contenu via un LLM. Le tout est exposé par une API
> `POST /analyze-video`. Toutes les décisions techniques sont justifiées et
> mesurées ci-dessous.

Runs on CPU-only. The full test suite runs with no
API keys.

---

## Quick start

```bash
uv sync --extra dev
cp .env.example .env          # add DEEPGRAM_API_KEY and OPENAI_API_KEY
uv run uvicorn app.api.routes:app --port 8000
```

Submit a URL:

```bash
curl -X POST http://127.0.0.1:8000/analyze-video \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtu.be/45P34gxmXw4"}'
```

Submit a file:

```bash
curl -X POST http://127.0.0.1:8000/analyze-video \
  -F "file=@interview.mp4"
```

Verify the installation without any credentials:

```bash
uv run pytest -q     
```

---

## Requirements traceability

Every numbered requirement in the brief maps to a module and a test.

| # | Requirement | Module | Test |
|---|---|---|---|
| 1 | Accept URL or local file; extract title, duration, description, source | `ingestion/` | `test_ingestion.py` |
| 2 | Extract and prepare audio; handle multiple formats | `audio/` | `test_ffmpeg.py`, `test_probe.py` |
| 3 | Timestamped transcription | `stt/` | `test_deepgram.py`, `test_cache.py` |
| 4 | Speaker detection and segment-to-speaker association | `transcript/segmentation.py` | `test_segmentation.py` |
| 5 | Dynamic speaker count, no fixed assumption | `transcript/speakers.py` | `test_speakers.py` |
| 6 | LLM summary / key points / topics | `analysis/` | `test_analysis.py` |
| 7 | Long-content strategy without single-prompt dependency | `analysis/chunking.py`, `reduce.py` | `test_chunking.py` |
| API | `POST /analyze-video` accepting URL or file | `api/routes.py` | `test_api.py` |
| Robustness | Named failure modes, no fabricated output | `core/errors.py`, `pipeline.py` | `test_api.py` |
| Bonus | Automatic language detection | `stt/deepgram.py` | `test_deepgram.py` |
| Bonus | Multi-language support | `stt/deepgram.py` | `test_deepgram.py` |

---

## Architecture

```
Input (URL | file)
   ↓  ingestion/          yt-dlp or upload  →  MediaSource
   ↓  audio/              FFmpeg → mono 16 kHz FLAC, ffprobe metadata
   ↓  stt/                Deepgram Nova-3, cached by audio SHA-256
   ↓  transcript/         words → segments → SPEAKER_NN → validation
   ↓  analysis/chunking   token-budgeted, segment-aligned chunks
   ↓  analysis/map        concurrent per-chunk analysis (semaphore-bounded)
   ↓  analysis/reduce     hierarchical fold → summary / key points / topics
   ↓  Validated JSON
```

`app/pipeline.py` orchestrates the six stages and owns the failure contract.
`app/api/routes.py` is a thin transport layer: it decodes the request, calls the
pipeline, and serialises the result. No stage knows whether its input arrived as
a URL or an upload — both converge on `MediaSource` at ingestion.

```
app/
  pipeline.py    stage orchestration, failure contract
  api/           routes.py, schemas.py, dependencies.py
  ingestion/     url_source.py, file_source.py, metadata.py
  audio/         ffmpeg.py, probe.py
  stt/           base.py (Protocol), deepgram.py, cache.py
  transcript/    segmentation.py, speakers.py, validation.py
  analysis/      chunking.py, map.py, reduce.py, prompts.py, schemas.py, llm.py
  core/          config.py, errors.py, models.py, concurrency.py, logging.py, process.py
evaluation/      run_suite.py, diarization_probe.py, prepare_uploads.py
tests/           unit (no network, no keys), integration (marked, skipped by default)
```

---

## Design decisions

### Speech-to-text: Deepgram Nova-3

CPU-only is satisfied with the use of external api service.

suggestion for potential local application: Self-hosted faster-whisper plus pyannote.

Nova-3 covers French as well

### Diarization: provider clustering, in-house segmentation

Deepgram returns word-level speaker labels. This system consumes
`alternatives[0].words[]` and builds segments itself. `utterances=true` is
requested only as a cross-check.

**The algorithm.** Walk the word stream. Close the current segment when:

1. the speaker changes;
2. the gap since the previous word exceeds `PAUSE_THRESHOLD` (0.7 s);
3. the segment has run past `MAX_SEGMENT_SEC` (30 s) and the previous word ends
   a sentence;
4. the segment has run past `HARD_MAX_SEGMENT_SEC` (60 s), regardless of
   punctuation.

Rules 3 and 4 exist because segments feed LLM chunking, which cuts only on
segment boundaries. One unbounded segment would defeat that.

Speaker labels are assigned **by order of first appearance**, not by the
provider's integer ordering. Provider labels are cluster identifiers with no
guaranteed ordering; identical audio must produce an identical transcript.
`SPEAKER_01` is whoever speaks first. Speech the provider could not attribute is
labelled `SPEAKER_UNKNOWN` and excluded from `number_of_speakers`, because
assigning it to `SPEAKER_01` would assert a count that was never measured.

### Audio preprocessing: mono, 16 kHz, FLAC

FFmpeg reduces every input to one representation. Mono.


**Audio is not segmented before transcription.** The brief's example diagram
shows `Segmentation → Transcription`, and -2 lists segmentation as optional.
Splitting audio before diarization breaks it: speaker labels are not comparable
across independently diarized chunks. Chunking belongs at the LLM stage.

### Long content: map-reduce with hierarchical aggregation

The transcript is never sent to the model in one prompt.

**Chunking** is token-budgeted (~2000 tokens, `tiktoken`) and cuts only on
segment boundaries, with one segment of overlap. A chunk ending mid-utterance
hands the model half a sentence. Overlap costs a duplicate key point, which
deduplication removes; a point lost across a boundary is not recoverable.

**Map** analyses chunks concurrently under a semaphore.

**Reduce** folds results in batches until one remains. A single reduce call over
every chunk summary would recreate the problem the brief asks us to solve, one
level up. Folding in batches bounds prompt size by `reduce_batch_size` rather
than by video length.

*Measured:* a 15-minute two-speaker transcript (180 segments) produces 7 chunks.
The largest single prompt is **19.9%** of the full transcript.

Every LLM call uses OpenAI Structured Outputs with Pydantic schemas, which
constrains generation to the supplied JSON Schema. Missing keys and invalid types
become impossible rather than retried around.

Each key point carries the timestamp of the transcript line supporting it. This
is an anti-fabrication device before it is a feature: a grounded claim is easier
to produce than an invented one when the model must cite where it came from.

**Token encoding is `o200k_base`.** `tiktoken` maps the gpt-5 family there;
`cl100k_base` is the GPT-4/3.5 vocabulary and would miscount every chunk.
`tiktoken` downloads its vocabulary over the network on first use, so token
counting falls back to a conservative character estimate when that is
unavailable — the estimate runs high, because over-counting shrinks chunks while
under-counting overruns the budget.

### Failure handling:


**Fatal** — ingestion, audio, transcription. Without a transcript there is no
partial result, and the error is the honest answer.

**Non-fatal** — everything after a successful transcription. A transcript is real
work and no later failure may discard it.

| Code | HTTP | Meaning |
|---|---|---|
| `INVALID_URL` | 400 | Malformed or disallowed scheme |
| `UNSUPPORTED_URL` | 400 | No extractor available |
| `INVALID_REQUEST` | 400 | Malformed body, or multipart with no file |
| `UNSUPPORTED_CONTENT_TYPE` | 415 | Body is neither JSON nor multipart |
| `SOURCE_UNAVAILABLE` | 422 | Removed, private, geo-blocked, bot-checked |
| `MEDIA_TOO_LONG` | 422 | Exceeds the duration cap |
| `MEDIA_TOO_LARGE` | 413 | Exceeds the size cap |
| `NO_AUDIO_STREAM` | 422 | Container has no audio track |
| `UNREADABLE_MEDIA` | 422 | ffprobe or FFmpeg cannot decode |
| `STT_FAILED` | 502 | Transcription failed after retries |
| `NO_SPEECH_DETECTED` | **200** | Pipeline succeeded; audio contains no speech |
| `ANALYSIS_FAILED` | **200** | Transcript preserved; analysis fields null |

The two `200`s are the design statement. Silent audio returns an empty transcript
and `summary: null` — a correct answer, not an error. A failed analysis returns
the full transcript with `summary: null`, empty arrays, and a populated
`errors[]`.

`summary` is always serialised, never omitted. A caller must be able to
distinguish *we could not produce this* from *this API does not return
summaries*.

The analysis stage catches broadly, including exceptions it does not anticipate.
That is deliberate: an unhandled error in summarisation must not turn a
successful transcription into a `500`. The exception is logged with a full
traceback, so a real bug stays visible while the caller keeps what succeeded.

`degraded_reasons[]` accompanies the `degraded` flag. Segment overlap during
crosstalk and an analysis built from partial coverage set the same flag and mean
entirely different things.

---

## Parallelism

Concurrency exists at two levels, bounded independently.

**Across requests.** The API is async end to end. yt-dlp runs in a worker thread,
FFmpeg through `asyncio.create_subprocess_exec`, and both providers over async
HTTP. Nothing blocks the event loop. Each request owns a temporary workspace,
removed in a `finally` block — failure paths leak scratch space as readily as
success paths.

**Within a request.** The map stage runs chunks concurrently under
`asyncio.Semaphore(map_concurrency)`, default 5. Reduce folds batches
concurrently under the same bound. Retries use exponential backoff with **full
jitter**: without randomisation, concurrent retries wake simultaneously and
collide again, turning one rate-limit response into a synchronised herd.

`504` is never retried. Deepgram's ten-minute limit is on *processing* time, so a
retry spends another ten minutes to reach the same answer. It signals that the
chunking fallback is needed, not a transient fault.

**These bounds multiply.** Five concurrent videos at `map_concurrency=5` is up to
25 simultaneous LLM calls. Deepgram allows 100 concurrent requests per project,
so it is not the ceiling; OpenAI rate limits are reached first. Every retry is
logged with its stage, so provider pressure is measurable rather than absorbed
silently:

```
retrying after failure │ label=map:chunk-3 attempt=1 of=3 error_type=RateLimitError
```

Lower `ANALYSIS__MAP_CONCURRENCY` when that appears.

No distributed framework is used. The workload is bounded concurrency over
external I/O; Celery, Redis, or a task queue would add operational surface
without addressing the actual constraint.

---

## API

### `POST /analyze-video`

Dispatches on `Content-Type`. `application/json` with a `url` field, or
`multipart/form-data` with a `file` part.

Response (abridged, from a real run):

```json
{
  "status": "success",
  "title": "Légionellose en Savoie : trois décès et 46 cas recensés depuis juin｜TF1 INFO",
  "duration": 113.941,
  "source": "Youtube",
  "language": { "code": "fr", "confidence": 0.994, "confidence_is_meaningful": true },
  "number_of_speakers": 3,
  "transcript": [
    {
      "start": 0.24,
      "end": 18.185,
      "speaker": "SPEAKER_01",
      "text": "En Savoie, une 3e victime de la légionellose...",
      "speaker_confidence": 0.886
    }
  ],
  "summary": "Le reportage signale une flambée de légionellose en Savoie...",
  "key_points": ["46 cas de légionellose depuis le début du mois de juin.", "..."],
  "topics": ["Legionellosis outbreak", "Public health investigation"],
  "stages": {
    "ingestion": "ok", "audio": "ok", "transcription": "ok",
    "diarization": "ok", "analysis": "ok"
  },
  "errors": [],
  "degraded": false,
  "degraded_reasons": [],
  "provenance": {
    "resolved_model": "general-nova-3",
    "diarizer_arch": "v2",
    "map_model": "gpt-5-mini",
    "reduce_model": "gpt-5",
    "transcription_cached": true,
    "chunk_count": 1,
    "failed_chunks": 0
  }
}
```

`status` is one of `success`, `partial_success`, `no_speech`.

`provenance` records what actually ran. Deepgram silently downgrades the model
when a detected language is unavailable, and `diarize_model=latest` resolves to
whatever is current — without this, an unexpected result has no explanation.
`chunk_count` above 1 is direct evidence the transcript was never sent in a
single prompt.

### `GET /health`

Reports liveness and whether each credential is configured. Reachable with no
credentials set, so a fresh clone can be started and inspected before any key
exists.

---

## Evaluation

Five real videos plus four failure cases, submitted concurrently against a
running server. Reproduce with:

```bash
uv run python evaluation/run_suite.py evaluation/manifest.json --concurrency 3
```

| Case | Route | Duration | Lang | Speakers | Expected | Segments | Status |
|---|---|---|---|---|---|---|---|
| French TV news | url | 113.9 s | fr | 3 | 6 | 10 | success |
| French political interview | url | 718.2 s | fr | 3 | 3 | 84 | success (degraded) |
| English single speaker | url | 182.8 s | en | 1 | 1 | 5 | success |
| English, noisy, two speakers | url | 914.8 s | en | 2 | 2 | 197 | success |
| French news, **file upload** | upload | 609.8 s | fr | 2 | 2 | 26 | success |
| Disallowed scheme | url | — | — | — | — | — | `400 INVALID_URL` |
| Nonexistent video | url | — | — | — | — | — | `422 SOURCE_UNAVAILABLE` |
| Silent audio | upload | — | — | 0 | 0 | 0 | `200 no_speech` |
| Text file named `.mp4` | upload | — | — | — | — | — | `422 UNREADABLE_MEDIA` |

**Speaker count: 4 of 5 exact.** The single miss is the TV news package, analysed
below.

**Language detection: 5 of 5**, across French and English, with no language hint
supplied.

**Concurrency.** Batch wall clock 339.6 s against 676.8 s of summed request time.
The longest single case ran 307.1 s, which is the floor for any batch — no amount
of concurrency finishes faster than its slowest member. The run reached **90% of
that floor**.

Processing costs roughly 0.2–0.35× video duration under concurrent load, so a
15-minute video completes in three to five minutes.

---

### Other

- **Arabic is out of scope.** Language detection covers 35 languages, not
  including Arabic. `language=multi` code-switching covers 10, also excluding it.
  `language_confidence` is meaningless outside the supported set, so the response
  carries `confidence_is_meaningful` alongside the score.
- **No speaker identity.** Labels are `SPEAKER_01`, no names.
- **URL ingestion fetches metadata before downloading** to reject over-long
  videos without transferring them, which costs an extra round trip.

---

## Testing

```bash
uv run pytest -q                  # no API keys required for testing
uv run pytest -m integration      # live provider tests, keys required (openai and deepgram both offer free tier plans)
uv run ruff check . && uv run mypy app
```

The unit suite makes no network calls. Deepgram is exercised through
`httpx.MockTransport`, which runs real request construction, retry logic, and
status handling against a fake transport. Media fixtures are generated with
FFmpeg at test time. An autouse fixture strips application environment variables
and ignores any developer `.env`, so results are identical on a fresh clone and
on a configured machine.

---

## Configuration

All settings carry working defaults. Only the two API keys are required.

| Variable | Default | Purpose |
|---|---|---|
| `DEEPGRAM_API_KEY` | — | Required for transcription |
| `OPENAI_API_KEY` | — | Required for analysis |
| `DEEPGRAM__MODEL` | `nova-3-general` | Speech model |
| `DEEPGRAM__DIARIZE_MODEL` | `latest` | Diarizer version; `v1` for broadcast audio |
| `DEEPGRAM__LANGUAGE_MODE` | `detect` | `detect`, `fixed`, or `multi` |
| `DEEPGRAM__DETECT_CANDIDATES` | `["en","fr"]` | Restricted detection set |
| `AUDIO__DENOISE` | `false` | Off pending measurement |
| `SEGMENTATION__PAUSE_THRESHOLD_SEC` | `0.7` | Segment boundary on silence |
| `SEGMENTATION__SMOOTHING_ENABLED` | `false` | Disabled on evidence |
| `ANALYSIS__MAP_MODEL` | `gpt-5-mini` | Per-chunk analysis |
| `ANALYSIS__REDUCE_MODEL` | `gpt-5` | Aggregation |
| `ANALYSIS__MAP_CONCURRENCY` | `5` | Concurrent LLM calls per request |
| `CACHE__ENABLED` | `true` | Response cache by audio hash |

See `.env.example` for the full set and `SPEC.md` for the decision log.