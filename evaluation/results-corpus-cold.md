# Evaluation results

Generated from `evaluation/run_suite.py`. 9 cases, 233.5s wall clock.

| Case | Route | HTTP | Status | Lang | Speakers | Err | Segments | Chunks | Summary | Degraded | Time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| news-multi-speaker | url | 200 | success | fr | 3 | -3 | 10 | 1 | yes | no | 46.2s |
| interview-long-form | url | 200 | success | fr | 3 | +0 | 83 | 3 | yes | yes | 155.8s |
| single-speaker-clean | url | 200 | success | en | 1 | +0 | 5 | 1 | yes | no | 48.8s |
| noisy-background | url | 200 | success | en | 2 | +0 | 197 | 4 | yes | no | 156.9s |
| upload-news-two-speaker | upload | 200 | success | fr | 2 | +0 | 118 | 5 | yes | yes | 184.6s |
| failure-silent-audio | upload | 200 | no_speech | en | 0 | +0 | 0 | - | no | no | 1.1s |
| failure-corrupt-media | upload | 422 | error | - | - | - | 0 | - | no | no | 0.1s |
| failure-invalid-url | url | 400 | error | - | - | - | 0 | - | no | no | 0.0s |
| failure-unavailable | url | 422 | error | - | - | - | 0 | - | no | no | 0.7s |

## Per-case notes

### news-multi-speaker
*French TV news package. Six speakers in 114s, four appearing once for under 12s each, across studio, street and field audio. The hardest diarization case in the set.*

- Route: URL
- Language detected: `fr` at 0.994
- HTTP 200, status `success`
- Title: Légionellose en Savoie : trois décès et 46 cas recensés depuis juin｜TF1 INFO
- Duration: 113.9s
- Speakers detected: 3 (expected 6, error -3)
- Diarization confidence: min 0.855, median 0.961
- Provenance: model `general-nova-3`, diarizer `v2`, cached `False`
- Analysis: 1 excerpt(s) 

### interview-long-form
*French political interview, ~12 minutes. Exercises LLM chunking and hierarchical reduce, plus rapid crosstalk between host and guest.*

- Route: URL
- Language detected: `fr` at 0.995
- HTTP 200, status `success`
- Title: Échange entre Bruno Retailleau et Lilia Bouziane autour du port du voile｜LCI
- Duration: 718.2s
- Speakers detected: 3 (expected 3, error +0)
- Diarization confidence: min 0.424, median 0.899
- Provenance: model `general-nova-3`, diarizer `v2`, cached `False`
- Analysis: 3 excerpt(s) - transcript was never sent in one prompt
- Degraded: 2 segment boundary defect(s) (overlapping), typically overlapping speech

### single-speaker-clean
*One English speaker, clean studio audio. Baseline transcription quality with diarization unambiguous, and the English half of the language-detection check.*

- Route: URL
- Language detected: `en` at 0.999
- HTTP 200, status `success`
- Title: How Meta's plan to restructure teams with AI imploded
- Duration: 182.8s
- Speakers detected: 1 (expected 1, error +0)
- Diarization confidence: min 0.752, median 0.797
- Provenance: model `general-nova-3`, diarizer `v2`, cached `False`
- Analysis: 1 excerpt(s) 

### noisy-background
*Two speakers over background noise and music, ~15 minutes. Robustness of STT and diarization, and the longest content in the set.*

- Route: URL
- Language detected: `en` at 0.982
- HTTP 200, status `success`
- Title: Rhett & Link Answer The Web's New Most Searched Questions | WIRED
- Duration: 914.8s
- Speakers detected: 2 (expected 2, error +0)
- Diarization confidence: min 0.116, median 0.737
- Provenance: model `general-nova-3`, diarizer `v2`, cached `False`
- Analysis: 4 excerpt(s) - transcript was never sent in one prompt

### upload-news-two-speaker
*A two-speaker French news segment submitted as a multipart upload rather than a URL, so both ingestion routes are covered. Upload titles derive from the filename, since an uploaded file carries no source metadata.*

- Route: file upload
- Language detected: `fr` at 0.997
- HTTP 200, status `success`
- Title: news_two_speaker
- Duration: 1258.7s
- Speakers detected: 2 (expected 2, error +0)
- Diarization confidence: min 0.259, median 0.814
- Provenance: model `general-nova-3`, diarizer `v2`, cached `False`
- Analysis: 5 excerpt(s) - transcript was never sent in one prompt
- Degraded: 2 segment boundary defect(s) (overlapping), typically overlapping speech

### failure-silent-audio
*Audio containing no speech. Expects 200 with status no_speech, an empty transcript and summary null - the no-fabrication requirement demonstrated on real audio rather than asserted.*

- Route: file upload
- Language detected: `en` at 0.000
- HTTP 200, status `no_speech`
- Title: silent
- Duration: 8.0s
- Speakers detected: 0 (expected 0, error +0)
- Provenance: model `general-nova-3`, diarizer `v2`, cached `False`
- Error `NO_SPEECH_DETECTED`: No speech was detected in the audio.

### failure-corrupt-media
*A text file carrying a video extension. Expects 422 UNREADABLE_MEDIA, proving validation is by probe rather than by filename.*

- Route: file upload
- HTTP 422, status `error`
- Error `UNREADABLE_MEDIA`: The media file could not be decoded.

### failure-invalid-url
*Disallowed scheme. Expects 400 INVALID_URL, rejected before any network call.*

- Route: URL
- HTTP 400, status `error`
- Error `INVALID_URL`: Only http and https URLs are accepted.

### failure-unavailable
*Nonexistent video. Expects 422 SOURCE_UNAVAILABLE.*

- Route: URL
- HTTP 422, status `error`
- Error `SOURCE_UNAVAILABLE`: The video could not be retrieved: it may be private, removed, region-restricted, or protected by a bot check.
