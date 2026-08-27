# Evaluation results

Generated from `evaluation/run_suite.py`. 5 cases, 934.5s wall clock.

| Case | Route | HTTP | Status | Lang | Speakers | Err | Segments | Chunks | Summary | Degraded | Time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| news-multi-speaker | upload | 200 | success | fr | 3 | -3 | 10 | 1 | yes | no | 26.2s |
| interview-long-form | upload | 200 | success | fr | 3 | +0 | 86 | 3 | yes | yes | 132.0s |
| single-speaker-clean | upload | 200 | success | en | 2 | +1 | 8 | 1 | yes | no | 34.3s |
| noisy-background | upload | 200 | success | en | 3 | +1 | 198 | 4 | yes | yes | 155.9s |
| upload-news-two-speaker | upload | 0 | ? | - | - | - | 0 | - | no | no | 900.2s |

## Per-case notes

### news-multi-speaker
*French TV news package. Six speakers in 114s, four appearing once for under 12s each, across studio, street and field audio. The hardest diarization case in the set.*

- Route: file upload
- Language detected: `fr` at 0.994
- HTTP 200, status `success`
- Title: news-multi-speaker
- Duration: 113.9s
- Speakers detected: 3 (expected 6, error -3)
- Diarization confidence: min 0.556, median 0.835
- Provenance: model `general-nova-3`, diarizer `v2`, cached `True`
- Analysis: 1 excerpt(s) 

### interview-long-form
*French political interview, ~12 minutes. Exercises LLM chunking and hierarchical reduce, plus rapid crosstalk between host and guest.*

- Route: file upload
- Language detected: `fr` at 0.996
- HTTP 200, status `success`
- Title: interview-long-form
- Duration: 718.2s
- Speakers detected: 3 (expected 3, error +0)
- Diarization confidence: min 0.323, median 0.930
- Provenance: model `general-nova-3`, diarizer `v2`, cached `True`
- Analysis: 3 excerpt(s) - transcript was never sent in one prompt
- Degraded: 2 segment boundary defect(s) (overlapping), typically overlapping speech

### single-speaker-clean
*One English speaker, clean studio audio. Baseline transcription quality with diarization unambiguous, and the English half of the language-detection check.*

- Route: file upload
- Language detected: `en` at 0.997
- HTTP 200, status `success`
- Title: single-speaker-clean
- Duration: 182.8s
- Speakers detected: 2 (expected 1, error +1)
- Diarization confidence: min 0.298, median 0.980
- Provenance: model `general-nova-3`, diarizer `v2`, cached `True`
- Analysis: 1 excerpt(s) 

### noisy-background
*Two speakers over background noise and music, ~15 minutes. Robustness of STT and diarization, and the longest content in the set.*

- Route: file upload
- Language detected: `en` at 0.974
- HTTP 200, status `success`
- Title: noisy-background
- Duration: 914.8s
- Speakers detected: 3 (expected 2, error +1)
- Diarization confidence: min 0.153, median 0.774
- Provenance: model `general-nova-3`, diarizer `v2`, cached `True`
- Analysis: 4 excerpt(s) - transcript was never sent in one prompt
- Degraded: 1 segment boundary defect(s) (overlapping), typically overlapping speech

### upload-news-two-speaker
*A two-speaker French news segment submitted as a multipart upload rather than a URL, so both ingestion routes are covered. Upload titles derive from the filename, since an uploaded file carries no source metadata.*

- Route: file upload
- HTTP 0, status `?`
