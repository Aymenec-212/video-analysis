# Evaluation results

Generated from `evaluation/run_suite.py`. 7 cases, 339.6s wall clock.

| Case | Route | HTTP | Status | Speakers | Err | Segments | Summary | Degraded | Time |
|---|---|---|---|---|---|---|---|---|---|
| news-multi-speaker | url | 200 | success | 3 | -3 | 10 | yes | no | 55.1s |
| interview-long-form | url | 200 | success | 3 | +0 | 84 | yes | yes | 166.6s |
| single-speaker-clean | url | 200 | success | 1 | +0 | 5 | yes | no | 32.4s |
| noisy-background | url | 200 | success | 2 | +0 | 197 | yes | no | 307.1s |
| upload-same-video | upload | 200 | success | 2 | -1 | 26 | yes | no | 114.6s |
| failure-invalid-url | url | 400 | error | - | - | 0 | no | no | 0.0s |
| failure-unavailable | url | 422 | error | - | - | 0 | no | no | 0.8s |

## Per-case notes

### news-multi-speaker
*French TV news package. Six speakers in 114s, four appearing once for under 12s each, across studio, street and field audio. The hardest diarization case in the set.*

- Route: URL
- HTTP 200, status `success`
- Title: Légionellose en Savoie : trois décès et 46 cas recensés depuis juin｜TF1 INFO
- Duration: 113.9s
- Speakers detected: 3 (expected 6, error -3)
- Diarization confidence: min 0.855, median 0.961
- Provenance: model `general-nova-3`, diarizer `v2`, cached `True`

### interview-long-form
*French political interview, ~12 minutes. Exercises LLM chunking and hierarchical reduce, plus rapid crosstalk between host and guest.*

- Route: URL
- HTTP 200, status `success`
- Title: Échange entre Bruno Retailleau et Lilia Bouziane autour du port du voile｜LCI
- Duration: 718.2s
- Speakers detected: 3 (expected 3, error +0)
- Diarization confidence: min 0.424, median 0.910
- Provenance: model `general-nova-3`, diarizer `v2`, cached `True`
- Degraded: 2 segment boundary defect(s) (overlapping), typically overlapping speech

### single-speaker-clean
*One speaker, clean studio audio. Baseline transcription quality with diarization unambiguous.*

- Route: URL
- HTTP 200, status `success`
- Title: How Meta's plan to restructure teams with AI imploded
- Duration: 182.8s
- Speakers detected: 1 (expected 1, error +0)
- Diarization confidence: min 0.717, median 0.762
- Provenance: model `general-nova-3`, diarizer `v2`, cached `True`

### noisy-background
*Background noise or music under speech. Robustness of STT and diarization; also the sample the denoising comparison is measured on.*

- Route: URL
- HTTP 200, status `success`
- Title: Rhett & Link Answer The Web's New Most Searched Questions | WIRED
- Duration: 914.8s
- Speakers detected: 2 (expected 2, error +0)
- Diarization confidence: min 0.116, median 0.737
- Provenance: model `general-nova-3`, diarizer `v2`, cached `False`

### upload-same-video
*The interview submitted as a multipart file upload rather than a URL, so both ingestion routes are covered by the suite.*

- Route: file upload
- HTTP 200, status `success`
- Title: test_sample
- Duration: 609.8s
- Speakers detected: 2 (expected 3, error -1)
- Diarization confidence: min 0.759, median 0.981
- Provenance: model `general-nova-3`, diarizer `v2`, cached `False`

### failure-invalid-url
*Disallowed scheme. Expects 400 INVALID_URL.*

- Route: URL
- HTTP 400, status `error`
- Error `INVALID_URL`: Only http and https URLs are accepted.

### failure-unavailable
*Nonexistent video. Expects 422 SOURCE_UNAVAILABLE.*

- Route: URL
- HTTP 422, status `error`
- Error `SOURCE_UNAVAILABLE`: The video could not be retrieved: it may be private, removed, region-restricted, or protected by a bot check.
