# Local video decoder check — October 26, 2026

**Passed:** actual FFmpeg 9.0 / ffprobe 9.0 decoding and extraction on generated video. This is a local media-ingestion check. It makes **no phone-camera, Meta-model or physical-repair claim** and used **zero model requests**.

| Case | Source | Extracted frames | Verified result |
| --- | --- | --- | --- |
| Scheduled checkpoints | 48 seconds, 320 × 180, 30 fps | 8 | Four pairs at 3/5, 16/18, 29/31 and 42/44 seconds; actual decoded timestamps retained. |
| Display rotation | Same generated content with a 90° display matrix | 8 | Preflight reports 180 × 320 display dimensions; actual JPEGs are 180 × 320. |
| Off-grid requests | Requests 0.017 seconds after the scheduled times | 8 | Returned frame PTS differ from the requested offsets; actual values are stored, including 3.033333 seconds for a 3.017 second request. |
| HD downscaling | 4 seconds, 1920 × 1080, 30 fps | 2 | Actual JPEGs are 1280 × 720 and below the hosted model's 5 MiB input limit. |

All original-source and extracted-frame SHA-256 hashes, JPEG dimensions, byte counts and stage-window bounds were checked against the local files. The extraction tool does not assume a seek request is the actual frame timestamp. PTS are normalized to the first video frame; they do not establish wall-clock capture time.

The initial integration exposed FFmpeg 9.0's removal of the legacy `-vsync` option. The extractor now uses `-fps_mode passthrough`; the failed attempt stayed clearly incomplete in ignored local storage.

The generated `testsrc2` patterns contain no tools or physical scene. Cases share generated source content. Stage IDs are timing placeholders, and no tool-state annotations were validated. Raw videos, JPEGs and complete capture manifests remain local and ignored by Git. The public metadata record is [capture-decoder-20261026.json](capture-decoder-20261026.json).
