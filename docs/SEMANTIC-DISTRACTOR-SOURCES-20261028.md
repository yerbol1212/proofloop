# Real video distractor source review — 28 October 2026

These are new, rights-declared online recordings for a **manually anchored semantic stress test**. A human selected real objects already present in the footage before any challenger inference. The test asks whether a candidate identity check withholds support from the wrong object while retaining support for the true target. It does not test automatic localization, placement accuracy, robot failure detection, or physical recovery.

The frozen local review is `artifacts/local/semantic-distractor-review-20261028/frozen-review-manifest.json`, SHA-256 `cba501f6226700d744fa4044671439c79c12feb1272311aabb8909816654ec5e`. The manifest contains source/frame/crop hashes, the exact source pixel boxes, reviewed contracts and labels. It was written before any semantic inference on these cases. Raw downloaded video and frames remain ignored local inputs.

## Selected sources and rights

| Source | Pinned revision | Dataset card license | Video SHA-256 | Bytes |
| --- | --- | --- | --- | ---: |
| [LeRobot screwdriver](https://huggingface.co/datasets/lerobot/aloha_static_screw_driver/blob/51448b4e7638c1773fd07c1e7febce8ebb104635/README.md), overhead episode 0 | `51448b4e7638c1773fd07c1e7febce8ebb104635` | MIT | `a64657a71c4819894cf913ae783656ca2a103e98de38d3d7063b785b85c46b36` | 3,065,319 |
| [LeRobot battery](https://huggingface.co/datasets/lerobot/aloha_static_battery/blob/dd8bda435f5068a0298bd109005828768ecbfa12/README.md), overhead episode 0 | `dd8bda435f5068a0298bd109005828768ecbfa12` | Apache-2.0 | `3de96889703429c1b9270e953a264d23bf9d45405138843fb12c782b206aa3c0` | 4,234,064 |

Both source files are at `videos/chunk-000/observation.images.cam_high/episode_000000.mp4` relative to the pinned dataset revision. Build the download URL as `https://huggingface.co/datasets/{dataset}/resolve/{revision}/{path}`. The exact pinned cards, dataset metadata, download receipts and file hashes are saved under `inputs/local/semantic-distractors-20261028/`. Dataset card licensing is the publisher's declaration; it is recorded without claiming any additional independent rights audit.

Both source frames are 640 × 480. Each pair uses exact video timestamps **0 and 1,000 ms**. Frames were decoded with FFmpeg 9.0 to JPEG at original resolution (`-q:v 2 -pix_fmt yuvj444p -threads 1 -map_metadata -1`). No objects were inserted, removed or composited. Candidate review crops are exact rectangles used for human QA; the challenger creates its own declared context crop from each source frame.

## Labels frozen before inference

Coordinates are inclusive source pixels `(x1, y1, x2, y2)`. The same listed box was reviewed separately in both frames. A negative identity label expects the diagnostic to remain `unknown`; that means identity support is withheld, not that placement failed.

| Case | Source | Reviewed candidate box | Human identity label | Expected diagnostic decision |
| --- | --- | --- | --- | --- |
| `screwdriver-target` | screwdriver e00 | `(308, 348, 432, 376)` | True white-handle screwdriver | `supported` |
| `screwdriver-tape` | screwdriver e00 | `(263, 280, 389, 296)` | Excluded flat white tabletop tape strip | `unknown` |
| `screwdriver-cup` | screwdriver e00 | `(169, 47, 260, 156)` | Excluded real blue-and-white cup | `unknown` |
| `remote-target` | battery e00 | `(214, 240, 338, 284)` | True rounded white remote body with open battery compartment | `supported` |
| `remote-tape` | battery e00 | `(263, 284, 386, 294)` | Excluded flat white tabletop tape strip | `unknown` |
| `remote-battery` | battery e00 | `(382, 269, 403, 294)` | Excluded detached cylindrical battery | `unknown` |

The screwdriver contract requires a wider solid white handle joined to a thin metallic shaft, forming one elongated hand tool. It excludes flat tabletop tape, a hollow drinking container and a two-finger robotic gripper. The remote contract requires a rounded oblong white plastic body with a dark rectangular battery recess. It excludes tabletop tape, a separate loose battery and a robotic gripper. These are visual attributes; the test does not verify screwdriver function, remote electronics or battery type. Both contracts require one full-scene matching target and full target visibility.

The tape negatives are useful because the source scene contains a real thin bright strip close to the true tool or remote. The cup and loose battery provide visibly different physical object controls. The loose battery occupies only 22 × 26 source pixels; a model may correctly withhold detailed recognition. A vague or incomplete response must never become semantic support.

## Screening and exclusions

Five bounded episode downloads total **21,116,190 bytes (20.14 MiB)**. Three were screened out before inference:

| Dataset | Pinned revision | License | Exclusion reason |
| --- | --- | --- | --- |
| [fork pickup](https://huggingface.co/datasets/lerobot/aloha_static_fork_pick_up/blob/1d4b7d9508e3b782d557fe5fb50d015503f01752/README.md) | `1d4b7d9508e3b782d557fe5fb50d015503f01752` | Apache-2.0 | The fork extends outside the lower frame edge before grasp. It cannot support an honest full-visibility positive label. |
| [cups open](https://huggingface.co/datasets/lerobot/aloha_static_cups_open/blob/fc9f40db2aba836a3005f7908652b85aa2a2c6e1/README.md) | `fc9f40db2aba836a3005f7908652b85aa2a2c6e1` | MIT | The tiny nested transparent stack seen from above produces weak identity/visibility labels. |
| [pencil](https://huggingface.co/datasets/lerobot/aloha_static_pro_pencil/blob/69ff95c9a8fd3b927f9ba355b6fd816775ce783f/README.md) | `69ff95c9a8fd3b927f9ba355b6fd816775ce783f` | MIT | Wide overhead view makes the target small; the gripper covers it by the second candidate frame. |

No model calls were used for acquisition, source screening, cropping or author labeling. The six selected cases reuse **two recordings and two frame pairs**, so they are correlated. They are deliberately authored, exposed development controls, not six independent real-world failures or a random holdout.

The earlier candy packet/label case is a useful separate regression: `artifacts/local/semantic-review-20261028-01/candy-e00-early-unsupported/prepared.json` binds genuine saved Spark locator anchors at 500/1,500 ms. Keep that cohort distinct from these manually authored candidates.

## Actual inference after the source review

The frozen six-case global-context trial completed all six requests, retained both actual targets and withheld three of four distractors. It incorrectly supported the loose battery as the remote body. The result remains [published separately](../artifacts/feasibility/semantic-v2-distractors-20261028.json), with 5/6 authored label agreement and one false support.

A prospectively frozen crop-only development retest then examined all three globally supported candidates. It retained both actual targets and withheld the battery. That local uncertainty does not positively identify the battery's type and does not establish full-scene uniqueness. The original source review, global-context failure and separate [crop-confirmation result](../artifacts/feasibility/semantic-crop-confirmation-20261028.json) remain distinguishable. See [method and costs](SEMANTIC-V2-RESULTS-20261028.md) for the complete account.
