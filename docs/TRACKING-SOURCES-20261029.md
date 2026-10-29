# Tracking source review — 2026-10-29

Two additional real ALOHA demonstration episodes were acquired for a separately evaluated temporal tracking experiment. These recordings have not appeared in ProofLoop's earlier experiments. Their **dataset families and scene types were already exposed** in the semantic distractor experiments. They therefore support new-episode development tests, not an independent dataset-family holdout or a general reliability claim.

No model inference was performed during acquisition or this review. Root-level visual label review and a frozen experiment manifest are required before model calls. Raw videos, derivatives, source cards, tree listings, frame maps and review images remain in ignored local directories; this document supplies attribution and reproducibility information.

## Publisher declarations and pinned files

| Source | Immutable revision | Dataset-card license | Original SHA-256 | Original bytes |
| --- | --- | --- | --- | ---: |
| [LeRobot screwdriver, overhead episode 000001](https://huggingface.co/datasets/lerobot/aloha_static_screw_driver/blob/51448b4e7638c1773fd07c1e7febce8ebb104635/README.md) | `51448b4e7638c1773fd07c1e7febce8ebb104635` | MIT | `7ea053856501fc819f3f80b00c4d56553404cfc5e4a1340298db49f1893903fd` | 3,089,388 |
| [LeRobot battery/remote, overhead episode 000001](https://huggingface.co/datasets/lerobot/aloha_static_battery/blob/dd8bda435f5068a0298bd109005828768ecbfa12/README.md) | `dd8bda435f5068a0298bd109005828768ecbfa12` | Apache-2.0 | `5070edbfa507a04b6b59a05dacc2db6c10eefb015107146cb84d4285faa61315` | 4,229,583 |

Both cards declare the stated license in their YAML metadata and dataset description. This is the publisher's data-license declaration, not an additional independent rights audit. Credit Hugging Face / LeRobot for the dataset conversion and Tony Z. Zhao, Vikash Kumar, Sergey Levine and Chelsea Finn for the original ALOHA project, *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, RSS 2023. The screwdriver card supplies that citation; the battery card's citation section is unspecified. See the [original ALOHA project](https://tonyzhaozh.github.io/aloha/).

Acquisition used the explicitly saved pinned tree listings for `videos/chunk-000/observation.images.cam_high/`; episode paths were selected from those listings. Each download had to match both the listed byte count and Git LFS SHA-256. The tree-listed path is `videos/chunk-000/observation.images.cam_high/episode_000001.mp4` for both sources. The download form is `https://huggingface.co/datasets/{dataset}/resolve/{revision}/{tree-listed-path}`. No gated media, unknown-rights footage or YouTube extraction was used.

The two source downloads total **7,318,971 bytes**, below the 80 MiB acquisition ceiling. Dataset cards, metadata and license text add only small text files. The original ALOHA MIT notice and the [standard Apache-2.0 license text](https://www.apache.org/licenses/LICENSE-2.0.txt) are retained alongside the publisher declarations. The separate original code notice is not used as the sole basis for video rights.

## Exact video preparation

Both originals are AV1 video in MP4 containers, native 640 × 480, `yuv420p`, 50 fps. The screwdriver has 400 native frames over 8 seconds; battery/remote has 600 over 12 seconds. FFprobe's per-frame timestamps were checked against `native_frame_index / 50` with 1 microsecond tolerance; all passed.

AV1 is outside the formats listed by the current [SAM video segmentation guide](https://dev.meta.ai/docs/sam/segmenting), so the experiment candidates use explicit H.264 derivatives. These preserve native dimensions and selected decoded pixels:

```text
select=not(mod(n,10)),setpts=N/(5*TB)
libx264 -crf 0 -preset veryfast -pix_fmt yuv420p -r 5 -fps_mode cfr
-an -movflags +faststart
```

| Prepared candidate | Frames / duration | Prepared SHA-256 | Prepared bytes |
| --- | --- | --- | ---: |
| `aloha_static_screw_driver-episode-000001-tracking-5fps-lossless.mp4` | 40 / 8 seconds | `ec407e9ef83ba3fdd1a42202ede13433564c74e289ed48e358133c8e905e3f26` | 3,617,873 |
| `aloha_static_battery-episode-000001-tracking-5fps-lossless.mp4` | 60 / 12 seconds | `213adcd18342fbaa3d42e525e7bedcc54edfc745e498a300a8172c1f39039f7f` | 4,881,213 |

For every derivative frame `n`, the original source frame is exactly `10 × n` and its source timestamp is `200 × n` milliseconds. Every decoded H.264 frame's FFmpeg frame-MD5 was compared to the corresponding native AV1 decoded frame: **40/40 and 60/60 matched**. No crop, spatial resize, invented frame or interpolated motion was introduced. Metadata, exact frame maps and decoded hashes are retained privately. These are frame-selection derivatives, not additional independent source recordings.

Sampling at 5 fps leaves the nine intervening native frames between selected samples unobserved by the candidate experiment. A successful track must not become a claim of continuous real-world visibility. SAM's tracking object IDs also do not independently prove semantic identity, physical execution or functional correctness.

## Visual screening observations — proposed, not frozen truth

The review contains 16 native screwdriver stills at 500 ms spacing and 15 native battery/remote stills at 800 ms spacing. Full prepared-video contact sheets additionally show every selected 5 fps frame, with original source timestamps overlaid. All 100 selected tracking frames are also extracted at full size as `tracking-frame-%03d.jpg`; the one-based filename `tracking-frame-001.jpg` is clip frame zero. Use these and their manifest mapping when freezing labels: some older 500 ms native review stills are not in the 200 ms tracking sample. JPEG review images are diagnostic renderings; the H.264 derivative is the candidate inference input.

- **Screwdriver:** the initial full tool has a wider pale handle connected to a narrow shaft, near a separate flat bright tape strip. A blue-and-white cup is another visible object. The right gripper overlaps the handle as it approaches at about 1.4 seconds; the handle is substantially covered around 1.8–3.2 seconds. The tool is subsequently held between grippers and moved toward the cup. Several later views show only part of it or have overlapping grippers. This is useful for testing whether a track improperly acquires the tape or continues to imply full target visibility through obstruction. The final cup area must not receive an automatic “fully visible target” label merely because the track retained its ID.
- **Battery/remote:** the initial target is a rounded white body with a central dark battery recess. A detached small cylinder and a bright flat tape strip are nearby distractors. Gripper fingers overlap the body/recess during manipulation, especially around 4–9 seconds, then withdraw enough to show more of the body again near 10.8–11.8 seconds. Much of the body remains visible throughout; this is **partial occlusion and re-exposure**, not demonstrated total disappearance followed by reacquisition. The review must label the exact target outline and visibility at each scored checkpoint rather than infer whole intervals from a thumbnail.

Approximate frame-zero review rectangles in native pixels are screwdriver `(310,338,432,357)`, screwdriver-scene tape `(265,281,389,294)`, remote body `(224,244,344,286)` and detached cylinder `(383,273,401,289)`. These are screening suggestions, not approved prompt boxes. Root should inspect the full frames and freeze exact inclusive-coordinate boxes, checked frames and visibility labels before inference. This review does not assert that a remote works, a battery is inserted correctly, a screwdriver is functional, or a whole procedure succeeds.

## Private handoff locations

- Sources, publisher cards, saved tree listings and receipts: `inputs/local/tracking-20261029/`.
- Prepared-source manifest: `artifacts/local/tracking-review-20261029/prepared-sources.json`; select `sources[].tracking_video` for the intended 5 fps candidate, not the earlier full-frame-rate transcode.
- Screwdriver all-frame contact sheets: `artifacts/local/tracking-review-20261029/aloha_static_screw_driver-episode-000001/tracking-5fps-contact-page-1.jpg` and `tracking-5fps-contact-page-2.jpg`.
- Battery/remote all-frame contact sheets: the analogous `aloha_static_battery-episode-000001/` directory, pages 1–3.
- Each source subdirectory also contains its native timestamp list, exact tracking frame map, source/derivative decoded frame hashes and full-size review stills.

Before inference, freeze the exact prepared-video hashes, frame mappings, initial boxes, negative-control identity, checkpoint labels, missing-evidence policy, model version and request budget. Keep model outputs separate from authored labels, and preserve failures or unsupported completions rather than tuning thresholds on them and relabeling the same recordings as fresh evidence.
