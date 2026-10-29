# Reproduce the new-source pilot preparation

This script prepares inputs for the three-recording pilot. It does not run Meta models, load API credentials, approve a manifest, choose labels or claim a physical repair. The nine image pairs belong to three recordings from two original dataset families; they are not nine independent videos.

Run the default command from the repository root to print the complete pinned plan. It performs no network requests, subprocess calls or writes:

```sh
.venv/bin/python scripts/prepare_new_source_pilot.py
```

Explicit download and preparation require FFmpeg and ffprobe, discovered in `.venv/bin` or `PATH`. No Pillow, Node or additional Python package is required:

```sh
.venv/bin/python scripts/prepare_new_source_pilot.py --download \
  --output inputs/local/new-source-pilot-reproduction-20261029
```

The output must be a **new directory strictly within this checkout's `inputs/local`**, with no symlink aliases. Existing directories and files are never adopted or overwritten. A failure can leave a partial new output directory; only `preparation-complete.json` denotes completed preparation. Use another fresh output directory for a retry. Existing acquired pilot inputs and model evidence remain separate.

## Fixed sources and attribution

| Recording | Redistributor revision | Exact video SHA-256 | Decoded frames |
| --- | --- | --- | --- |
| `bridge-scripted-0` | `VyoJ/BridgeData-V2-Scripted-Videos` at `d6c6064ca8aca851cae1b4f05e72732177490420` | `e1c8e5f7942ae4bdab72c897b5fa09a53dcc6099b0bea64e5bacb52675085161` | 50 |
| `ur5-cloth-0` | `lerobot/berkeley_autolab_ur5` at `6306aa91c00b9b0de28831b7fceaa42b8096bb4f` | `ae5da22ea02178c9336ff04473bbd2c2c0794f1826e2805193de13193ca8a26f` | 71 |
| `ur5-cup-4` | Same pinned UR5 revision, episode 4 | `2759938cf27fc003f68d01611095f168f95ae6e313c92e49dc2dd3e997adb9be` | 123 |

Each source URL, exact byte count, card URL and card SHA-256 is embedded in the script's printed plan. Downloaded cards are preserved alongside the videos. Downloads use bounded HTTPS reads with a fixed Hugging Face/CDN redirect policy, exact byte/hash verification and a monotonic preparation deadline. Media processes have separate duration and output bounds. The script does not accept arbitrary source URLs or execute downloaded content.

The [BridgeData V2 primary project page](https://rail-berkeley.github.io/bridgedata/) states that all its data is provided under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The [pinned derived Bridge card](https://huggingface.co/datasets/VyoJ/BridgeData-V2-Scripted-Videos/raw/d6c6064ca8aca851cae1b4f05e72732177490420/README.md) identifies that original project and preserves its licensing. Attribute BridgeData V2 to Walke et al., CoRL 2023, retain the original source and license links, and identify the video/JPEG derivation.

For UR5, the [pinned LeRobot redistribution card](https://huggingface.co/datasets/lerobot/berkeley_autolab_ur5/raw/6306aa91c00b9b0de28831b7fceaa42b8096bb4f/README.md) declares `cc-by-4.0` and links the [Berkeley UR5 original project](https://sites.google.com/view/berkeley-ur5/home). This is a pinned **redistributor declaration**. The original publisher's license was not independently verified in this preparation work; the script and receipts explicitly preserve that limitation. Source videos remain private inputs and are not added to the public repository by this script.

## Capture timing and pair selection

The Bridge MP4 encodes its 50 frames at **1 fps**. That playback clock is unsuitable for physical sample spacing. The script fetches only the first metadata JSONL row at the same pinned revision, requires its exact 66,906 bytes and SHA-256 `9fb8476f1d1ae70fdfeb94d677009ac70f69118f19dfb821595bfe61b11ce55a`, and verifies its video filename and 50-entry timestamp inventory. Relative original dataset capture times determine the pairs. For example, Bridge decoded frame 1 has 1,000 ms encoded PTS but 2,169 ms relative capture time.

UR5 pair spacing uses decoded presentation timestamps. These are distinct from independently verified physical capture times. Both clock bases are named explicitly in the receipts.

The unchanged deterministic schedule chooses start index zero, the source frame closest to `(last_source_time − 1000 ms) / 2`, and the latest frame at or before `last_source_time − 1000 ms`. Each end is the first later frame at least 1,000 source milliseconds after that start. Ties choose the earlier frame. Sampling rejects overlapping windows, repeated indices, nonmonotonic clocks and insufficient source coverage; it does not search for model success or replace failed cases.

| Recording | Fixed decoded-frame pairs |
| --- | --- |
| Bridge | `(0, 1)`, `(17, 23)`, `(44, 49)` |
| UR5 cloth | `(0, 5)`, `(32, 37)`, `(65, 70)` |
| UR5 cup | `(0, 5)`, `(58, 63)`, `(117, 122)` |

FFmpeg selects the original decoded index with `select=eq(n\,INDEX)` and `-fps_mode passthrough`; no frame-rate conversion, crop, resize or rotation is applied. Metadata is stripped from derived JPEGs. The [FFmpeg documentation](https://ffmpeg.org/ffmpeg.html) describes `passthrough` as preserving frame timestamps; the source-frame selection and timing receipt still supply the explicit mapping.

Each prepared receipt preserves the original video hash, probe hash, integer encoded PTS, stream time base, displayed PTS, decoded frame index, source/capture time, complete timing-map hash, selected JPEG hashes and tool versions. The receipt has a SHA-256 over its canonical content. These hashes bind the generated bytes and mapping; they do not attest API origin, capture truth or physical action. JPEG hashes are decoder/version dependent, so reproduction checks the pinned source and the same selected indices/times while separately recording the locally generated JPEG hashes. A changed image hash requires a separately reviewed experiment freeze rather than relabeling historical model results.

## Verification scope

Sixteen focused tests pass on Python 3.12 and Python 3.9. They cover a default plan with no effects, download and tool bounds, hashes, metadata-first-row reads, Bridge capture/playback separation, malformed timing, fixed pair selection, path confinement and refusal to overwrite. An integration check uses a newly authored six-frame test video, actual local FFmpeg/ffprobe, injected in-memory source bytes and no network or Meta calls. It verifies derived JPEG hashes, timing-map and receipt hashes, source-frame pairs and passthrough extraction. A separate read-only comparison against the already acquired pilot metadata/probes confirmed all nine existing pair indices and mapped timestamps without modifying those inputs.

The exact public `--download` command was also executed into a separate fresh private directory using FFmpeg/ffprobe 9.0. All three video/card pins and the Bridge metadata pin passed. All nine selected pairs, mapped timestamps and 18 JPEG byte hashes matched the frozen study inputs, with zero Meta calls. The [public result artifact](../artifacts/feasibility/new-source-results-20261029.json) includes allowlisted reproduction receipts and tool versions. This is a local reproduction check; another person's setup and usability remain untested.
