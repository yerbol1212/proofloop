# Online video sources and evaluation scope

Reviewed and acquired on 2026-10-28. Raw videos, extracted frames, approvals, contact sheets, and license evidence stay in ignored local folders. The public repository contains attribution and reviewed summaries, not downloaded media.

## Active pilot: real ALOHA coffee demonstrations

Source: [LeRobot / ALOHA static coffee dataset](https://huggingface.co/datasets/lerobot/aloha_static_coffee). We pinned revision [`1c52c4c794b6f93d931489263cf15e4331600bfc`](https://huggingface.co/datasets/lerobot/aloha_static_coffee/tree/1c52c4c794b6f93d931489263cf15e4331600bfc), which retains small per-episode MP4 files. The current dataset revision bundles episodes into larger files.

These are real robot teleoperation demonstrations, not simulated images. The original research is [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://tonyzhaozh.github.io/aloha/), Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn, RSS 2023. Credit Hugging Face / LeRobot for the dataset conversion.

The **dataset card itself** explicitly declares MIT in both its metadata and dataset description; this is the basis for selecting the data. We do not infer video rights merely from the separate ALOHA code license. [Pinned data license statement](https://huggingface.co/datasets/lerobot/aloha_static_coffee/blob/1c52c4c794b6f93d931489263cf15e4331600bfc/README.md). The original ALOHA [MIT notice](https://github.com/tonyzhaozh/aloha/blob/main/LICENSE), copyright (c) 2023 Tony Z. Zhao, is also retained locally.

Four `cam_high` recordings were acquired. Each is 640 × 480 at 50 fps, duration 22 seconds. Together they occupy 39,885,785 bytes.

| Episode | Bytes | SHA-256 |
| --- | ---: | --- |
| 000000 | 9,964,446 | `02bc95dcc0922f5d7d759861e6e5151f4aa068631d5491e82badfc043f68cfd2` |
| 000001 | 9,842,148 | `20c258c99f463bd2451ab02b782f5f79bad3cba1bab7be7d45de6b43bb4f105e` |
| 000002 | 10,076,093 | `37d4930f315ac3a19288aa33b7446697be3e8a83cd31879674b40dbb8343d32f` |
| 000003 | 10,003,098 | `b10ceabf7c90ab2bea281965a2a9699b287e2f84e57dfb734dec07c8bd33ebd0` |

Direct download pattern, substituting the six-digit episode number:

```text
https://huggingface.co/datasets/lerobot/aloha_static_coffee/resolve/1c52c4c794b6f93d931489263cf15e4331600bfc/videos/chunk-000/observation.images.cam_high/episode_000000.mp4
```

### What the pilot measures

Authored placement gate: **the sole transparent drinking glass must occupy the calibrated dispense region under the coffee machine outlet.** The same glass visibly in the right staging region fails that placement gate. Missing detections, multiple candidates, uncertain masks, region crossings, or insufficient stable evidence remain unknown.

For each of four recordings, two frames at **0.500 and 1.500 seconds** form a staging case; two frames at **20.000 and 21.000 seconds** form a dispense case. All 16 selected frames were individually reviewed before any Meta inference on this corpus. Labels and calibration were frozen in one hashed contract before inference:

```text
dispense: [0.38, 0.18, 0.64, 0.46]
staging:  [0.66, 0.18, 0.87, 0.46]
contract SHA-256:
a80ecfc424e5ae7de2204a653192389a430a7e83f920bff3773d2a064d3d62f2
```

This produces **eight scoped cases across four source episodes: four authored fail and four authored pass**. Both conditions come from each episode; the eight cases are not eight independent recordings.

The early staging state is an ordinary intermediate state in a normal demonstration. Calling this placement gate “fail” does **not** mean the source robot made a procedural error. Its later motion was recorded independently of ProofLoop and is **not** a robot following our repair recommendation. These cases do not certify that the robot may dispense, that the glass is safely grasped, that the beverage is correct, or that the full procedure succeeded.

This is a small, deliberately selected development pilot, **not an official ALOHA benchmark, a held-out evaluation, a natural-error recovery benchmark, or evidence of general industrial reliability**. Keeping annotation independent of model outputs prevents one form of leakage; it does not make a selected pilot representative. No performance claim belongs in this source document; use the actual run report.

Private files:

```text
inputs/local/online/aloha_static_coffee-episode-000000.mp4
inputs/local/online/frames/coffee-e00-000500.jpg
artifacts/local/online-review/contract-draft.json
artifacts/local/online-review/manifest-approved.json
artifacts/local/online-review/provenance.json
```

The manifest binds each source MP4 and each extracted frame to its SHA-256 and exact source timestamp. Frames were decoded with FFmpeg without creative editing, compositing, generated content, or altered object placement. Review contact sheets are separate convenience artifacts and are not inference evidence.

## Acquired but excluded from pilot scores

- **Mobile ALOHA wash-pan, episode 000000.** Original real robot video, 22 seconds, 640 × 480, 10,980,189 bytes. [Pinned dataset card and MIT declaration](https://huggingface.co/datasets/lerobot/aloha_mobile_wash_pan/blob/11149fc00957a1998fa715ee77dc1486d739bfe9/README.md); [original Mobile ALOHA project](https://mobile-aloha.github.io/). Credit Zipeng Fu, Tony Z. Zhao, Chelsea Finn, and Hugging Face / LeRobot. The moving camera and pan occlusion require a separately reviewed contract. No label or inference currently certifies washing, rinsing, or cleanliness.
- **Five CaptainCook4D publisher example montages**, totaling 9,752,759 bytes. The project and annotation pages explicitly license the **dataset** under Apache-2.0 and report written participant consent. [Primary project](https://github.com/CaptainCook4D/CaptainCook4D), [dataset license and consent statement](https://github.com/CaptainCook4D/annotations#license--consent), [Apache-2.0 text](https://github.com/CaptainCook4D/annotations/blob/main/LICENSE). Credit Rohith Peddi and coauthors, *CaptainCook4D: A Dataset for Understanding Errors in Procedural Activities*, NeurIPS 2024. These videos show five source clips side by side; no original episode mapping is supplied in the montage MP4. They are illustrative material, not independent benchmark episodes. Any future panel crop must disclose its transformation and remain excluded from scored results unless separately justified.

The initial acquired MP4 batch is **60,618,733 bytes**. Exact direct URLs, hashes, license evidence, and roles for these files are in the ignored provenance record.

## Episodes 000004 / 000005: v1 follow-up, then v2 development

Episodes **000004 and 000005** were subsequently acquired from the same pinned ALOHA dataset revision, adding **20,276,528 bytes**. Both are original 22-second, 640 × 480 recordings. Their eight selected frames at the same 0.500 / 1.500 and 20.000 / 21.000 second checkpoints were individually reviewed before inference, retaining the original calibration rectangles and placement labels.

| Episode | Bytes | SHA-256 |
| --- | ---: | --- |
| 000004 | 9,969,665 | `df615acc2f5b6955e9f5d94dbac3946a8d5e752b9d409981ebf5562ccd2ed5dd` |
| 000005 | 10,306,863 | `db802ba1036c19d81a3c631ca5eeea67c37b01196e568ba935d1d4b26423e19a` |

The draft and independent annotation freeze are saved locally as `contract-draft-new-episodes.json` and `new-episodes-label-review.json` under the online-review folder. The initial draft's concept was a placeholder; the evaluated contract retained the original semantic concept, `transparent drinking glass`, after development-only probing. The original eight-case manifest remains unchanged.

These episodes were withheld from model testing during v1 selection, but their failures subsequently informed v2 development. Their later replay is therefore development evidence. Their near-identical camera, objects, and task support only a narrow transfer check; they do not establish generalization to another environment or procedure. Total acquired MP4 media at this stage was **80,895,261 bytes**.

## Episodes 000006 / 000007: fresh sources after the v2 fix

Two further original `cam_high` recordings were acquired from the same pinned revision and [explicit MIT dataset license](https://huggingface.co/datasets/lerobot/aloha_static_coffee/blob/1c52c4c794b6f93d931489263cf15e4331600bfc/README.md). The ALOHA authors and Hugging Face / LeRobot retain the same attribution. Each recording is 22 seconds at 640 × 480 and 50 fps.

| Episode | Bytes | SHA-256 |
| --- | ---: | --- |
| 000006 | 9,971,739 | `e5229682cc6ee275d13f6010a355c54428d8c9dd07ebb4681c107fc4d92fd91e` |
| 000007 | 9,959,554 | `cdef316e36fdbb74ca7ed6eeee64ef2bc4c0749e2d2734ed36397c1dbdcf5e9c` |

Both files' bytes and hashes were rechecked against the retained download records. Direct URLs follow the original pinned download pattern above. The eight frames use the same preselected **0.500 / 1.500** staging and **20.000 / 21.000** dispense checkpoints; no checkpoint or region rectangle changed after visual review. Labels describe supported visible placement only. In episode 000006 at 20 seconds, the gripper overlaps part of the glass rim, which may legitimately cause a stricter perception method to abstain; the label does not claim amodal completeness.

The source and assistant review records are in `guided-v2-holdback-provenance.json` and `guided-v2-holdback-label-review.json` under the ignored online-review folder. The source/label contract hash is:

```text
88a3772f85dba0749cbc2d6996740da204d03597ab42cb3eab642e083bac9d88
```

`algorithm-freeze-guided-v2.json` separately records the v2 module and locator schema hashes before inference. Keeping both freezes distinguishes previously untested source episodes from a replay used during development. The [completed comparison](../artifacts/feasibility/guided-fix-20261028.md) reports the actual results and remaining unknown. This remains a narrow same-task, same-camera transfer check.

The two new videos add **19,931,293 bytes**. Total acquired MP4 media across all batches is **100,826,554 bytes**, below the initial 150 MB cap. Raw media remains ignored and unpublished.

## Temporal-stability stress reuse of existing sources

The [temporal stress evaluation](../artifacts/feasibility/temporal-stress-20261028.md) reuses episodes **000000–000003** without acquiring additional videos or changing image content. Four inconsistent pairs use 0.500 / 20.000 second frames; two stable staging controls use 2.500 / 3.500 seconds from episodes 000000 / 000001; two stable dispense controls use 19.500 / 21.500 seconds from episodes 000002 / 000003. All 16 selected frame hashes are distinct within this suite and were visually reviewed and frozen before its inference run.

UNKNOWN labels refer to failure to establish the required two consistent location samples, rather than ignorance of the latest physical state. These are constructed **development** cases on already exposed sources; new pairings or newly extracted controls do not make them an independent holdback. The source license, attribution and calibrated regions remain unchanged. The approved contract hash is `bbfd2b963d398dd34b1b11072a1791860b9a2bb60427daff30132375c359472a`; private frame extraction, label review and approval records remain in the ignored online-review folder. See the linked report for actual results and limits.

## Other sources checked

- **IndustReal:** [the TU Eindhoven data record](https://research.tue.nl/en/datasets/industreal-dataset-of-egocentric-videos-for-procedure-understandi/) explicitly lists Apache-2.0 for the dataset. Its linked 4TU publisher page timed out during this acquisition; no IndustReal video was downloaded. The [authors’ project](https://timschoonbeek.github.io/industreal.html) remains a useful source for the need to recognize correct completion rather than merely recognize an action.
- **Full CaptainCook4D recordings:** the first 12 GoPro 360p Box links from the [official downloader metadata](https://github.com/CaptainCook4D/downloader) returned HTTP 404 on 2026-10-28. We did not substitute an unknown mirror or bypass a consent gate.

## Contact/tracking extension: 29 October 2026

The [contact protocol](CONTACT-PROTOCOL-20261029.md) adds two original ALOHA robot recordings distributed through LeRobot. Both pinned dataset cards were downloaded and their declared licenses checked before inference. These are external recordings, not team-created video or naturally annotated ProofLoop failures.

| Source | Declared license | Bytes | Original SHA-256 |
| --- | --- | ---: | --- |
| [Screwdriver episode 000003](https://huggingface.co/datasets/lerobot/aloha_static_screw_driver/resolve/51448b4e7638c1773fd07c1e7febce8ebb104635/videos/chunk-000/observation.images.cam_high/episode_000003.mp4) | [MIT card](https://huggingface.co/datasets/lerobot/aloha_static_screw_driver/blob/51448b4e7638c1773fd07c1e7febce8ebb104635/README.md) | 3,132,360 | `6b69b8cab6a8c0702f12fb7a270eda723aecf2f9671bca3f517c4e5d48562c90` |
| [Battery/remote episode 000003](https://huggingface.co/datasets/lerobot/aloha_static_battery/resolve/dd8bda435f5068a0298bd109005828768ecbfa12/videos/chunk-000/observation.images.cam_high/episode_000003.mp4) | [Apache-2.0 card](https://huggingface.co/datasets/lerobot/aloha_static_battery/blob/dd8bda435f5068a0298bd109005828768ecbfa12/README.md) | 4,161,469 | `2579796191bd0d20dec5d77463062357ef74b39b34227f2935b387b773acf9f3` |

The two files add 7,293,829 bytes. Native 640 × 480, 50-fps source frames are sampled every tenth frame and encoded losslessly as 5-fps H.264 without resizing or cropping. All 100 selected decoded frames and their extracted JPEGs replay exactly. Later boxes and clear/unclear labels are assistant-authored scoring references frozen before the contact-model trial; only reviewed frame-zero boxes initialize tracking. Later strict-mask and warning experiments reuse these now-exposed recordings and the prior episode-2 sources, without changing labels. The cached warning retest is selected after inspecting failures and is not held-out validation. Original media, masks and overlays stay private; public reports retain attribution, transformation records and hashes.

## New original source families: preregistered pilot

The [new-source pilot](NEW-SOURCE-RESULTS-20261029.md) uses three pinned recordings from two additional original families, totaling 4,342,810 video bytes. The [source-preparation guide](PROSPECTIVE-SOURCES-20261029.md) lists exact source revisions, video/card hashes, attribution and a reproducible download command. BridgeData V2 trajectory 0 comes from VyoJ's scripted-video conversion; the [original BridgeData V2 publisher](https://rail-berkeley.github.io/bridgedata/) explicitly licenses its data CC BY 4.0. Credit Walke et al., CoRL 2023, RAIL-Berkeley and the VyoJ conversion. Berkeley UR5 episodes 0 and 4 come from LeRobot's [pinned dataset card](https://huggingface.co/datasets/lerobot/berkeley_autolab_ur5/blob/6306aa91c00b9b0de28831b7fceaa42b8096bb4f/README.md), which declares CC BY 4.0. Credit Lawrence Yunliang Chen, Simeon Adebola, Ken Goldberg and Hugging Face LeRobot. The UR5 basis is the redistributor's declaration; the original publisher's license was not established independently.

The 18 original decoded views are converted to JPEG without resizing, cropping or rotation and remain private. Bridge selection uses original per-frame capture timestamps, separately retaining its slowed 1-fps playback PTS. UR5 uses native decoded presentation timestamps. Newly authored placement contracts and labels do not turn normal demonstration trajectories into publisher-annotated robot failures. Public reports contain hashes and results, without original source pixels.

## Reuse obligations

Keep dataset credit, applicable copyright and license notices, and a clear statement of authored labels and any transformations with public demo excerpts. The reviewed dataset cards add no restriction prohibiting inference through a third-party API; this selection is based on their explicit permissive data licenses. No gated access, YouTube ripping, or unknown-rights footage was used. Raw media remains outside Git. A demo must identify these recordings as external source assets and show only ProofLoop functionality built during the hackathon as the team's contribution.
