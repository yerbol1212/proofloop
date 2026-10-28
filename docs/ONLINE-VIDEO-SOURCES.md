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

## New source episodes reserved for a frozen follow-up

Episodes **000004 and 000005** were subsequently acquired from the same pinned ALOHA dataset revision, adding **20,276,528 bytes**. Both are original 22-second, 640 × 480 recordings. Their eight selected frames at the same 0.500 / 1.500 and 20.000 / 21.000 second checkpoints were individually reviewed before inference, retaining the original calibration rectangles and placement labels.

| Episode | Bytes | SHA-256 |
| --- | ---: | --- |
| 000004 | 9,969,665 | `df615acc2f5b6955e9f5d94dbac3946a8d5e752b9d409981ebf5562ccd2ed5dd` |
| 000005 | 10,306,863 | `db802ba1036c19d81a3c631ca5eeea67c37b01196e568ba935d1d4b26423e19a` |

The new draft and independent annotation freeze are saved locally as `contract-draft-new-episodes.json` and `new-episodes-label-review.json` under the online-review folder. Concept selection must finish on development episode 000000 before this follow-up contract is approved or tested. The draft concept is a placeholder until then; it is not evidence that an alternative prompt works. The original eight-case manifest remains unchanged.

These are new source episodes withheld from model testing during prompt selection, but their near-identical camera, objects, and task support only a narrow transfer check. They do not establish generalization to a different environment or procedure. Total acquired MP4 media across both batches is **80,895,261 bytes**, below the initial 150 MB cap.

## Other sources checked

- **IndustReal:** [the TU Eindhoven data record](https://research.tue.nl/en/datasets/industreal-dataset-of-egocentric-videos-for-procedure-understandi/) explicitly lists Apache-2.0 for the dataset. Its linked 4TU publisher page timed out during this acquisition; no IndustReal video was downloaded. The [authors’ project](https://timschoonbeek.github.io/industreal.html) remains a useful source for the need to recognize correct completion rather than merely recognize an action.
- **Full CaptainCook4D recordings:** the first 12 GoPro 360p Box links from the [official downloader metadata](https://github.com/CaptainCook4D/downloader) returned HTTP 404 on 2026-10-28. We did not substitute an unknown mirror or bypass a consent gate.

## Reuse obligations

Keep dataset credit, applicable copyright and license notices, and a clear statement of authored labels and any transformations with public demo excerpts. The reviewed dataset cards add no restriction prohibiting inference through a third-party API; this selection is based on their explicit permissive data licenses. No gated access, YouTube ripping, or unknown-rights footage was used. Raw media remains outside Git. A demo must identify these recordings as external source assets and show only ProofLoop functionality built during the hackathon as the team's contribution.
