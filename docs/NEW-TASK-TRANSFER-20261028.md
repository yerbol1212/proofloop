# New task transfer: real ALOHA candy recording

ProofLoop's unchanged guided observation method returned **UNKNOWN → PASS** on two fixed pairs from one new candy-manipulation recording. Spark alone returned the same results. Both matched **2/2** authored labels. This is a small development transfer check, not evidence that the guided method outperforms Spark or generalizes reliably.

## Source and rights

The source is the primary [LeRobot / ALOHA static candy dataset card at the pinned revision](https://huggingface.co/datasets/lerobot/aloha_static_candy/blob/7a19ab84a6b7f376d699c7f12c28535725fd0593/README.md). That dataset card explicitly declares **MIT**; the separate ALOHA software license is not the basis for inferring video rights. Credit Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn, *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, RSS 2023, and Hugging Face / LeRobot for the dataset conversion.

One physical robot demonstration was selected by episode index before inference: `episode_000000`, camera `observation.images.cam_high`. The [pinned MP4](https://huggingface.co/datasets/lerobot/aloha_static_candy/resolve/7a19ab84a6b7f376d699c7f12c28535725fd0593/videos/chunk-000/observation.images.cam_high/episode_000000.mp4) contains 14 seconds at 640 × 480, 50 fps, without audio. It is 5,176,124 bytes, SHA-256 `8c17c51737f82d63ed9a605da08173c2db40678e193a07efb08f3c9055fcebe1`. The raw recording, pinned card, extracted images, approvals and responses remain in ignored local folders.

This adds **one recording from a different manipulation task and scene** to the coffee development data. It does not add independent evidence in proportion to its frame count. The wider ALOHA platform and camera type are shared; this is not a test across unrelated hardware families.

## Frozen task and cases

The sole target concept is `small white oval candy without a wrapper`. The authored location check requires that target entirely in `collection`, with two distinct, consistent views, `min_confidence=0.8`, `min_stable_samples=2` and `max_age_ms=30000`.

The fixed normalized image rectangles are:

```text
handling:   [0.25, 0.40, 0.75, 0.65]
collection: [0.20, 0.70, 0.80, 0.95]
```

Eight predetermined source timestamps were reviewed before inference: 0.5, 1.5, 4.5, 5.5, 8.5, 9.5, 12.5 and 13.5 seconds. Only the first and final pairs are scored; the four intermediate images are unscored context. No additional timestamps, concepts, calibration or labels were selected after seeing model results.

| Case | Source timestamps | Authored label | Guided Spark → SAM | Spark alone |
| --- | --- | --- | --- | --- |
| Early unsupported target | 0.500 / 1.500 s | UNKNOWN | UNKNOWN | UNKNOWN |
| Visible candy in collection | 12.500 / 13.500 s | PASS | PASS | PASS |

The early label describes missing positive support for the fully visible unwrapped target. The packet and its white label are visible distractors, and a gripper overlaps the packet in the second view. It does not assert that the candy is absent. The final pair shows the small white oval candy clearly in the lower collection area, while the wrapper remains higher in the image.

These are authored image-zone checks. They do not certify unwrapping quality, food safety, grasp correctness, continuous tracking, a complete procedure or a physical correction caused by ProofLoop. There are no ground-truth FAIL cases in this transfer check, so it cannot estimate erroneous acceptance of a known wrong placement.

The frozen contract SHA-256 is `14e0ad0b9854fa39ebdef7905e01347d21f87ab9c10fcffc04f2a07ec4ddfc88`. The run uses the existing `development_reprocessing` enum, with `new_task_transfer` explicitly described in the contract notes and freeze record. This selected development check is not described as an independent holdout.

## What the outputs actually show

The observation method stayed at `normalized_boxes_v2`, module SHA-256 `8f77c12f32911d28c3a17bf53d47f2bf30ac021bea0d68a20d6c1dd0487a3a23`.

On the early pair, Spark's task-independent locator supplied packet/label-area anchors and marked them fully visible. The two unmodified SAM scores were **0.4140625** and **0.77734375**, below the frozen 0.8 threshold. The guided condition therefore abstained. This result demonstrates the confidence gate retaining UNKNOWN; it does **not** demonstrate that semantic identity or the distractor was independently verified. SAM refines a Spark-assigned candidate and does not authenticate its identity.

On the final pair, the two SAM scores were **0.92578125**. Source-supported geometry placed the target inside collection consistently, producing PASS. These are reported model scores, not calibrated correctness probabilities.

| Condition | Actual requests | Estimated USD | End-to-end wall seconds |
| --- | ---: | ---: | ---: |
| Guided Spark → SAM | 6 | 0.031686 | 30.062 |
| Spark alone | 2 | 0.016726 | 15.365 |
| Total | 8 | 0.048412 | 45.427 |

The run stayed inside its eight-request cap, used zero paid retries and did not tune or rerun after inspecting results. Costs use the existing estimated ledger and are not a provider-enforced billing guarantee.

## Audit artifacts

The [public allowlisted summary](../artifacts/feasibility/online-candy-task-transfer-20261028.json) contains the case decisions, metrics and attributed source metadata. Local validation rehashed the source media, recomputed the metrics, replayed both grounded decisions and both inference sidecars, and checked all four condition/case decisions.

Private inputs and freeze: `artifacts/local/transfer-review/`. Private inference artifacts: `artifacts/local/online-benchmark-candy-transfer-20261028-01/`.

```bash
.venv/bin/python -m proofloop.benchmark validate artifacts/local/transfer-review/manifest-approved.json
```

The later images were recorded before ProofLoop requested anything. Their result is a transfer test of observation and checking, not evidence that a robot or person executed a suggested repair.
