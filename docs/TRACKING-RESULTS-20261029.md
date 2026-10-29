# Tracking through partial occlusion and nearby distractors

ProofLoop now has an experimental SAM 3.1 video adapter, source-bound crop assessments and a deterministic semantic reacquisition policy. They are separate from the live placement acceptance path. These tests establish neither continuous physical visibility nor a completed repair.

## Prospective protocol

The [plan](../artifacts/feasibility/tracking-plan-20261029.json) was pushed in commit `f74649ca9884748efa91c7e7fcf79f8adceeb2c8` before inference. Its freeze hash is `92ee474a66ca90a38d713effd8481fd72392a3dfa201a06e8d4f65338f878512`. Code/configuration, all original/derived/source-frame bytes, initial seeds, visual contracts and reference labels were frozen. A pre-inference replay audit was retained locally; no provider requests occurred before the final freeze.

The recordings are episode 1 from two previously exposed ALOHA dataset families: screwdriver (MIT) and battery/remote (Apache-2.0). These are two new recordings, not new unseen domains. Native 640×480 frames were selected at 5 fps and encoded as lossless H.264 because the original AV1 codec is unsupported by SAM's video endpoint. All 100 selected decoded frames were compared with the originals. Attribution, pinned commits, source URLs, selection rules and hashes are in [source credits](TRACKING-SOURCES-20261029.md).

Sixteen checkpoint labels were authored by assistant visual review under user authorization before inference: seven fully visible targets and nine insufficient-visibility states. The videos show partial gripper/cup occlusion and re-exposure. They do not establish full disappearance, an actual identity switch, natural task failures or customer usefulness. Reference labels and later object boxes are scoring-only; models receive target descriptions, initial reviewed seed boxes or their actual tracker candidates.

The unchanged two-view guided adapter runs on four fixed image pairs per video. Its actual Spark localization is also the shared Spark-only baseline. SAM video receives a manually reviewed target and white-tape distractor seed on frame zero and tracks 40/60 frames respectively. Consequently initialization, image context and compute differ: this comparison cannot isolate an intrinsic SAM-vs-Spark accuracy advantage.

## First frozen live run

The [complete public result](../artifacts/feasibility/tracking-results-20261029.json) preserves frame-level decisions, thresholds, source/method hashes, confidence/mask diagnostics, costs and separate completion denominators.

| Diagnostic method | Completed checkpoints | Correct clear targets retained | Supported occluded/unclear checkpoints | Support coverage |
| --- | ---: | ---: | ---: | ---: |
| Shared two-view Spark localization | 14/16 | 0/7 | 0/9, with 2 unscored | 5/16 |
| Unchanged guided-v2 still refinement | 14/16 | 0/7 | 0/9, with 2 unscored | 0/16 |
| Manually seeded SAM video presence | 16/16 | 7/7 | 3/9 | 10/16 |
| Semantic reacquisition v1 | 16/16 | 1/7 | 1/9 | 2/16 |

“Correct clear target” requires a supported decision and reference-box IoU ≥0.3. Spark supported five clear targets but all five boxes missed that primary localization threshold. At IoU ≥0.2 it retained 3/7; all threshold sensitivity results are retained. This is not evidence that Spark generally fails object identity. Existing published placement cohorts remain unchanged.

One unchanged Spark baseline response exhausted its 4,096-token output budget, leaving both checkpoints in that pair unscored. It consumed reported tokens and remains a failed response, not an UNKNOWN prediction. The nine unclear reference states include those two unscored frames; the baseline actually withheld seven completed unclear states.

The semantic policy reduced unsupported support from three unclear checkpoints to one, but also lost six of seven correct clear targets. Its supported-result accuracy was only 1/2 and it completed zero reacquisitions. Those tradeoffs prevent a robustness claim or promotion into live acceptance. Track presence preserves localization well here; it does not establish full visibility or semantic continuity.

There were **31 provider attempts: 30 transport completions and one rejected incomplete response**, with no automatic retries. The estimated total was **$0.217498**. The two video requests used frame pricing (50-frame minimum; 50 and 60 reported billed frames), rather than the still-image rate. Reported token usage from the incomplete response is included. Unknown/unreported charges and account billing remain the provider's authority. [Meta pricing](https://dev.meta.ai/docs/pricing-rate-limits)

## Identified engineering weaknesses

- Bounding-box overlap overreacts to adjacent objects. In the remote video, the tracked body and tape rectangles overlap while their decoded foreground masks are disjoint. The v1 rule nevertheless reports a hard gap on all 60 frames. A source-bound mask correction is evaluated separately, preserving this result.
- The crop model incorrectly supports full visibility for the partly gripped screwdriver at 1.6 seconds. It also supports a partly obscured remote at 9 seconds; the latter is rejected only by SAM's unchanged confidence gate. Crop/model agreement is therefore not independent identity or visibility proof.
- The remote crop checks reject several genuinely clear states as not fully visible. Removing the rectangle-overlap veto alone cannot fix that semantic coverage loss.
- The unchanged Spark locator still has an incomplete-response failure. A separate low-reasoning completion repair is tested on those exposed images, with its own budget and provenance; it cannot replace the original comparison.

## Implementation and replay

`proofloop.video_tracking` makes one bounded non-streaming Meta request with no retry, records actual video usage, validates dimensions/object IDs, sorts explicit frame markers and retains missing records as UNKNOWN. [Meta documents frame order and warns about identity across full occlusion](https://dev.meta.ai/docs/sam/reading-segmentation).

`proofloop.tracking_policy` requires distinct source-bound semantic views after a hard evidence gap. Dense unassessed frames do not count as confirmations. It cannot infer an occlusion's physical cause. `tracking_guard` crops only actual video candidates, checks attributes/exclusions and replays exact crop geometry/bytes. `tracking_evaluation` reports coverage, positive retention, selective accuracy, unsupported support, localization sensitivity and missing completions separately.

The private run is `artifacts/local/tracking-experiment-20261029-01`. With its local source/media/raw-response files present, offline replay is:

```bash
python -m proofloop.tracking_experiment replay \
  --frozen artifacts/local/tracking-experiment-20261029-01/freeze.json
```

Replay makes zero model calls and rejects changed source derivations, frame inventories, contracts, code, prompts, candidate geometry, usage or reported costs. Local hashes are consistency checks, not signatures proving API origin or capture time. A public clone contains the code, tests, references and safe summaries; it does not contain licensed videos or raw provider responses.
