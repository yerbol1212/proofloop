# Visibility assessment: measured improvement with a remaining boundary error

The new full-frame assessor retained all six clear targets in two new recordings, versus five for the unchanged crop guard. It also called one of ten unclear views clear, versus zero before. That error is a borderline white-on-white gripper contact. The unchanged two-view policy withheld it. This is a component improvement with a measurable tradeoff, not a solved visibility problem or a production reliability claim.

## Frozen protocol and sources

The [protocol](VISIBILITY-PROTOCOL-20261029.md), [public plan](../artifacts/feasibility/visibility-plan-20261029.json), code and tests were pushed in commit `4addfc4ac134bf51b8ea4b3d0f1d39f2ba593eca` before new inference. The freeze hash is `f751a4ffa2502b61d15694e4f5902e861cc408be65afe2ecff1de7b96006caf6`. The run used every planned request once, without retries, response-based case selection, threshold changes or post-inference reference changes.

The exposed cohort reuses the original episode-1 SAM and crop-guard responses, scoring the same sixteen checkpoints. Its new cost is only sixteen new-assessor calls. The new-recording cohort adds episode 2 from the MIT screwdriver and Apache-2.0 battery/remote datasets. Native decoded pixels were preserved in 100 selected 5-fps frames, and sixteen checkpoint labels were frozen before inference. Both assessors receive the same actual SAM candidates, initialized by reviewed frame-zero target/tape boxes. Later labels/boxes enter scoring only. Sources, pinned URLs, licenses and exact original/derived hashes are in the protocol and plan.

Both cohorts share exposed dataset families, camera setups and tasks. The new recordings are new source files, not independent cross-domain validation. The reference labels are assistant-authored visual judgments, not official dataset annotations or precise occlusion measurements.

## Visibility and candidate support

All 48 new Spark classifications completed, and both new SAM video requests returned their complete 40/60-frame inventories. There are no unscored checkpoint classifications in this run.

| Cohort | Assessor | Clear targets retained | Unclear views called clear | Support coverage |
| --- | --- | ---: | ---: | ---: |
| Exposed episode-1 development | Original crop guard | 3/7 | 2/9 | 5/16 |
| Exposed episode-1 development | Full frame + crop | 5/7 | 0/9 | 5/16 |
| New episode-2 recordings | Original crop guard | 5/6 | 0/10 | 5/16 |
| New episode-2 recordings | Full frame + crop | 6/6 | 1/10 | 7/16 |

These counts are the direct visibility classifications. Identity-plus-visibility candidate support has the same counts in this sample, and every retained clear candidate also meets the primary box-IoU threshold of 0.3. Those are distinct diagnostics in the saved result; they are not assumed to be equivalent generally. IoU 0.2/0.5 sensitivity, completion, per-condition errors and source-level grouping are retained in the [full result](../artifacts/feasibility/visibility-results-20261029.json). A [32-checkpoint CSV](../artifacts/feasibility/visibility-checkpoints-20261029.csv) makes disagreements easier to inspect.

The new assessor withheld all **7/7 exposed** and **6/6 new** checkpoints labeled partial occlusion. The sole new false-clear view is `remote_new` frame 59, labeled hidden/insufficiently identifiable before inference because the body boundary blends with nearby white grippers. It is still counted as an error. This does not support a claim of zero missed uncertainty.

The two remaining exposed clear-view rejections are remote frames 15 and 59. On frame 15, the model incorrectly reports a gripper hiding the right end despite a visible dark gap. Frame 59 is annotation-sensitive: the white gripper meets or blends with the upper-left white body boundary. An independent image-only review after the run found that historical label borderline, without reading model answers. Every frozen label remains unchanged; neither borderline case is excluded to improve metrics.

## Downstream tracking policy

Both policy versions use the previously evaluated mask-based distractor correction, confidence ≥0.8, two distinct source-bound views, and 500-ms spacing. Only the guard observations change.

| Cohort | Guard supplied to policy | Correct clear checkpoints supported | Unclear checkpoints supported |
| --- | --- | ---: | ---: |
| Exposed development | Original guard | 1/7 | 1/9 |
| Exposed development | Full-frame guard | 2/7 | 0/9 |
| New recordings | Original guard | 2/6 | 0/10 |
| New recordings | Full-frame guard | 2/6 | 0/10 |

The new-recording component gain does **not** improve downstream retention on this schedule. Each initial clear view still needs a second confirmation. The clear frame at 1.0 seconds is only 400 ms after the accepted 0.6-second view, so the unchanged spacing rule withholds it. The borderline final remote view is the first semantic support after a gap and is withheld. No threshold, timing requirement or evidence rule was relaxed to inflate coverage. Both versions complete zero reacquisitions.

SAM presence alone retained 7/7 exposed clear targets while supporting 3/9 unclear views; on the new recordings it retained 6/6 but supported 9/10 unclear views. Tracking presence remains insufficient for full-visibility verification. These are candidate diagnostics, not placement PASS or observed physical recovery.

## What changed in the implementation

`proofloop.visibility_guard` registers the full current source frame and its deterministic unscaled crop in one low-reasoning Spark call. It looks beyond the candidate box for hidden continuation and keeps physical visibility independent of semantic identity. Explicit observations describe complete/interrupted boundaries, visible/hidden extremities, external occluders, image-edge truncation, and blur. Clear requires at least two distinct named terminal landmarks and no conflicting evidence. Normal unseen surfaces and adjacent objects alone do not cause rejection. Identity still requires direct distinctive evidence and at least one visible matching attribute; recognizability cannot override occlusion.

`proofloop.visibility_experiment` freezes source bytes, derivations, JPEG inventories, labels, contracts, code and request settings. It validates actual candidate geometry and raw provider output, preserves incomplete/error paths as unscored, compares seven diagnostic methods, and verifies the durable request journal. The original tracking/guard/model modules and all earlier public results remain unchanged.

The intervention changes full-frame context and the explicit assessment rubric together. It cannot establish which change caused the measured improvement. Same-model judgments remain correlated; a verbal landmark description is fallible evidence, not an independent measurement.

## Cost, checks and reproduction

The run made **50 new requests: 48 Spark and 2 SAM video**. Reported Spark usage was 47,893 input and 78,507 output tokens; SAM reported 110 billed video frames including the 50-frame minimum for the shorter clip. The estimated incremental total is **$0.415522**: $0.152475 for exposed development and $0.263047 for new recordings. Reused historical inference is not charged again. Costs were independently recalculated from retained usage; account billing remains authoritative. [Meta pricing](https://dev.meta.ai/docs/pricing-rate-limits)

The complete suite passes **1,011 tests on Python 3.12**; Python 3.9 passes with two expected optional-parser skips. The pre-inference commit's GitHub CI passed both Python jobs and the production frontend build. A separate read-only audit recomputed all fourteen primary score reports, all forty-two method/cohort/IoU comparisons, direct visibility counts and request costs. It matched the exposed original policy exactly to the earlier published mask retest and found no secrets, private paths, provider identifiers, arbitrary model observations or mask payloads in the public projection.

With the retained private inputs present, offline replay makes no API calls:

```bash
python -m proofloop.visibility_experiment replay \
  --frozen artifacts/local/visibility-experiment-20261029-01/trial/freeze.json
```

The public repository contains implementation, tests, source credits, the prospective plan, safe detailed results and checkpoint CSV. Local hashes are consistency receipts, not signatures proving API origin. The new guard is implemented in the experimental tracking pipeline; it remains outside live placement acceptance because near-contact boundaries, broader generalization and physical recovery remain unresolved. The next improvement should address spatial separation at object/gripper boundaries and collect independently reviewed near-contact cases, rather than reduce the visibility requirement.
