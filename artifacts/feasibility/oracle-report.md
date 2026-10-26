# ProofLoop synthetic oracle diagnostic

This diagnostic supplies authored symbolic facts. It makes **zero API calls** and contains **no camera or model benchmark**. It checks software behavior under the supplied facts.

Generated (UTC): 2026-10-26T17:37:55.024681+00:00

Episode assertions: **17/17**. Stage assertions: **24/24**. Planner assertions: **1/1**.

| Synthetic episode | Expected | Actual | All stage assertions |
| --- | --- | --- | --- |
| correct | pass | pass | pass |
| alternate-order | pass | pass | pass |
| wrong-region | fail | fail | pass |
| missing-visible-item | fail | fail | pass |
| missing-observation | unknown | unknown | pass |
| occluded-after-pass | unknown | unknown | pass |
| low-confidence | unknown | unknown | pass |
| unsupported-negative | unknown | unknown | pass |
| unstable-negative | unknown | unknown | pass |
| expired-observations | unknown | unknown | pass |
| new-attempt-stale-facts | unknown | unknown | pass |
| repair-fresh-evidence | pass | pass | pass |
| repair-partial-refresh | unknown | unknown | pass |
| temporal-correct | pass | pass | pass |
| temporal-missing-checkpoint | unknown | unknown | pass |
| temporal-wrong-order | fail | fail | pass |
| edited-requirement | pass | pass | pass |

## Repair model

A shortest path in this authored finite model; physical repair still needs fresh observations.

Expected actions: `['move-screwdriver-left', 'inspect-screwdriver-region']`; returned actions: `['move-screwdriver-left', 'inspect-screwdriver-region']`.

## Interpretation limits

- All observed facts and expected answers are authored synthetic fixtures.
- No image, video, phone recording, segmentation result or Meta model inference is evaluated here.
- Fixture assertion success establishes software behavior under supplied facts, not perception accuracy.
- The alternative-order fixture tests independent fact arrival; it does not infer real physical dependencies.
- Real Meta access and the camera-to-observation-to-correction loop require separate feasibility checks.

The adjacent JSON contains each stage's expected/actual status, check reasons, evidence IDs, synthetic origins, specification hashes, raw counts and the authored repair action model.
