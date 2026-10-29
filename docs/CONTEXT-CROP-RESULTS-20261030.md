# Native context-crop results

The publicly declared fifteen-call probe passes its **proposal feasibility gate**: both previously rejected clear-cup pairs now have two eligible SAM projections. This is a component result on exposed development cases, not a task-accuracy improvement or production acceptance change. A reference-UNKNOWN cloth pair also has eligible proposals, showing why eligibility alone is insufficient.

The [frozen plan](CONTEXT-CROP-PLAN-20261030.md) and implementation were published in commit [`8f7a557684138fa2b1cff60ddbe3d5bd511466a4`](https://github.com/yerbol1212/proofloop/commit/8f7a557684138fa2b1cff60ddbe3d5bd511466a4) before the first paid admission. Public plan file SHA-256 is `806a76abd70fcc7fc3532286106a1059163a75af11648c332f7b1f778b1ac9fa`. The [machine-readable result](../artifacts/feasibility/context-crop-results-20261030.json) retains every original view and pair, each outcome, all source/crop/response/projection hashes and full request accounting.

## Complete original inventory

| Original pair | Authored reference | Two eligible proposals | Task verdict |
| --- | --- | --- | --- |
| Bridge fork 0 | FAIL | Yes | Not evaluated |
| Bridge fork 1 | UNKNOWN | No: missing original proposal | Not evaluated |
| Bridge fork 2 | UNKNOWN | No: one crop score below 0.8 | Not evaluated |
| UR5 cloth 0 | UNKNOWN | No: cached visibility veto | Not evaluated |
| UR5 cloth 1 | UNKNOWN | No: cached visibility veto | Not evaluated |
| UR5 cloth 2 | UNKNOWN | Yes | Not evaluated |
| UR5 cup 0 | PASS | Yes | Not evaluated |
| UR5 cup 1 | PASS | Yes | Not evaluated |
| UR5 cup 2 | UNKNOWN | No: both original proposals missing | Not evaluated |

All eighteen original views/nine pairs remain. Fifteen available crops each receive one unchanged concept-SAM request; three empty original proposals remain explicit and select no crop. All fifteen responses complete, with zero retries, Spark calls, request failures or unscored crop outcomes. Four of nine pairs have eligible geometry; this fraction is neither coverage of correct decisions nor accuracy.

## Clear-cup record scores

| Clear pair/view | Original full-source concept score | Crop concept score |
| --- | ---: | ---: |
| Cup 0 / first | 0.729958 | 0.875000 |
| Cup 0 / second | 0.870968 | 0.840000 |
| Cup 1 / first | 0.714286 | 0.851852 |
| Cup 1 / second | 0.875000 | 0.851852 |

These are unmodified provider scores at the unchanged 0.8 threshold. They are not probabilities that the placement is correct. The [hosted documentation](https://dev.meta.ai/docs/sam/reading-segmentation) describes confidence per record; it does not establish calibration or the score's formula. This trial changes both spatial context and JPEG-to-lossless-PNG transport. It does not isolate which caused the difference, estimate repeat-run variance or establish behavior on new recordings.

## Cost and boundaries

The persistent journal and response ledgers report **15 SAM requests, zero Spark requests, 15 successful settlements, zero uncertain attempts and zero remaining allowance**. Estimated incremental cost is **$0.0375**; this is not a provider billing invoice. The entire launcher takes 73.789 seconds, including repeated immutable-input validation; summed per-client experiment wall time is 46.770 seconds. Neither is interactive production latency or a speed comparison with the earlier methods.

Native crop pixels, actual crop-coordinate responses and derived source-coordinate geometry remain separately bound. Confidence is unchanged; crop and original count as one physical view. Low scores, missing or multiple detections, artificial crop edges and original cached identity/visibility/uniqueness vetoes cannot yield an eligible projection.

An [independent integrity audit](../artifacts/feasibility/context-crop-audit-20261030.json) found zero errors. The reviewer officially re-decoded all fifteen retained SAM responses, reconstructed the exact request payloads and matched their hashes to all fifteen journal rows. Each crop's decoded RGB pixels match its declared original-source rectangle byte-for-byte; translated boxes, masks and scores remain exact. A visual diagnostic of selected fork/cup masks suggests neighboring objects are excluded, while cloth/gripper-edge ambiguity persists. This is assistant inspection of exposed examples, not independent human ground truth, signed provenance or a semantic accuracy estimate.

The positive result permits further investigation of this branch. It does not permit live task acceptance. The next method must verify full-source identity, uniqueness, visibility, gripper separation and whole-mask region placement before any task verdict, including the known ambiguous cloth and neighboring-spoon cases. It must then be frozen and evaluated on additional untouched sources. Automatic gripper localization, an operator following a displayed repair request and practical user benefit remain unproved. The existing production perception method and all historical results remain unchanged.

The concurrent [freshness investigation](FRESHNESS-INVESTIGATION-20261030.md) fixes independently reproduced production timing defects; those fixes do not rely on this experimental perception result.
