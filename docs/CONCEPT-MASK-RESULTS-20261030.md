# Concept SAM and mask-aware Spark: negative result

**Rejected for live acceptance.** The method detects the known merged fork/spoon foreground, but neither recovers the clear positive controls nor reliably rejects gripper contact. The [predeclared gate](CONCEPT-MASK-PLAN-20261030.md) fails. No thresholds, reference labels or acceptance regions were changed after inference.

All nine exposed development cases remain: two PASS, one FAIL and six UNKNOWN from three recordings in two source families. The implementation and [exact declaration](../artifacts/feasibility/concept-mask-plan-20261030.json) were public in commit `930a7622e3d5215e562c1b6ad987f2ff93ca976c` before the first paid admission.

| Method on the same nine cases | Reference matches | Clear PASS retained | Unsupported PASS / nonpassing references | UNKNOWN outputs |
| --- | ---: | ---: | ---: | ---: |
| Original direct Spark | 6/9 | 1/2 | 2/7 | 5/9 |
| Original guided v2 | 7/9 | 0/2 | 0/7 | 8/9 |
| Fixed spatial padding | 6/9 | 0/2 | 0/7 | 9/9 |
| Concept SAM + mask-aware Spark | 5/9 | 0/2 | 1/7 | 8/9 |

The new method loses the supported fork FAIL because its second SAM score is 0.773. The two clear cup pairs have first-view scores 0.730 and 0.714, below the unchanged 0.8 gate; their second-view scores are 0.871 and 0.875. A high score in one view cannot replace two supporting views. Near-identical predicted masks with differing scores suggest context/score sensitivity, rather than demonstrating that target localization alone was fixed. Predicted-mask agreement is not ground-truth segmentation accuracy.

The one admitted main semantic challenge marks the final cloth pair fully visible and supports a PASS. The first view has a gripper finger beside the cloth, and the unchanged authored reference is UNKNOWN. This is the new method's incorrect acceptance. The other five UNKNOWN references remain withheld. The error is preserved rather than relabeled as a positive example.

A separate completed Spark challenge examines two retained high-scoring spatial masks from the padding experiment. It marks `mask_target_only` contradicted in both views because foreground includes neighboring utensils, returning UNKNOWN. This control uses no new SAM requests and is outside the nine-case metric denominator. One correctly rejected known failure is not general semantic validation.

The main trial completes 18 SAM requests and one Spark request for an estimated $0.049724. The separate control adds one Spark request, estimated $0.006883: **20 new requests, estimated $0.056607 total**, no retries, no recorded accounting uncertainty, and eight unused request slots. Estimates use the existing local rate accounting and are not a reconciled provider invoice.

The [independent audit](../artifacts/feasibility/concept-mask-audit-20261030.json) re-parses and officially re-decodes all 18 SAM responses, strictly parses both Spark responses, reproduces ordered attachments and matches all 20 request payload hashes against persistent admissions. It verifies local consistency, not provider authenticity or independent human ground truth. [Full public result](../artifacts/feasibility/concept-mask-results-20261030.json).

The two prepared holdback recordings remain untested. This result does not qualify promotion or a fresh-source success claim. The immediate next diagnostic separates gripper foreground from target foreground using the already fixed three-source-pixel warning margin. Such a warning can only withhold support; it cannot fix clear-positive retention, prove occlusion or authorize a physical action.
