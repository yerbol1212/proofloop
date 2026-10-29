# Still-image gripper text proposals: negative result

**Rejected component.** All 18 SAM requests completed, but the fixed `robot gripper` concept returned no objects in every image. The foreground-distance code therefore had no gripper mask to compare. Missing proposals are not verified absence.

The [public declaration](IMAGE-GRIPPER-PLAN-20261030.md) was pushed in commit `975a8f2718eed462fbd098e64a467ed3d95864f7` before inference. The follow-up preserves all 18 original views, nine task references and cached parent decisions. It uses no Spark requests or retries.

| Diagnostic | Result |
| --- | ---: |
| Completed SAM responses | 18/18 |
| Returned gripper objects | 0 |
| Available geometry comparisons | 0/18 |
| Measured near warnings | 0/18 |
| Near-reference matches | 0/1; unscored 1/1 |
| Separated-reference matches | 0/5; unscored 5/5 |

The eight genuine parent UNKNOWN results remain UNKNOWN. The ambiguous cloth PASS becomes **UNSCORED because additional evidence is unavailable**. It is not corrected by a measured near warning and must not be reported as a successful occlusion rejection. All nine task references remain in the report; clear-positive retention stays 0/2.

The added cost is an estimated **$0.045 for 18 SAM calls**, with no recorded accounting uncertainty. Together with the concept-mask trial, this assessment has used 38 requests for an estimated $0.101607. Estimates are local accounting, not a reconciled invoice.

The [independent audit](../artifacts/feasibility/image-gripper-audit-20261030.json) reconstructs all 18 requests, matches their payload hashes to persistent admissions, re-parses and officially decodes all returned text, recomputes projections and verifies response/ledger/cost consistency. Empty responses remain empty; no masks are fabricated. [Full result](../artifacts/feasibility/image-gripper-results-20261030.json).

A final, separately declared six-view spatial control will use manual gripper anchors on the already fixed diagnostic references. It isolates proposal availability from geometry. It cannot establish automatic gripper localization, independent semantic identity, recovery of clear task positives or qualification for production. The failed text experiment remains unchanged.
