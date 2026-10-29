# Prospective transfer pilot: results and failures

Live execution: **30 October 2026, 00:06:44–00:08:34, Asia/Almaty** (29 October UTC). The study's `20261029` filenames retain its preregistration identifier.

On nine preregistered paired cases, guided v2 matched **7/9** authored references versus direct Spark's **6/9**. Guided made **0/7 incorrect PASS decisions**, compared with Spark's **2/7**, but **accepted neither of the two clear positive cases**. Spark accepted one of those two. Guided returned UNKNOWN on eight of nine cases. This is a conservative tradeoff on a small curated pilot, not evidence of a generally better or useful placement verifier.

The [public preregistration](../artifacts/feasibility/new-source-preregistration-20261029.json) was published in [commit a7c3e347](https://github.com/yerbol1212/proofloop/commit/a7c3e347d7a9b13914607d559ec6281a94eb8a15) before the first inference. Its exact file SHA-256 is `7415f12ea5e5b1f19af6ca2bc32fa180abb083f9430b83037eb2c859f4a7e5b0`. The [allowlisted machine-readable result](../artifacts/feasibility/new-source-results-20261029.json) preserves the outcomes without media, raw provider text or private paths.

## What was fixed before inference

The pilot contains three original recordings from two previously untested source families: BridgeData V2 scripted manipulation and Berkeley UR5. The cloth and cup recordings share the UR5 family and camera setup; they are not two independent source families. There are nine paired cases, eighteen distinct extracted images and eighteen case/method results. Reference labels are **two PASS, one FAIL and six UNKNOWN**.

Start, midpoint and end pair starts were selected by a fixed schedule, with a second source view at least one second later. The Bridge redistributor encodes frames at one frame per second, so its original per-frame capture metadata supplies elapsed source time; encoded playback time is separately retained. UR5 uses decoded presentation timestamps. Inputs were decoded to upright JPEGs without cropping, resizing, creative edits or generated scene content.

An independent assistant reviewer inspected both contact sheets and all original selected frames before any model calls. The initial six fork/cloth cases contained no strict PASS reference. They were all retained, and three fixed-schedule brown receiving-cup cases were added before inference to provide clear-target controls. This selection history is public. The reviewer is another AI agent, not an independent human or official dataset annotator.

Both methods receive the same target concepts, two exact source images and reviewed placement requirement. Direct Spark evaluates the whole-target placement contract. Guided v2's Spark locator receives the target concept without desired placements or scoring annotations; SAM refines proposed boxes, followed by the unchanged grounding and checker. The confidence threshold remains 0.8 and the contract requires two distinct views. Seeds, tracking, gripper-warning policies and the historical tracking fixtures are not part of this comparison.

The published transport profile uses Muse Spark `muse-spark-1.3` with low reasoning and a 4,096-token output cap, plus SAM `sam-3.1`. The shared persistent journal caps the whole study at 36 attempts. Each case/method runs once, clients are fresh and automatic retries are disabled. Nine implementation files, source bytes, annotations and approvals were frozen before inference and checked before subsequent admissions. No label, region, threshold, prompt or implementation was adjusted after observing responses.

## Outcomes by recording

“Incorrect PASS” counts a PASS against either FAIL or UNKNOWN references. “Clear positives retained” counts PASS against the two PASS references. Missing predictions remain distinct from an actual UNKNOWN response; none were missing in this run.

| Recording | References | Method | Matches | Incorrect PASS | Clear positives retained | UNKNOWN outputs |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Bridge scripted utensils | 1 FAIL, 2 UNKNOWN | Direct Spark | 3/3 | 0/3 | Not applicable | 2/3 |
| Bridge scripted utensils | 1 FAIL, 2 UNKNOWN | Guided v2 | 3/3 | 0/3 | Not applicable | 2/3 |
| UR5 cloth | 3 UNKNOWN | Direct Spark | 2/3 | 1/3 | Not applicable | 2/3 |
| UR5 cloth | 3 UNKNOWN | Guided v2 | 3/3 | 0/3 | Not applicable | 3/3 |
| UR5 receiving cup | 2 PASS, 1 UNKNOWN | Direct Spark | 1/3 | 1/1 | 1/2 | 1/3 |
| UR5 receiving cup | 2 PASS, 1 UNKNOWN | Guided v2 | 1/3 | 0/1 | 0/2 | 3/3 |
| **All three recordings** | **2 PASS, 1 FAIL, 6 UNKNOWN** | **Direct Spark** | **6/9** | **2/7** | **1/2** | **5/9** |
| **All three recordings** | **2 PASS, 1 FAIL, 6 UNKNOWN** | **Guided v2** | **7/9** | **0/7** | **0/2** | **8/9** |

Direct Spark's non-UNKNOWN decision coverage is 4/9; guided v2's is 1/9. Both identify the single reviewed wrong-placement case. Guided's higher match total cannot establish usefulness when its two clear positive controls both remain unverified.

## Every case remains visible

| Case | Source times, seconds | Reference | Direct Spark | Guided v2 | Reference rationale |
| --- | --- | --- | --- | --- | --- |
| Bridge pair 0 | 0.000, 2.169 | FAIL | FAIL | FAIL | Blue-handled fork wholly LEFT in both views; reviewed requirement is RIGHT. |
| Bridge pair 1 | 5.360, 6.563 | UNKNOWN | UNKNOWN | UNKNOWN | First view has gripper occlusion; later view shows LEFT. |
| Bridge pair 2 | 10.757, 11.768 | UNKNOWN | UNKNOWN | UNKNOWN | First view crosses the middle boundary; later view shows LEFT. |
| UR5 cloth pair 0 | 0.000, 1.000 | UNKNOWN | UNKNOWN | UNKNOWN | Cloth lies in the other region but is partially occluded by the gripper. |
| UR5 cloth pair 1 | 6.400, 7.400 | UNKNOWN | UNKNOWN | UNKNOWN | Cloth crosses the reviewed LEFT boundary and is partially occluded. |
| UR5 cloth pair 2 | 13.000, 14.000 | UNKNOWN | **PASS** | UNKNOWN | One view has ambiguous gripper contact at the cloth boundary; the later clear view alone is insufficient. |
| UR5 cup pair 0 | 0.000, 1.000 | PASS | PASS | **UNKNOWN** | Unique brown receiving cup's rim and body are visible wholly LEFT. |
| UR5 cup pair 1 | 11.600, 12.600 | PASS | **UNKNOWN** | **UNKNOWN** | Brown cup remains visible wholly LEFT; gripper acts on the separate blue cup behind/above it. |
| UR5 cup pair 2 | 23.400, 24.400 | UNKNOWN | **PASS** | UNKNOWN | Nested blue cup covers the brown receiving cup's upper opening/rim. |

The cup requirement concerns the visible brown receiving cup's staging location. It does not evaluate blue-cup nesting, movement execution, cup contents or overall manipulation success. FAIL labels describe contradiction of an authored placement requirement; they are not publisher labels that a robot made a procedural error.

## Why guided withheld support

The two false Spark acceptances were withheld by guided, but this does **not** prove that guided recognized the physical occlusion mechanism. For cloth pair 2, the guided Spark locator itself marked both views fully visible and identity unambiguous. SAM's reported scores were approximately **0.547 and 0.324**, below the frozen 0.8 threshold, so the checker had insufficient supported observations.

The same confidence gate also rejected the clear brown-cup controls. Cup pair 0's SAM scores were approximately **0.494 and 0.586**; pair 1's were **0.416 and 0.570**. The locator supported their visibility and identity, but both pairs remained UNKNOWN. On the nested cup pair 2, the locator again reported full visibility; SAM scores of approximately **0.398 and 0.428** withheld support. Neither this outcome nor low reported scores independently prove occlusion, semantic identity or calibrated risk.

Direct Spark withheld the clear cup pair 1 because it judged the cup too close to a region boundary. The frozen visual reference placed the cup comfortably inside LEFT. This unnecessary abstention is retained rather than repairing the prompt or enlarging the region after seeing the response.

## Cost, latency and accounting

| Recording | Method | Actual attempts | Known estimated USD | Sum of case/method elapsed seconds |
| --- | --- | ---: | ---: | ---: |
| Bridge scripted utensils | Direct Spark | 3 | 0.014579 | 14.306 |
| Bridge scripted utensils | Guided v2 | 8 | 0.028608 | 23.604 |
| UR5 cloth | Direct Spark | 3 | 0.017583 | 16.866 |
| UR5 cloth | Guided v2 | 5 | 0.019685 | 20.032 |
| UR5 receiving cup | Direct Spark | 3 | 0.017642 | 16.565 |
| UR5 receiving cup | Guided v2 | 9 | 0.028114 | 19.169 |
| **Total** | **Direct Spark** | **9** | **0.049804** | **47.737** |
| **Total** | **Guided v2** | **22** | **0.076407** | **62.805** |

The complete run used **31/36 allowed attempts**: 18 Spark generations and 13 SAM image requests. The known estimated total is **$0.126211**, summed from rounded condition subtotals. All 31 journal entries settled `ok`; no rejected, uncertain or unresolved attempts were recorded, every recorded `token_usage_incomplete`, `request_accounting_incomplete` and `cost_estimate_incomplete` flag was false, and no possible unreported charge was flagged. The saved token, request and estimated-cost records passed their consistency checks. These are local estimates at the recorded Standard rates, not an independently reconciled provider invoice or a dollar cap.

The study's actual wall time was **110.826 seconds**. Table timings are sums of whole case/method work, including local validation and file work; they are not provider-only latency or a latency distribution. Sequential fixed method order and a single run do not support a speed benchmark. Unsupported locator views skip their corresponding SAM requests, explaining the five unused request slots.

## Independent consistency checks

All three saved cohort summaries were regenerated against their frozen sources and approvals and matched exactly. All eighteen completed method results replayed their retained private inference sidecars. A separate check reparsed the actual retained provider output text for all nine direct Spark results and all nine guided locators; each matched its stored parsed JSON. All nine references remain in both method denominators, and there are no missing or error predictions. This validates recorded consistency, not the correctness of the provider's perception or cryptographic authenticity of API origin.

The additive `proofloop.prospective_audit` module now enforces this raw-text check without altering the nine frozen study files. It rejects contradictory parsed copies, duplicate JSON fields, nonfinite numbers including exponent overflow, conflicting response text lanes, incomplete output and changed or reordered source bindings. Fifteen focused regressions pass on both Python versions. The public result includes all three allowlisted audit receipts. With retained private evidence, run:

```sh
.venv/bin/python -m proofloop.prospective_audit \
  artifacts/local/prospective-20261029/pilot-ur5-cup-4
```

The command performs no credential loading, network access or new inference. A public clone lacks the private source media and raw responses; receipt hashes alone do not reproduce or authenticate those unavailable inputs.

The additive `python -m proofloop.prospective_audit PRIVATE_PILOT_FOLDER` command repeats this raw-text check without changing the nine frozen study modules. It validates current media and reports, exact source attachment order, completed provider status, the direct/locator schemas and strict parsed equality. Duplicate fields, invalid JSON, nonfinite values including exponent overflow, conflicting text lanes and missing completed evidence fail the audit. Its public receipt contains counts and hashes only. Fifteen authored regressions exercise these checks with zero provider requests; both Python 3.12 and Python 3.9 pass.

## Comparison with earlier evidence and remaining limits

The earlier fresh coffee placement comparison was guided **3/4** versus Spark **4/4**; the reused temporal stress comparison was **7/8** versus **8/8**. Those failures remain unchanged. This pilot adds previously untested task/source families and camera conditions, and exposes two unsupported Spark acceptances under the frozen strict references. Its higher guided match total comes with zero clear positive acceptance. The recordings, contracts, class balance and transport profiles differ across experiments; pooling them into a single accuracy headline would obscure those differences.

These nine selected cases are not an official benchmark, independent human annotation, naturally occurring error dataset, full-occlusion identity recovery test or real-time robot experiment. Near-contact and nested-object references are annotation-sensitive. Source selection and the added clear-target recording were assistant authored before inference; they do not make the sample representative. Two of the three recordings share one source family and camera. Public attribution distinguishes Bridge's upstream CC BY declaration from UR5's pinned redistributor declaration, whose original homepage license was not independently verified. Raw media and license evidence remain private.

No operator or robot followed a ProofLoop repair request in this experiment. No physical action, whole-procedure completion, safety, dispensing or autonomous recovery was verified. The practical next gate remains a current operator-followed correction and independent usability observation, alongside a new prospective positive-retention experiment that preserves the unsupported-acceptance controls. Lowering the threshold on these now-exposed cup frames would be development tuning and could not retroactively improve this pilot's result.
