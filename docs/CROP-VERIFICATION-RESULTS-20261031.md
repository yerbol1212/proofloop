# Full-source crop verification results

The candidate fails its unchanged development gate: **7/9 reference matches, 1/2 clear positives retained, eight UNKNOWN results, zero unsupported PASS and zero missing predictions**. It withholds the known cloth ambiguity, but loses the wrong-side fork decision and rejects one clear cup. It remains outside production, and the separately prepared recordings receive no inference.

The [exact declaration](CROP-VERIFICATION-PLAN-20261031.md) and implementation were published in commit [`b89a37b77a22b8db916533c1014731fefa7682c4`](https://github.com/yerbol1212/proofloop/commit/b89a37b77a22b8db916533c1014731fefa7682c4) before the sixteen paid requests. The [full result](../artifacts/feasibility/crop-verification-results-20261031.json) retains all nine exposed cases and eighteen source views. References are assistant-authored judgments, not independent human ground truth.

## Original cases, unchanged denominator

| Pair | Reference | New result | Governing evidence |
| --- | --- | --- | --- |
| Bridge fork 0 | FAIL | UNKNOWN | Both mask-extent assessments contradicted |
| Bridge fork 1 | UNKNOWN | UNKNOWN | Completed original crop gate withheld support |
| Bridge fork 2 | UNKNOWN | UNKNOWN | Completed original crop gate withheld support |
| UR5 cloth 0 | UNKNOWN | UNKNOWN | Cached visibility veto remains |
| UR5 cloth 1 | UNKNOWN | UNKNOWN | Cached visibility veto remains |
| UR5 cloth 2 | UNKNOWN | UNKNOWN | Returned tool coverage contradicted; first-view near warning |
| UR5 cup 0 | PASS | PASS | Both views satisfy all frozen gates and whole-region checks |
| UR5 cup 1 | PASS | UNKNOWN | Weak tool masks and contradicted returned-mask coverage |
| UR5 cup 2 | UNKNOWN | UNKNOWN | Original proposals missing |

Five cases stop at their completed original crop gate; four receive the new localization, spatial-SAM and eight-image mask review. No case is dropped, substituted or assigned UNKNOWN because of a failed request. All required requests complete.

The [original new-family pilot](NEW-SOURCE-RESULTS-20261029.md) and this exposed development follow-up have different computation and selection histories:

| Method | Reference matches | Clear positives retained | Wrong-side FAIL retained | Unsupported PASS |
| --- | ---: | ---: | ---: | ---: |
| Original direct Spark | 6/9 | 1/2 | 1/1 | 2/7 |
| Original guided | 7/9 | 0/2 | 1/1 | 0/7 |
| New full-source crop verification | 7/9 | 1/2 | 0/1 | 0/7 |

Always returning UNKNOWN would match six references. The new method makes just one non-UNKNOWN decision. Matching seven labels does not establish general usefulness or a fresh-source advantage over either baseline. The original results remain unchanged.

## What the additional checks reveal

For the fork, Spark supports identity, visible outline, mask purity, uniqueness, apparent release and returned tool coverage, but contradicts target extent in both views. The unchanged verifier therefore withholds its original FAIL. This is a model assessment, not an independently established defect in SAM's mask. Incomplete visible material cannot be silently treated as a full-object location.

For the cloth, Spark still reports apparent release and full visibility. Its separate returned-tool coverage assessment contradicts both masks. Independently recomputed foreground proximity is **two source pixels** in the first view, triggering the unchanged three-pixel negative warning; the second distance is beyond three. Coverage and geometry prevent the release assertion from authorizing PASS. This is an image-plane warning, not physical-contact proof or a sensor measurement.

The first clear cup passes both views. Tool confidence is approximately **0.852 and 0.805**, with supported coverage and no near warning. The second clear cup remains UNKNOWN: tool confidence is approximately **0.559 and 0.676**, and coverage is contradicted in both views. Scores remain unchanged; no threshold is lowered. Automatic proposal quality remains an unresolved bottleneck despite recovering one previously refused positive.

## Accounting and independent audit

The persistent journal settles **16/16 requests successfully: eight Spark and eight SAM**, with no retries, errors, uncertain settlements or unscored cases. Reported token totals are **18,146 input and 6,654 output**. Estimated incremental cost is **$0.070963**, not a reconciled provider invoice. Paid historical locators and target/crop proposals are reused, so this is not a cold-start cost comparison.

The timed run section takes **648.847 seconds**, including repeated immutable-input validation, raw replay, rendering and guards. The sum of retained provider-client latency fields is **57.849 seconds**; summed per-stage wall time is **480.337 seconds**, including guards. These distinct scopes do not measure interactive UI latency. Repeated integrity replay dominates this experimental harness; a future performance change must preserve the same bindings and receive separate validation.

The [independent audit](../artifacts/feasibility/crop-verification-audit-20261031.json) matches all sixteen exact request-payload hashes to the journal, re-decodes all eight new SAM responses and eight reused target responses, and independently checks eight crop pixel rectangles, eight target projections, eight tool-mask unions and eight foreground distances. It finds zero retained-record integrity errors. It does not authenticate provider origin or establish model correctness.

Two additional recordings—UR5 cloth episode 1 and cup episode 10—were selected before downloading/viewing and have different video hashes from all 46 prior inputs. Twelve native frames and six proposed cases are prepared privately. Their pinned redistribution card declares CC BY 4.0; original-publisher licensing remains unverified. They remain **unevaluated**, with zero new-source provider requests, because this candidate fails its declared gate. The prepared operator-followed trial also remains unexecuted.

## Current implementation boundary

The separate [admission and recovery corrections](ADMISSION-AND-RECEIPTS-20261031.md) are production fixes. The perception candidate and its recorded results are experimental. All four [CI jobs pass on the exact implementation/plan commit](https://github.com/yerbol1212/proofloop/actions/runs/41199631616), covering both Python versions, frontend checks and clean-wheel installation. The 1,570-test suite and CodeRabbit's zero-issue review establish software behavior, not perception quality.

A separately declared negative-witness diagnostic may explain positively identified target material outside a goal while preserving the original UNKNOWN. It must not rewrite this failed candidate, assert the whole target's location, provide a repair origin, grant PASS or override the original boundary-crossing policy. Useful new-source task verification and actual operator benefit remain the next capability gates.
