# Freshness investigation and production corrections

This investigation found two reproducible production correctness defects. The reproductions use authored transport images and injected model fixtures; they test time and workflow behavior, not Meta perception or physical action.

## Delayed inference and inconsistent recovery

Previously, uploaded pairs evaluated at logical time `1000` even after the server had spent more than thirty seconds waiting for models. The final snapshot correctly showed expired UNKNOWN, but a separate recovery receipt could still claim a verified correction. A fixture advancing time by 31,001 ms reproduced a valid successful-repair receipt alongside stale current evidence.

Current upload evaluation now advances from the declared logical pair epoch by actual elapsed server time. Saved evaluations and recovery receipts use that same decision time. Current workflow replay derives total elapsed directly, avoiding a second addition of already counted inference latency. Expired evaluations remain UNKNOWN, request typed `stale_evidence`, withhold motion proposals and preserve the two-follow-up limit. Camera observations retain their original client-declared timestamps. A shared guard preserves the workbench's existing thirty-second current-pair ceiling, including when a reviewed contract allows a longer observation age.

## Clock rollback and renewed authority

Independent review reproduced a second defect: a pair became stale after 31,001 ms, then became PASS again when the wall clock moved backwards. Workflow continuation was restored without new images or model calls.

Each accepted current pair now has a server monotonic anchor and a nondecreasing elapsed floor: the maximum of nonnegative wall elapsed, monotonic elapsed and previously observed elapsed. Snapshot age, model-result evaluation, current acquisition, workflow replay and pre-repair evidence share it. A clock correction cannot revive expired evidence or freeze aging while monotonic time advances. Only accepted new observations reset the anchor. Missing runtime anchors withhold current authority; historical receipts cannot recreate them.

## Decision time and history

A receipt records its actual evaluation instant. If later disk persistence, process suspension or delivery consumes the freshness window, that genuinely fresh evaluation can remain a valid historical record. Current state and workflow must nevertheless be expired. The regression suite covers that boundary explicitly; the system makes no fresh-at-HTTP-delivery or physical-execution claim.

Existing source timestamps and sealed historical receipts are not rewritten. Old uploads with unrecorded past model latency cannot acquire a reconstructed trustworthy completion time. Historical consistency replay remains distinct from current workflow permission.

## Validation

Fifteen new regressions cover delayed success and wrong placement, recovery consistency, no double-counting, bounded follow-ups, camera/historical handling, later expiry, pre- and post-expiry rollback, monotonic aging during wall regression, missing anchors, exhausted budgets and delayed persistence. The production owner's six-module suite passes 141 tests on Python 3.12 and 3.9. The independent reviewer separately reran the original rollback attack and verified UNKNOWN with workflow blocked and no additional fixture calls.

The full repository suite passes 1,517 tests on Python 3.12 and Python 3.9, with four expected optional-dependency skips on 3.9. CodeRabbit completed an additional review of the eleven changed or added files with zero findings. These software checks do not measure perception accuracy.

The concurrent [context-crop probe](CONTEXT-CROP-PLAN-20261030.md) is a separate experimental perception investigation. It does not alter production acceptance or claim to fix these clock defects through models.
