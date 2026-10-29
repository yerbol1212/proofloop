# Negative-only gripper warning: cached development retest

The [model-only boundary guard and strict gripper-separation trials](CONTACT-RESULTS-20261029.md) remain published, including their failures. Strict separation reduced exposed clear retention to 0/6. This separate method is selected **after inspecting those failures and four exposed recordings**. It is a cached development retest, not a held-out comparison or fresh evidence of generalization.

The method reuses exact source-bound target and reviewed gripper responses, frozen references, v2/v3 observations and the original greedy/current-view policies. It admits **zero new API calls**, no retries, and no replacement of the earlier uncertain network operation. A new receipt freezes implementation hashes, all raw dependencies, unchanged historical methods and the canonical strict-result reference before derivation. The receipt's safe public plan and code are pushed before running the retest.

The source-space Chebyshev foreground-pixel-center margin remains **three pixels**. Valid decoded near-gripper proposals may add a negative warning even if their reported gripper confidence is weak. This is a different, asymmetric use of evidence: weak proposals can withhold a positive decision but can never qualify one. Reported scores remain visible. No-near, far, missing, weak or unavailable geometry does not verify absence, separation, complete visibility, identity or physical contact. A wrong proposal can cause unnecessary abstention.

For each v2/v3 guard and greedy/current-view policy:

1. Apply a near-proposal warning as a current-frame hard gap and reset confirmation history.
2. Recompute the unchanged policy using the same source order, 0.8 target confidence, distinct-image requirement, two-view count and 500-ms spacing.
3. Intersect its published supported rows with the corresponding unchanged policy's supported source/frame/box/evidence rows. A new gap can otherwise change greedy timing and accidentally introduce a new positive. Update full policy rows, final status, counts, recovery metrics and derived policy hash together.
4. Validate that each published current-view confirmation chain ends at the current frame and contains only source-bound, positive, distinct, spaced, risk-free evidence from the same target continuity segment.

Candidate-only diagnostics also remove existing support on warnings; they never promote UNKNOWN or unscored results. The unchanged baselines, failed strict gate, every original denominator and cost remain accessible in hash-bound canonical public artifacts. Localization sensitivities remain 0.2, 0.3 and 0.5; references and later boxes remain scoring-only.

The retest can demonstrate a specific regression fix and software invariants on these cached recordings. It cannot establish robust handling of unseen grippers, long occlusions, complete disappearance or identity switches. It does not change live placement acceptance or authorize physical action.

## Local execution

These commands require the retained private receipts, original media and provider responses from the two preceding trials. Public artifacts alone cannot reconstruct masks. From the repository root, create a new receipt and export its safe plan before pushing:

```sh
.venv/bin/python -m proofloop.gripper_veto_experiment freeze \
  --strict-frozen artifacts/local/gripper-experiment-20261029-01/trial/freeze.json \
  --baseline-public artifacts/feasibility/gripper-results-20261029.json \
  --output artifacts/local/gripper-veto-experiment-20261029-01/trial
.venv/bin/python -m proofloop.gripper_veto_experiment public-plan \
  --frozen artifacts/local/gripper-veto-experiment-20261029-01/trial/freeze.json \
  --output artifacts/feasibility/gripper-veto-plan-20261029.json
```

After the plan is published, `run --frozen <receipt>` derives the result once. Subsequent `replay --frozen <receipt>` calls validate it offline. Neither action dispatches model inference. Original receipts and public files remain unchanged.
