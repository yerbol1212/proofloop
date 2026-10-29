# Independent harsh review of ProofLoop

29 October 2026. This review uses the perspective of a skeptical hackathon judge and AI systems reviewer. It does not represent Meta or predict a judging outcome. Findings describe the repository before the fixes in this milestone.

The project would not yet earn a top ranking from this reviewer. Its evidence discipline is stronger than its demonstrated practical capability. Additional infrastructure alone cannot establish perception quality, customer need or a completed physical loop.

## Ranked findings

### 1. P1: A public judge cannot experience the strongest contribution

`proofloop/video_lab.py:85–100` discovers private experiment files. Without those files, `web/src/VideoLab.tsx:290–295` displays “No reviewed runs are available.” The packaged replay exists, but the durable demo in `proofloop/cli.py:16–33` is JSON-only. Clean-install CI can therefore pass without providing a useful in-app walkthrough.

Smallest valuable fix: add a read-only public replay inspector in Video lab using the existing replay endpoint. Show both cohort comparisons, four selectable source timelines, gap codes, confirmation evidence and gripper warnings. Label the measurements as cached development evidence, with original image pixels and new inference absent. If comparing a checkpoint across methods, use recomputed decisions rather than expected labels as predictions.

Acceptance: an installed wheel with empty data storage and no key shows the known contact frame `remote_new:59` as UNKNOWN, the unnecessary warning at `screwdriver_new:5`, the unscored positive and all denominators. Opening and using the inspector makes no provider requests and does not mutate the active inspection task.

### 2. P1: Additional perception machinery has not established incremental user value

`README.md:24–33` reports guided 3/4 versus Spark 4/4, 7/8 versus 8/8, and 4/6 versus 6/6. `README.md:49` reports 3/6 and 4/7 retained clear tracking checkpoints. These results support an evidence-control contribution, not a perception-superiority claim. The warning itself loses one clear view compared with current-view confirmation alone.

Smallest fix now: expose the tradeoff and remaining failures in the public inspector and demo. Do not relax acceptance gates or change the historical comparison to improve a headline.

Research acceptance: freeze genuinely new source families, camera conditions, wrong placements, confusable distractors and full-disappearance cases before inference. Compare the unchanged baseline and report retention, unsupported acceptance, abstention, latency and cost. Exposed recordings and rule-selected retests cannot establish held-out transfer.

### 3. P1: The defining physical loop remains unvalidated

`README.md:33`, `docs/FEASIBILITY.md:20` and `docs/RECOVERY.md:87` say that no current operator has followed a displayed request and supplied new corrective evidence. The phone recording predates its planner proposal. A replay or operator declaration cannot replace this validation.

Required experiment: retain a current task review, supported mistake, displayed repair request, actual operator correction, new paired observations and verified receipt. Include a wrong repair and an UNKNOWN branch. This is an observation gap, not a software bug that another abstraction fixes.

### 4. P2: Doctor can succeed for an unusable workbench

Before this fix, `proofloop/cli.py:43` required only supported Python and replay-file presence. `proofloop/installation.py:57` checked only that the replay file existed. Missing frontend files, unwritable storage or corrupted replay bytes could leave `doctor` exiting zero even when the advertised offline workbench was unusable.

Smallest fix: make the default doctor profile check the offline workbench, including actual replay validation, frontend availability and writable data storage. Provide an explicit core-only profile for intentional Python-only installations. Report optional live dependencies separately; do not authenticate a key, call a provider, create storage or print private paths.

Acceptance: missing frontend assets, unwritable data storage and a corrupted replay bundle fail the workbench profile with nonzero CLI exit. A valid core-only installation can pass the explicit core profile without UI assets. A complete offline installation passes with zero network/model calls and no filesystem writes.

### 5. P2: Mode selection and demo content lag behind the evidence

`web/src/TaskEditor.tsx:46–47` defaults to concept-only SAM, while `docs/FEASIBILITY.md:23–26` records its failures on the coffee cohorts. The owned cutlery clip did work, so globally switching the default to guided would also lack evidence. The existing hint at `web/src/TaskEditor.tsx:275–282` does not explain that empirical scope. `docs/DEMO-SCRIPT.md:25–46` emphasizes earlier coffee/candy results and omits the latest tracking and service work.

Smallest fix: explain the measured scope beside the perception choices and revise the demo around one coherent contribution: unsupported completion, specific missing evidence, current-ending confirmation, conservative warning tradeoff and replayable record. Keep live placement acceptance separate from experimental tracking. A default-mode change needs fresh task-specific comparative evidence.

### 6. Product limitation: demand and independent usability are unvalidated

`docs/AI-SYSTEMS-REVIEW-20261028.md:15,33,47–50` records the absence of independent usability and practical demand evidence. Cutlery, coffee, robotics diagnostics and developer infrastructure currently compete for the primary user story.

Next useful validation: ask an unfamiliar person to reproduce the public walkthrough and complete one narrowly defined inspection task. Record setup failures, completion time and whether a missing-evidence request helps them choose their next action. Additional software tests cannot substitute for this observation.

## Implementation order and boundaries

Implement the public replay inspector and truthful doctor behavior first, then update the demo and mode guidance. Preserve frozen perception implementations, source hashes, cohorts, thresholds and historical results. New accuracy and physical-loop claims require new observations. This review used repository inspection only and made no paid API calls or changes to experiment evidence.

## Confirmed adversarial implementation defects

A second independent agent reproduced three additional defects with no model calls: the store admitted 65 distinct pending jobs without a capacity bound; offline HTTP returned 202 for a new inspection; and an SDK response shaped as `{"error":"bad-envelope"}` raised AttributeError. The changes add atomic limits of 32 pending jobs and 256 MiB retained payloads, reject new offline work before capture consumption, preserve exact historical recovery and normalize malformed errors. A smaller CLI recovery gap was also fixed by printing the cached-demo idempotency key before dispatch. These are software defects, distinct from the unresolved research and product limitations above.

## Changes implemented after the review

The public inspector, empirical mode guidance, updated three-minute demo, profile-aware doctor, queue/storage bounds, early offline rejection and client/CLI recovery fixes are implemented. Their verification is recorded in the milestone report after tests complete. The findings about incremental perception value, current physical correction and independent demand/usability remain open. No gate, confidence threshold, frozen perception module or historical score was changed to make the project look stronger.

## Verification record

The full suites passed 1,278 tests on Python 3.12 and Python 3.9 (two expected optional-parser skips on 3.9), along with 14 Node recovery checks and the strict frontend build. Empty-data desktop/mobile browser QA exercised the public walkthrough, error/retry behavior, source proofs and unchanged active-task binding with zero Meta requests. A fresh wheel outside the checkout served its bundled UI, validated all 32 public checkpoints and completed cached tracking without a provider attempt. [The sanitized check record](../artifacts/feasibility/harsh-review-checks-20261029.json) preserves the scope and open research findings.

CodeRabbit CLI 0.9.1 completed its review of the 24 milestone files on retry and raised **0 issues**. Its first attempt failed with `Connection failed: WebSocket closed`; that failed attempt supplied no review. Final wording-only clarifications followed the successful check. CodeRabbit results are distinct from the independent agents’ confirmed defects and do not resolve the research limitations.
