# ProofLoop: critical AI systems review

28 October 2026. This is a technical review using the perspective of an AI systems evaluator, not a statement from Meta or evidence of its internal priorities.

The strongest product claim is **inspect why completion is unproven, request the missing evidence, and replay the resulting decision**. Current measurements do not show better label matching than Spark alone. The system needs a demonstrable contribution in task contracts, evidence handling and recovery, rather than a claim that adding another model automatically improves accuracy.

| Order | Reviewer comment | Concrete next change | Acceptance evidence | Status |
| --- | --- | --- | --- | --- |
| 1 | The audit must validate its own calculations. At review start, some imported reports could supply contradictory stored metrics or check verdicts. | Shared strict report validation, metrics recomputation, grounded-check replay and guided-version validation before export or display. | Deliberate metric/check/source/version mutations are rejected; all four actual published experiments retain identical results. | Implemented; 49 benchmark/lab tests pass |
| 2 | At review start, a fresh PASS had no single durable explanation of recovery. | Link the original evidence, failed checks, proposed actions, invalidation and fresh result in an offline-verifiable recovery receipt. Show the chain in the workbench. | Wrong repairs remain FAIL, uncertain evidence cannot close recovery, old images and task changes cannot satisfy a pending repair, altered media/masks/contracts/plans/parents fail replay. | Implemented; local UI and offline fixtures verified; physical interactive closure remains pending |
| 3 | UNKNOWN should lead to a bounded acquisition decision. | Preserve typed object/view gaps, request specific evidence, enforce a finite attempt budget and stop when no valid source pair is available. | No searching until PASS; no repeated hashes or retroactive frames; source/time/calibration changes are explicit. | Planned |
| 4 | The tested task must be portable into the product. | Support one to six targets and reviewed region names, so the coffee task can load unchanged instead of being recast as two objects in LEFT/RIGHT. | One-glass task loads; renamed/swapped regions require review; changed calibration invalidates evidence. | Planned |
| 5 | At review start, the evaluation lacked a ground-truth UNKNOWN cohort and semantic distractors. | Freeze temporal inconsistency, occlusion, multiple-target, boundary and changed-requirement cases before inference. Separate development exposure from untouched source/task evaluation. | UNKNOWN labels penalize false acceptance; source counts cannot be inflated by replay; report per-check results and extra observation cost. | Temporal pilot complete: guided 7/8, Spark 8/8; distractors and broader task/environment transfer remain pending |
| 6 | Placement is narrower than procedure verification. | Deliver one authored workflow with explicit observable preparation gates and a separately labeled operator-declared continuation. | A failed, unknown or expired gate blocks continuation; a click is not labeled observed physical action. | Planned after the evidence foundation |
| 7 | A judge needs to reproduce the result and understand its practical need. | Provide a clean public-clone walkthrough with licensed source acquisition, fixed hashes, frozen task/labels, a working demo and a short story around one workflow. | Another person can follow the instructions; source assets and new team contributions are distinguishable; no claim of validated customer demand without evidence. | Planned |

## Findings behind the first changes

The review reproduced three import/replay gaps without modifying published reports: a changed stored accuracy could be exported; a private check status could change while its single aggregate-only public case record remained unchanged; an unrecognized guided observation version could reach evaluation. These are consistency gaps in handling local records. The published measurements themselves remain internally consistent.

At review start, starting a repair invalidated old evidence and cleared the current proposal. A subsequent passing pair verified the current goal, with no single durable artifact linking the earlier failure, intended action and resulting evidence. The new [recovery receipt](RECOVERY.md) now makes that relationship inspectable without claiming to prove that the proposed movement occurred. The [temporal stress pilot](../artifacts/feasibility/temporal-stress-20261028.md) adds measured uncertainty cases while preserving the observed baseline advantage.

## Limits that improvements must preserve

- Spark assigns semantic identity in guided mode; SAM refines the prompted mask. A good mask does not independently prove identity.
- The unresolved episode 000006 pair is at 20/21 seconds in a 22-second recording. Two wholly new later frames at least one second apart cannot fit after observing 21 seconds. Reusing 20.5 seconds would not be a causal new capture.
- A later state may have physically changed. A later PASS never rewrites an earlier UNKNOWN or proves that a robot followed our proposal.
- The current camera zones are image-coordinate calibration, not 3D physical regions. Another camera needs separately reviewed calibration.
- Confidence thresholds and observed failures remain intact. Test fixtures, prerecorded inference and fresh physical operation retain distinct provenance.

Relevant primary sources: [Meta SAM prompting and coordinate conventions](https://dev.meta.ai/docs/sam/segmenting), [Meta Spark perception grounding](https://dev.meta.ai/docs/image-understanding#localize-objects-with-perception-grounding). The recommendations above are engineering judgments based on the actual implementation and measured results.
