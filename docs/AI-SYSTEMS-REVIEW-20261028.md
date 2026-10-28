# ProofLoop: critical AI systems review

28 October 2026. This is a technical review using the perspective of an AI systems evaluator, not a statement from Meta or evidence of its internal priorities.

The strongest product claim is **inspect why completion is unproven, request the missing evidence, and replay the resulting decision**. Current measurements do not show better label matching than Spark alone. The system needs a demonstrable contribution in task contracts, evidence handling and recovery, rather than a claim that adding another model automatically improves accuracy.

| Order | Reviewer comment | Concrete next change | Acceptance evidence | Status |
| --- | --- | --- | --- | --- |
| 1 | The audit must validate its own calculations. At review start, some imported reports could supply contradictory stored metrics or check verdicts. | Shared strict report validation, metrics recomputation, grounded-check replay and guided-version validation before export or display. | Deliberate metric/check/source/version mutations are rejected; all four actual published experiments retain identical results. | Implemented; 49 benchmark/lab tests pass |
| 2 | At review start, a fresh PASS had no single durable explanation of recovery. | Link the original evidence, failed checks, proposed actions, invalidation and fresh result in an offline-verifiable recovery receipt. Show the chain in the workbench. | Wrong repairs remain FAIL, uncertain evidence cannot close recovery, old images and task changes cannot satisfy a pending repair, altered media/masks/contracts/plans/parents fail replay. | Implemented; local UI and offline fixtures verified; physical interactive closure remains pending |
| 3 | UNKNOWN should lead to a bounded acquisition decision. | Preserve typed object/view gaps, request specific evidence, enforce a finite attempt budget and stop when no valid source pair is available. | No searching until PASS; no repeated hashes or retroactive frames; source/time/calibration changes are explicit. | Implemented typed object/view requests and server budget; fixed future-evidence replay completed |
| 4 | The tested task must be portable into the product. | Support one to six targets and reviewed region names, so the coffee task can load unchanged instead of being recast as two objects in LEFT/RIGHT. | One-glass task loads; renamed/swapped regions require review; changed calibration invalidates evidence. | Implemented one-to-six targets, named regions and authored one-glass preset; real portability beyond selected tasks remains limited |
| 5 | At review start, the evaluation lacked a ground-truth UNKNOWN cohort and semantic distractors. | Freeze temporal inconsistency, occlusion, multiple-target, boundary and changed-requirement cases before inference. Separate development exposure from untouched source/task evaluation. | UNKNOWN labels penalize false acceptance; source counts cannot be inflated by replay; report per-check results and extra observation cost. | Temporal 7/8 versus 8/8; fixed future pairs 4/6 versus 6/6; one candy recording 2/2 for both; semantic identity and broader environment transfer remain unresolved |
| 6 | Placement is narrower than procedure verification. | Deliver one authored workflow with explicit observable preparation gates and a separately labeled operator-declared continuation. | A failed, unknown or expired gate blocks continuation; a click is not labeled observed physical action. | Authored preparation checkpoint and separate continuation declaration implemented; full procedure and physical operation remain unverified |
| 7 | A judge needs to reproduce the result and understand its practical need. | Provide a clean public-clone walkthrough with licensed source acquisition, fixed hashes, frozen task/labels, a working demo and a short story around one workflow. | Another person can follow the instructions; source assets and new team contributions are distinguishable; no claim of validated customer demand without evidence. | Reproduction CLI executed on a fresh checkout; three-minute script prepared; another-person walkthrough and practical demand remain unvalidated |

## Findings behind the first changes

The review reproduced three import/replay gaps without modifying published reports: a changed stored accuracy could be exported; a private check status could change while its single aggregate-only public case record remained unchanged; an unrecognized guided observation version could reach evaluation. These are consistency gaps in handling local records. The published measurements themselves remain internally consistent.

At review start, starting a repair invalidated old evidence and cleared the current proposal. A subsequent passing pair verified the current goal, with no single durable artifact linking the earlier failure, intended action and resulting evidence. The new [recovery receipt](RECOVERY.md) now makes that relationship inspectable without claiming to prove that the proposed movement occurred. The [temporal stress pilot](../artifacts/feasibility/temporal-stress-20261028.md) adds measured uncertainty cases while preserving the observed baseline advantage.

## Changes now implemented

UNKNOWN requests now read typed source evidence rather than classifying generated reason strings. They identify the target, affected views and missing visibility, identity, score, boundary or stability support. The server enforces two follow-up submissions through an unresolved inspection chain; another inspection button does not reset the budget, and model errors consume a submitted attempt. The plan cannot authorize a physical repair, infer absence, relax a threshold or scan until it finds PASS.

The [fixed future-evidence run](../artifacts/feasibility/fixed-followup-20261028.md) tested six preselected pairs from three **already exposed** coffee recordings. Guided matched **4/6**, Spark **6/6**. Guided resolved one of two trajectories with future evidence after two follow-ups; Spark resolved both after one. A terminal third timeline remained UNKNOWN for both. Source ordering, image hashes, metrics, locator/mask replay and unused counterfactual costs are checked by the public replay helper. This is prerecorded timeline analysis, not physical response to a request.

The editor now accepts one to six targets and two named regions. The authored coffee preset loads the tested one-glass DISPENSE/STAGING contract as an unapproved draft. Exact review is still required; another camera requires new calibration. The [candy task check](NEW-TASK-TRANSFER-20261028.md) adds one different manipulation recording, with guided **2/2** and Spark **2/2**. Its early packet/label anchors were rejected by low SAM scores. Semantic identity was not independently solved, and there are no FAIL labels in that two-case check.

The [workflow panel](WORKFLOW.md) gates a preparation checkpoint on exact task review, current supported PASS and replayed evidence. It records continuation as a separate operator declaration, never an observed physical action. Historical, failed, uncertain, expired or altered evidence blocks the gate. This completes a bounded authored preparation flow in software, not full procedure verification.

The [reproduction guide](REPRODUCING.md) was exercised in a fresh source checkout: one exact pinned clip/card downloaded, four original frames decoded, an unapproved contract written and zero API calls made. The [three-minute demo script](DEMO-SCRIPT.md) separates external recordings, event-created code, measured outputs and unresolved claims. A walkthrough by another person and validation of practical customer need remain open.

## Next review priorities

1. Validate the missing physical loop: a current operator follows the displayed request and supplies new evidence, with actual task/revision/timing provenance.
2. Test semantic distractors and wrong placements across new tasks and camera conditions with frozen labels and an unchanged Spark baseline. Report unresolved identity, acceptance and abstention rather than only headline matches.
3. Extend beyond placement only where an observable predicate and separately validated evidence method exist. The current continuation declaration cannot certify a hidden action or complete procedure.
4. Have another person reproduce the public clone and review whether the preparation gate helps their work. Treat market value as a hypothesis until that happens.

## Limits that improvements must preserve

- Spark assigns semantic identity in guided mode; SAM refines the prompted mask. A good mask does not independently prove identity.
- The unresolved episode 000006 pair is at 20/21 seconds in a 22-second recording. Two wholly new later frames at least one second apart cannot fit after observing 21 seconds. Reusing 20.5 seconds would not be a causal new capture.
- A later state may have physically changed. A later PASS never rewrites an earlier UNKNOWN or proves that a robot followed our proposal.
- The current camera zones are image-coordinate calibration, not 3D physical regions. Another camera needs separately reviewed calibration.
- Confidence thresholds and observed failures remain intact. Test fixtures, prerecorded inference and fresh physical operation retain distinct provenance.

Relevant primary sources: [Meta SAM prompting and coordinate conventions](https://dev.meta.ai/docs/sam/segmenting), [Meta Spark perception grounding](https://dev.meta.ai/docs/image-understanding#localize-objects-with-perception-grounding). The recommendations above are engineering judgments based on the actual implementation and measured results.
