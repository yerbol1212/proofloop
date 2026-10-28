# Candidate identity challenge: actual feasibility result

The bounded challenger did **not** establish a useful semantic acceptance gate. Only two of four selected cases completed with classifications: it retained support for the clear coffee glass, but abstained on the visible candy positive control. The packet/label negative never received a completed classification. Keep this diagnostic separate from placement acceptance; no existing guided v2 behavior or benchmark result changed.

## What was tested

The [prospective design](SEMANTIC-CHALLENGE-DESIGN-20261028.md) adapts source-context/candidate separation, explicit counterexamples and locally validated structured output from primary Meta and Grounded SAM practices. Spark receives two original full views plus two deterministic context crops around its previously saved locator anchors. It receives positive attributes and excluded concepts, but no desired placement, calibrated acceptance region, expected label or original locator rationale. Each crop keeps source pixels without upsampling and is re-derived byte-for-byte during replay.

The two candy cases reuse the [licensed real ALOHA candy recording](NEW-TASK-TRANSFER-20261028.md). The coffee controls reuse episodes 000004 and 000005 from the [pinned MIT-declared coffee source](ONLINE-VIDEO-SOURCES.md). Four cases across three recordings were selected before semantic inference. All original views and candidate crops were visually reviewed, expected identity support was frozen, and module/schema/prompt/source/crop hashes were recorded. These are development cases on previously exposed sources, not an independent holdout.

Expected identity support describes whether the particular anchor supports a unique, fully visible target matching the reviewed semantic description. It is a different measurement from the historical location check. UNKNOWN does not establish that an object is absent or that a proposed repair happened.

| Fixed candidate pair | Expected identity support | Completed challenge | Result |
| --- | --- | --- | --- |
| Early candy packet/label anchors | Unsupported | None | Two incomplete API attempts; unscored |
| Final visible white oval candy | Supported | UNKNOWN | Candidate, physical-candy attribute and exclusions were uncertain |
| Coffee 000004, low-contrast glass on tray | Supported | None | One incomplete API attempt; unscored |
| Coffee 000005, clear separate glass | Supported | Supported | Both views supported the target attributes and rejected exclusions |

The candy positive's white color and oval shape were supported, while its status as physical candy remained uncertain. That is a real distinction between observed appearance and inferred semantic identity. It is not evidence that the wrong packet anchor was successfully rejected. No duplicate-object, contradictory-attribute or adversarial printed-instruction real-media case was evaluated in this batch.

## Completion limits and cost

The first early-candy request returned incomplete at 2,048 output tokens. Its explicit development rerun, prospectively frozen at 4,096, also returned incomplete. The ledgers report 2,046 and 4,094 reasoning tokens respectively. No classification was substituted for either failed request. The three remaining preselected positive controls each received one 4,096-cap request, without further tuning or retries; the low-contrast coffee request also returned incomplete.

The final batch used **five total attempts: two completed and three incomplete**, at an estimated **$0.076588** including all recorded incomplete-response tokens. The three final control requests took 61.294 seconds in total; that timing excludes the two earlier incomplete requests. It is not the total experiment latency. The initial two rejected responses retained their usage ledgers and observed failure classification, but not their raw provider bodies; the final incomplete response was retained locally.

Meta's [reasoning documentation](https://dev.meta.ai/docs/reasoning) explains that reasoning consumes the output-token budget and supports explicit effort control; the [Responses guide](https://dev.meta.ai/docs/protocols/responses) documents `reasoning.effort`. A future, separately frozen experiment could assess a low-effort classification request. This run did not change that setting after observing failures, and it does not establish that such a change would solve completion or semantic quality.

Among completed cases, **one of two** matched expected identity support. Only **one of the three selected positive controls** retained supported identity; one abstained and one was unscored. There were **zero scored negative controls**. These denominators must remain separate. There is no successful distractor-veto, accuracy-lift, calibrated confidence or independent semantic-proof claim.

## Implemented safeguards and replay

The new diagnostic is versioned `semantic_candidate_v1`. It cannot grant placement PASS, alter confidence scores, choose a different target or modify the frozen `normalized_boxes_v2` adapter. The same Spark model assigns and challenges candidates, so errors can remain correlated even when the challenger sees different context.

The [public allowlisted result](../artifacts/feasibility/semantic-candidate-challenge-20261028.json) contains classifications, safe source hashes/rectangles, reviewed labels, unavailable cases, all five attempt records and recomputed cost totals. It contains no local paths, raw provider responses or image payloads. Failed cases remain `incomplete` and unscored rather than being converted to semantic UNKNOWN.

`proofloop.semantic_experiment.replay_public_experiment(project_root)` rebuilds this result against the freeze chain, actual original guided locator boxes/flags, original source/crop hashes, reproduced crop bytes, strict completed classifications, completed response usage/ledger and incomplete attempt ledgers. It rejects edited decisions or costs. These hashes verify local consistency; they are not signatures, trusted capture timestamps or proof of model origin or physical execution.

Twenty-three authored offline tests passed on Python 3.9 and 3.12. They cover semantic disagreements, multiple matches, attribute/exclusion conflicts, declared image-text reliance, unsupported visibility, duplicate IDs, invalid bindings, changed source/crop bytes, edited costs, original method/selection changes and failed-versus-completed aggregate accounting. They test software control behavior, not whether models detect those real-world failure types.

The useful next step is a new low-effort feasibility contract and richer genuine negative controls, potentially with an operator-reviewed reference exemplar. Do not add this challenger to the live acceptance path until completion, positive retention and negative rejection have been measured together.
