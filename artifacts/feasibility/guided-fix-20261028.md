# Guided grounding correction — 28 October 2026

The revised integration resolves both original uncertain cases on a development replay: guided SAM now matches **4/4** labels on the same episode 000004/000005 images, compared with the preserved v1 result of **2/4**. On two additional source episodes reserved until the revision was frozen, guided SAM matches **3/4**, while Spark alone matches **4/4**. The remaining gap is disclosed; this is an integration improvement, not evidence of superiority over Spark or a complete reliability fix.

## Diagnosed causes and change

- **Original episode 000004 dispense:** the old locator set a combined ambiguity flag in both images. No SAM calls ran. That flag conflated uncertain identity, visibility and precise boundaries; its trace does not identify which caused the abstention.
- **Original episode 000005 staging:** the proposed boxes clipped the upper glass rim. Actual SAM confidence was 0.36328125 and 0.2392578125, below the approved 0.8 threshold. The poor box is an observed input defect and a plausible contributor, not an isolated causal experiment.
- **Version `normalized_boxes_v2`:** Spark uses its documented normalized 0–1000 box coordinates. Strict conversion maps outer edges to inclusive source-pixel SAM coordinates with floor(left/top) and ceil(right/bottom) minus one. The conversion uses source dimensions, never expected regions or labels.
- Identity ambiguity, full visibility and approximate boundary precision now have separate fields. A uniquely identified, fully visible object with imprecise edges may receive actual SAM refinement. Missing, multiple, identity-ambiguous, occluded or truncated targets remain unsupported.

The confidence threshold stays **0.8**, source-pair stability stays unchanged, and the cap stays one Spark call plus at most two SAM calls per object, with no automatic retries. SAM still refines a Spark-assigned target; its mask does not independently prove semantic identity. Guided method versions are bound to the reviewed draft, activation and capture provenance.

Primary references: [Spark perception grounding](https://dev.meta.ai/docs/image-understanding#localize-objects-with-perception-grounding), [SAM box prompting and inclusive coordinates](https://dev.meta.ai/docs/sam/segmenting#prompt-with-boxes-and-points). No non-Meta inference was used.

## Exposed development replay

Original contract `e78ef11c8a8086c4d1f49784e2ea186dca020a060e33cd637de875b0506991cf`, original images/labels/calibration, episodes 000004 and 000005.

| Method | Matches labels | Unknown | Incorrect passes | Accepts correct |
| --- | ---: | ---: | ---: | ---: |
| Historical guided v1 | 2/4 | 2 | 0 | 1/2 |
| Revised guided v2 replay | 4/4 | 0 | 0 | 2/2 |

The replay used 12 real requests, 44.860 seconds, estimated $0.050461. The weak episode 000005 staging masks now report 0.9609375 and 0.953125. The formerly ambiguous episode 000004 dispense masks report 0.875 and 0.87109375.

These images informed the debugging work, so this is **development evidence**. Coordinate conventions and uncertainty fields changed together; their separate contributions were not isolated. Original responses and [v1 results](online-coffee-new-source-20261028.json) remain unchanged. [V2 replay summary](online-guided-v2-development-20261028.json).

## Frozen new-source comparison

Two previously untested original recordings, episodes **000006 and 000007**, supplied four cases: two authored FAIL staging placements and two authored PASS dispense placements. Fixed source times, regions and concept descriptions match the earlier experiments. All eight frames and labels were reviewed before inference.

- Frozen contract: `88a3772f85dba0749cbc2d6996740da204d03597ab42cb3eab642e083bac9d88`.
- Frozen guided module SHA-256: `8f77c12f32911d28c3a17bf53d47f2bf30ac021bea0d68a20d6c1dd0487a3a23`.
- The locator received concepts and exact images, without requirements, desired regions or labels. The Spark-only evaluator received the same source pair and approved checks, without labels or SAM output.

| Method | Matches labels | Unknown | Incorrect passes | Accepts correct | Actual requests | Estimated USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Text SAM + checker | 0/4 | 4 | 0 | 0/2 | 8 | $0.020000 |
| Spark alone | 4/4 | 0 | 0 | 2/2 | 4 | $0.043140 |
| Normalized guided SAM + checker | 3/4 | 1 | 0 | 1/2 | 11 | $0.052156 |

Condition wall times: text SAM 7.784 seconds, Spark 47.586 seconds, guided 49.320 seconds. The new comparison used 23 actual requests under a 24-request cap. Combined with the replay, this round used **35 requests**, estimated **$0.165757**. Estimates are ledger calculations, not a provider billing guarantee.

**Remaining unknown: episode 000006 dispense.** The first view has partial rim occlusion; the locator did not support full visibility, so no mask was requested. The second mask reports 0.74609375, below 0.8. The pair cannot provide two qualifying observations. The reviewed PASS label describes visible placement inside the zone; it is retained and the UNKNOWN counts against guided accuracy. It is not relabeled, excluded or counted as successful verification. A clearer supported source pair is needed under this method.

[Machine-readable fresh comparison](online-guided-v2-holdback-20261028.json), [source attribution and hashes](../../docs/ONLINE-VIDEO-SOURCES.md), [implementation and review details](../../docs/GUIDED-PERCEPTION.md).

## Interpretation and validation

The new-source 3/4 result and historical 2/4 result use different recordings; they are not a paired percentage improvement. Both comparisons are tiny, selected, correlated same-camera samples with no ground-truth UNKNOWN cases. They establish neither population reliability nor useful abstention calibration. Spark remains stronger on observed label matching here. More unknowns are not automatically better.

These are authored placement gates on normal prerecorded demonstrations, not natural robot-error labels, robots following ProofLoop, continuous tracking, hidden-step completion, or physical interactive recovery. Eight coffee recordings now provide 16 unique selected placement pairs across experiments; replaying an existing pair does not create another independent case.

The software suite has 334 passing tests after adding normalized-coordinate geometry, uncertainty gates, unchanged-confidence behavior, exact method-version review and evaluation-role regression coverage. The production frontend builds with strict TypeScript checks. Software checks establish contract behavior, not perception accuracy. Raw videos, frames, responses and approval records remain local; public files contain reviewed summaries and source attribution.
