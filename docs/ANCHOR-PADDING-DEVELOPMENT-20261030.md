# Fixed anchor-padding development test

The guided-v2 condition in the [new-family pilot](NEW-SOURCE-RESULTS-20261029.md) rejects both clear cup placements. Inspection of those now-exposed outputs shows imprecise Spark anchors and low SAM scores. This follow-up tests one deterministic anchor change across all nine existing cases. It is development on exposed inputs, with known labels, and cannot retrospectively improve the original pilot or establish fresh-source reliability.

The change expands each eligible source-pixel box symmetrically by `ceil(max(width,height)/20)` pixels, clamped to the image. That is 32 pixels for the 640 × 480 sources. Cached Spark locators supply identity, visibility and the original anchor; no new locator inference, target reference box, acceptance region or scoring label is supplied to SAM. The existing single-candidate, unambiguous-identity and full-visibility conditions still determine eligibility. All nine cases stay in the denominator, including both clear positive controls.

The [hosted SAM documentation](https://dev.meta.ai/docs/sam/segmenting) requires choosing text concepts or structured spatial prompts rather than mixing them. This experiment retains the existing structured box syntax. It does not assume that the richer open-source SAM interface is the hosted API contract.

The new `proofloop.anchor_padding_experiment` module is separate from production acceptance and the frozen guided-v2 implementation. It freezes exact code, source bytes, prior approvals, cached locator responses, original/padded boxes, eligibility and request limits before new calls. The [machine-readable plan](../artifacts/feasibility/anchor-padding-plan-20261030.json) is published before padded-mask inference. It preserves all nine cases and exactly 13 eligible frames under an 18-attempt maximum. Its one-shot runner uses fresh one-request clients, no retries, persistent total admission, HTTP authentication/billing/rate-limit stops and private raw response retention. Failed or unattempted necessary requests leave a case unscored rather than generating UNKNOWN. Locator-withheld views retain their original unsupported status.

Scoring uses the original grounding/checker with confidence 0.8 and two distinct supporting views. Provider confidence is retained unchanged. Neither padding nor a prompted mask independently verifies semantic identity; a larger anchor can include another object. Saved response, source, plan and accounting checks support offline replay. This is a candidate experiment, with no implicit fallback or live acceptance promotion.

The public plan and outcome artifacts linked here retain all nine cases, their positive and negative denominators, missing predictions, per-frame scores and actual incremental SAM cost. Original Spark/locator and baseline costs are reused evidence, not newly incurred costs. A successful development outcome would still require new-source and identity-sensitive validation before any production change.

## 30 October result: fixed padding failed

The corrected plan was published in [commit f82745c4](https://github.com/yerbol1212/proofloop/commit/f82745c437806fb6f5f6d36d4bf4e4140e0e40c9) before new requests. Its frozen plan hash is `244e8464cc49f633ebabaad3b505caa3eb5f078fc911936a9f17201266062c8b`. The actual persistent journal allowed exactly 13 attempts, tighter than the plan's 18-attempt ceiling. [All outcomes and incremental accounting](../artifacts/feasibility/anchor-padding-results-20261030.json) are preserved.

Padding returned **UNKNOWN on all nine cases**: **6/9** reference matches, **0/2** clear positive placements retained and **0/7** unsupported PASS decisions. The original guided method matched 7/9, retained 0/2 positives and made 0/7 unsupported PASS decisions. Padding therefore lost its supported wrong-placement FAIL without recovering a clear positive. Zero unsupported PASS here reflects complete abstention, not a useful completed verifier.

| Case | Frozen reference | Direct Spark, original | Guided v2, original | Padded SAM, development |
| --- | --- | --- | --- | --- |
| Bridge fork pair 0 | FAIL | FAIL | FAIL | **UNKNOWN** |
| Bridge fork pair 1 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| Bridge fork pair 2 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| UR5 cloth pair 0 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| UR5 cloth pair 1 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| UR5 cloth pair 2 | UNKNOWN | PASS | UNKNOWN | UNKNOWN |
| UR5 brown cup pair 0 | PASS | PASS | UNKNOWN | UNKNOWN |
| UR5 brown cup pair 1 | PASS | UNKNOWN | UNKNOWN | UNKNOWN |
| UR5 brown cup pair 2 | UNKNOWN | PASS | UNKNOWN | UNKNOWN |

All cases completed as evaluations; none were converted from request errors into UNKNOWN, and none are missing predictions. Five source views stayed withheld by their original locator gates. The remaining thirteen views received one new SAM request each, with no new Spark calls or automatic retries. The two earlier Spark unsupported acceptances remain historical results, not new baseline inferences.

The lost fork FAIL exposes a distractor concern. Both new SAM scores exceeded 0.8, approximately **0.855 and 0.875**, but visual comparison of the original source images and decoded masks showed foreground covering the blue fork, neighboring red spoon and additional fragments. The first returned box, x136–339 in a 640-pixel-wide source, crosses LEFT's x313.6 boundary. It cannot supply the second stable, wholly LEFT observation. High confidence did not ensure a single semantically correct target mask. This is an assistant visual diagnosis of retained masks, not an independently annotated instance-segmentation score.

The clear cup controls still fail to supply supported observations. Their new score pairs are approximately **0.293/0.613** and **0.668/0.598**, below the unchanged 0.8 threshold. The nested-cup UNKNOWN control also remains below threshold. No new PASS required further acceptance review, and neither the mask scores nor abstention establish correct identity or detection of physical occlusion.

The follow-up used **13 SAM attempts**, estimated **$0.032500** incrementally, with **17.733 seconds** summed request/local-work elapsed time. All journal entries settled `ok`; no uncertain, rejected, unresolved or incomplete-accounting entries were recorded. An independent offline check regenerated the full summary, verified the published plan binding and re-parsed/re-decoded all thirteen raw provider responses with the official SAM parser; every decoded object and mask matched its saved counterpart. These checks establish recorded consistency, not provider attestation or perception quality. The original pilot plus this exposed-development follow-up used **44 attempts**, estimated **$0.158711** in total.

The experiment does not support promoting this padding rule, lowering the confidence threshold or adding a silent retry. Original guided v2 and production acceptance remain unchanged. Any further anchor or identity intervention would require its own development test followed by new prospective sources; it cannot rewrite this failed result.
