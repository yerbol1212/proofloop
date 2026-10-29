# Adversarial review and new-source testing

This round adds independent implementation and evidence review, a publicly preregistered real-model comparison and a separate exposed-development experiment. It is a repository review under the participant's instruction, not a Meta endorsement or a prediction of hackathon placement.

## Confirmed defects

| Priority | Defect | Correction and evidence |
| --- | --- | --- |
| P2 | Failed Spark calls with lost or malformed usage could omit possible charges from durable tracking reports. | The mutable tracking service examines child attempt ledgers, retaining uncertainty independently of a zero reported subtotal. Six composition regressions distinguish lost/malformed responses from explicitly reported zero usage. [Detailed review](EVIDENCE-REVIEW-20261029.md). |
| P2 | The first anchor-experiment draft skipped ledger recomputation when `input_tokens` was absent. A forged replay could retain failed ledgers while claiming zero attempts, zero cost and no uncertainty. | The independent reviewer reproduced the bypass before any paid anchor call. Finished attempts now require ledger-bound counts and accounting; explicit incomplete accounting retains uncertainty and cannot be relabeled as known zero cost. Regression fixtures preserve the failing ledger and remove token fields. |
| P2 | A SAM parser/decoder exception could discard its attached raw provider response. | The independent reviewer reproduced counted parse failures with no retained response files before anchor inference. Failed responses are preserved privately with bounded source/plan binding and hashes, remain unscored, and cannot become empty detections or trigger a retry. |
| P3 | A point-in-time missing submission lookup said that the original submission had not been admitted. | The workbench now explains that the job may still be in flight and preserves recovery fencing. Desktop/mobile recovery checks use authored fake inference. [Correctness review](SECOND-CORRECTNESS-REVIEW-20261029.md). |

The two defects in the new experimental module were caught after its first offline test pass and private freeze, before paid inference or publication of that plan. The original private freeze remains retained; the corrected implementation receives a separate new freeze. All nine source cases and their labels remain unchanged.

CodeRabbit separately reviewed fifteen staged files and raised one documentation issue: an ambiguous sentence described false `*_incomplete` accounting flags as false completeness flags. The report now names the fields and explains the recorded consistency checks. The adversarial agent's two executable P2 reproductions above were fixed separately and rechecked by that reviewer. An external review's issue count is not evidence that no defects remain.

## Evidence improvements

The [new-family study](NEW-SOURCE-RESULTS-20261029.md) fixes nine paired cases from three recordings in two original families. Its plan was public before its 31 Meta requests. Guided matches 7/9 labels versus Spark's 6/9, but accepts neither clear positive control. Its zero incorrect PASS decisions therefore establish a measured abstention tradeoff, not useful general verification. Both methods' failures, all denominators, the $0.126211 estimated cost and source reuse remain visible.

An additive raw-text auditor checks all eighteen direct Spark/locator answers against retained provider text, exact ordered source images and original report replay. It rejects contradictory parsed copies, duplicate fields, nonfinite numeric encodings, incomplete output and conflicting text lanes. Its public receipts contain counts and hashes; private media and raw responses remain necessary to reproduce those checks. This validates record consistency without authenticating a provider or establishing perception truth.

The [source-preparation command](PROSPECTIVE-SOURCES-20261029.md) pins videos, cards and the Bridge capture metadata row. A fresh actual download reproduced all eighteen JPEG bytes and all nine pair timestamps without model calls. Bridge's slowed playback clock is explicitly separate from physical capture timing. UR5's license basis remains a pinned redistributor declaration, with the original homepage license unverified.

The [anchor-padding follow-up](ANCHOR-PADDING-DEVELOPMENT-20261030.md) changes one fixed spatial prompt parameter on the now-exposed cases. It preserves visibility/identity gates, confidence 0.8 and two supporting views. It is not a production fallback. Its results, including failure or new unsupported acceptance, remain separate from the preregistered pilot.

That follow-up completed all 13 planned SAM requests for an estimated $0.0325 and returned UNKNOWN on all nine cases. It lost the original supported wrong-placement decision and did not retain either clear positive. The reviewer independently re-parsed and officially re-decoded all thirteen SAM outputs, matching the saved objects and masks. Two high-scoring fork masks merged the neighboring spoon, so low confidence alone does not explain this failure. The candidate was rejected for live acceptance. Across both experiments this round used 44 requests, estimated $0.158711, with no retries or reported accounting uncertainty. The [full negative result](../artifacts/feasibility/anchor-padding-results-20261030.json) remains public alongside its pre-inference plan.

## Remaining project risks

### Subsequent elite-field assessment

The [harsh competitive assessment](COMPETITIVENESS-20261030.md) gives a subjective starting score of 53/100 and does not shortlist the current evidence in an exceptionally strong field. Two independent agents reviewed the implementation and project value. They identified useful perception, a genuine operator-followed correction and practical user benefit as more urgent than additional infrastructure.

The new returned-mask challenge and its runner initially allowed inconsistent accounting in failed attempts: erased child ledgers or altered usage/status could contradict retained provider responses. Before any paid call, the reviewers reproduced these paths and the authors bound both failed and completed provider bodies to exact attempt accounting. A later spatial-probe review also caught an authored decoded frame index that was not checked against parent metadata. The final probe rejects that mismatch before inference. These are replay-integrity fixes, not independent authentication of provider origin.

The [concept-mask trial](CONCEPT-MASK-RESULTS-20261030.md) fails its gate: 5/9 exposed reference matches, 0/2 clear positives, one incorrect cloth PASS. Its separate known mixed-mask control succeeds. The [text-gripper follow-up](IMAGE-GRIPPER-RESULTS-20261030.md) also fails, with zero gripper masks in 18 completed calls and no measured geometry. A [six-view manually anchored spatial control](SPATIAL-GRIPPER-CONTROL-20261030.md) matches one near and five separated references. It can withhold the erroneous cloth result in that single exposed diagnostic; it cannot establish automatic localization or recover clear-positive task support. All three remain outside live acceptance.

Each declaration was public before its requests. Independent auditors checked all 44 exact payload hashes against three bounded persistent journals, officially re-decoded all 42 new SAM responses and strictly parsed both new Spark responses. Total estimated cost is $0.116607, with no automatic retries or recorded accounting uncertainty. Prior results and frozen source digests remain unchanged.

All **1,464 Python 3.12 tests pass**; Python 3.9 passes with three expected optional-dependency skips. The new components add 108 tests beyond the earlier 1,356. Linux CI initially exposed a missing FFmpeg dependency in the new real-codec integration. CI now installs it explicitly; local environments without it get an explicit optional integration skip. The complete four-job CI passes on source commit `8c95e4efdbe65aef9be90f50e60a02282b510ae7`.

Two CodeRabbit passes completed with **0 issues** across their seven-file and eight-file scopes. A third pass covering the last spatial probe/publication records failed with `rate_limit`: **“Rate limit exceeded”**, with the free CLI allowance exhausted. It produced no review result. Another CodeRabbit pass requires the allowance to reset or the repository to be connected to an eligible plan through the [CodeRabbit dashboard](https://app.coderabbit.ai). The last probe separately passed independent agent review and seven focused tests; that review is not attributed to CodeRabbit.

The [exact live-trial guide](OPERATOR-TRIAL-20261030.md) is prepared for later at the participant's request. No new operator action, user benefit, physical safety capability or winning probability is claimed.

The measured bottleneck is still useful perception: clear-target retention is poor, and conservative abstention can conceal mistaken visibility judgments. A larger anchor can also segment a distractor. More tests, stronger ledgers and additional model calls cannot independently verify semantic identity, hidden state, physical repair execution or practical demand. A winning-quality demonstration needs an honestly bounded task and inspectable evidence for both successful and refused verification; the available experiments do not establish general manufacturing or safety capability.

The earlier pilot-and-padding suite passed 1,356 tests on Python 3.12. Python 3.9 passed that suite with two expected optional-parser skips. The padding module has 23 focused tests, including both independently reproduced defects; source preparation has 16, and the additive raw-text auditor has 15. No regression test performs paid inference. The twenty historical perception-module digests and nine original pilot implementation digests remain unchanged. The subsequent 1,464-test suite is recorded above.
