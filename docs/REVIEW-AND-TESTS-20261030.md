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

The measured bottleneck is still useful perception: clear-target retention is poor, and conservative abstention can conceal mistaken visibility judgments. A larger anchor can also segment a distractor. More tests, stronger ledgers and additional model calls cannot independently verify semantic identity, hidden state, physical repair execution or practical demand. A winning-quality demonstration needs an honestly bounded task and inspectable evidence for both successful and refused verification; the available experiments do not establish general manufacturing or safety capability.

The final complete suite passes 1,356 tests on Python 3.12. Python 3.9 passes the same suite with two expected optional-parser skips. The padding module has 23 focused tests, including both independently reproduced defects; source preparation has 16, and the additive raw-text auditor has 15. No regression test performs paid inference. The twenty historical perception-module digests and nine original pilot implementation digests remain unchanged.
