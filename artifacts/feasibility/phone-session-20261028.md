# Recorded phone correction check — 28 October 2026

ProofLoop produced **pass → fail → unknown → pass** on selected frames from one real, 47.4-second phone recording. A genuine Meta Spark call compiled the supplied location requirements; sixteen genuine Meta SAM image calls supplied the visual evidence. This supports the recorded observation loop for this scripted case. It does not establish an interactive planner-directed repair system or general camera accuracy.

## Source and review

The caller supplied the recording as owned phone footage. Eight distinct JPEG frames were extracted with rotation applied and no interpolation. They are 720 × 1,280 pixels. The original video is encoded at 1,920 × 1,080 with 270° display rotation, so its displayed orientation is portrait.

The paper's handwritten **LEFT** label names the upper displayed region; **RIGHT** names the lower displayed region. Region names therefore follow the physical labels rather than screen-left and screen-right. The normalized rectangles are `left = [0, 0.11, 1, 0.48]` and `right = [0, 0.495, 1, 0.9]`.

The assistant reviewed the extracted views, selected offsets and calibration under the user's authorization to proceed. **The human did not review the exact contract or timestamps in this run.** The scoped session approval records `assistant_under_user_authorization`, with exact human review explicitly false. The original compiler draft still records `approved: false`; the session uses a separate reviewed contract with that provenance. Approval records are assertions rather than independent proof of a review.

The scene contract declares one unique fork and one unique spoon. Spark compiled location checks requiring the fork in LEFT and spoon in RIGHT, each with confidence threshold 0.8, two stable samples and a 30-second evidence age limit. The protocol hash is `bec50a01b7bc713c80894e786c32f8cc24176a54f9b591b4b55361cb918287aa`.

## Observed results

All times below are decoded video timestamps relative to the first video frame, **not wall-clock capture times**. Stage boundaries and labels were supplied annotations; detecting transitions automatically was not tested.

| Stage | Actual sampled times | Fork check | Spoon check | Combined result |
| --- | --- | --- | --- | --- |
| Initial arrangement | 3.000s, 5.000s | Pass: located LEFT | Pass: located RIGHT | **Pass** |
| Wrong placement | 16.000s, 18.000s | Fail: positively located RIGHT | Pass: located RIGHT | **Fail** |
| Covered | 29.000s, 31.000s | Unknown | Unknown | **Unknown** |
| Corrected arrangement | 42.001667s, 44.001667s | Pass: located LEFT | Pass: located RIGHT | **Pass** |

SAM returned one fork and one spoon detection in each of the six uncovered frames. All detected masks decoded with the official Meta parser. Their confidence values ranged from 0.953216 to 0.968750; these are model scores, not calibrated probabilities of correctness. SAM returned zero detections for each concept in both covered frames. The checker retained `unknown` and did not infer that an object was absent.

Before each later stage, the runner invalidated prior evidence. All three immediate invalidation snapshots were `unknown`. The corrected stage used a fresh attempt and later source frames, rather than reusing the initial pass. Every source frame still matched its recorded hash, and all sixteen inference calls completed without recorded inference or storage failures.

At the wrong-placement stage, the bounded planner proposed `move-fork-right-to-left`, followed by `inspect-fork`. It expanded four states and found a plan costing **1.25 declared planning units**: 1 for a move and 0.25 for inspection. These units are neither seconds nor dollars. The minimum-cost claim applies only to the supplied finite deterministic action model, which does not model physical obstacles. Predicted effects remain hypothetical and never create real observations.

**The person recorded the correction before this inference run and did not see or follow the later planner proposal.** The final pass verifies a later arrangement in prerecorded footage; it does not verify execution of that proposal or an interactive feedback loop.

## Runtime and estimated cost

The single `muse-spark-1.3` compilation call took **4.987 seconds** and reported **433 input tokens and 1,203 output tokens**, including 1,096 reasoning tokens. The sixteen `sam-3.1` image calls had a **1.2425-second median** and **0.605–3.015-second range**. Their serial call latencies sum to 22.863 seconds. This small run is not a latency benchmark or a measurement of live camera frame rate.

At the [published Standard rates](https://dev.meta.ai/docs/pricing-rate-limits), the Spark estimate is **$0.005654** and sixteen SAM image calls cost an estimated **$0.040000**, for **$0.045654 total**. This is list-price arithmetic for these completed calls, not an invoice, credit-balance measurement or billing guarantee.

The [reviewed JSON report](phone-session-20261028.json) contains the protocol, calibration, source hashes, actual timestamps, per-frame counts and confidence scores, check statuses, invalidation trace, hypothetical repair proposal and usage arithmetic. The original video, extracted images, encoded masks, provider response IDs, credentials and private authorization text remain local and are excluded from this public report.

## Limits and next evidence gate

The location checks depend on the declared unique-target scene contract. These detections do not prove exact counts, absence or complete visibility. Source ownership is caller-declared; offsets and regions were assistant-reviewed. This is one scripted case with selected views, rather than a held-out accuracy study, continuous tracking evaluation or industrial reliability result. It does not establish general procedure learning or completion of all ProofLoop features.

The next gate is an interactive session: observe a wrong arrangement, display the proposed move to the person, invalidate evidence when the move starts, and verify fresh frames after that person follows the proposal. Additional setups and deliberately ambiguous views should then test whether unsupported conclusions consistently remain `unknown`.
