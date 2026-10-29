# Contact error and current-view retention

The new current-view confirmation policy improves the original exposed recording's clear-checkpoint retention from **2/6 to 4/6**, with **0/10 unclear checkpoints supported**. A new model-only boundary rubric and enlarged crop still classify the known gripper-contact frame clear. That failed intervention is retained alongside the successful temporal change. Independent gripper-mask separation is evaluated separately.

## First frozen comparison

The [protocol](CONTACT-PROTOCOL-20261029.md), [prospective plan](../artifacts/feasibility/contact-plan-20261029.json), source-bound implementations and tests were pushed in commit `7ccdbdbab01caebf38adf6b1017294a527a5d060` before new inference. Freeze hash: `614ec17ec81a63c2a98668f7e17c94041656247e0b6a1ea6dc7d5a9ee7967352`.

Episode 2 reuses all sixteen previously published references, exact target tracks and full-frame v2 judgments. Episode 3 adds two new licensed online recordings, 100 sampled native-pixel frames and sixteen references frozen before model calls. The source families, cameras and tasks remain exposed. Sources and licenses are retained in the protocol. Initial target/tape seeds are authored on frame zero; later boxes and labels enter scoring only.

| Cohort | Guard / policy | Correct clear checkpoints supported | Unclear checkpoints supported |
| --- | --- | ---: | ---: |
| Exposed episode 2 | v2 / original greedy | 2/6 | 0/10 |
| Exposed episode 2 | v2 / current-view | 4/6 | 0/10 |
| Exposed episode 2 | v3 / original greedy | 2/6 | 0/10 |
| Exposed episode 2 | v3 / current-view | 3/6 | 0/10 |
| New episode 3 | v2 / original greedy | 3/7, one positive unscored | 0/9 |
| New episode 3 | v2 / current-view | 4/7, one positive unscored | 0/9 |
| New episode 3 | v3 / original greedy | 2/7 | 0/9 |
| New episode 3 | v3 / current-view | 4/7 | 0/9 |

The temporal change preserves 0.8 confidence, two distinct views and 500-ms spacing. It selects a qualifying chain ending at the current frame, rather than comparing only with the latest accepted sample. Every counted view also requires its own positive source-bound semantics. First views remain UNKNOWN; occlusion, ambiguity, weak masks, conflicting identity and other hard gaps reset history. No future frame supplies support retrospectively.

| Cohort | Direct guard | Clear views called clear | Unclear views called clear |
| --- | --- | ---: | ---: |
| Exposed episode 2 | v2 full frame + crop | 6/6 | 1/10 |
| Exposed episode 2 | v3 explicit boundaries + enlargement | 5/6 | 1/10 |
| New episode 3 | v2 full frame + crop | 6/7, one unscored | 1/9 |
| New episode 3 | v3 explicit boundaries + enlargement | 6/7 | 1/9 |

V3 adds four boundary-region judgments and a byte-verified nearest-neighbor enlargement of the same source crop. It adds no image detail. Both versions still call each cohort's final borderline remote/gripper view clear; v3 also withholds a clear target. More structured judgments from the same model did not solve this error. Neither result justifies claiming contact was fixed or promoting v3 into live acceptance.

The [full frozen result](../artifacts/feasibility/contact-results-20261029.json) preserves all nine method reports, per-frame predictions, localization sensitivity, references, source/mask hashes, confirmation chains and costs. These are candidate visibility/localization diagnostics, not placement PASS or physical recovery. Assistant-authored white-on-white boundary labels are annotation-sensitive and remain unchanged.

## Attempt accounting

The run admitted **50 new provider attempts: 49 completed and one uncertain network outcome**, without retries. The failed `remote_3` v2 assessment at frame 5 is unscored, not UNKNOWN. It remains in the positive denominator and may have an unknown provider charge; it is not retried or replaced. The estimated total from reported usage is **$0.431506**. The two new SAM video calls use reported frame billing, not image pricing. Reused historical calls are not charged as new inference.

The implementation passed **1,088 tests on Python 3.12**, and Python 3.9 passed with two expected optional-parser skips. The [pre-inference CI](https://github.com/yerbol1212/proofloop/actions/runs/40829820256) passed both Python jobs and the frontend build. Software checks do not establish model accuracy.

## Separate gripper geometry test

The separate trial used four SAM video calls, with independently reviewed frame-zero left/right gripper boxes. It preserves every target mask, v2/v3 judgment, reference, incomplete request and prior score. Both grippers must have valid source-bound foreground at confidence ≥0.8. A fixed Chebyshev foreground-pixel-center distance greater than three source pixels can supplement an existing clear semantic assessment; proximity or unavailable evidence yields UNKNOWN. The margin was frozen before obtaining these gripper responses.

Separation is a conservative veto, not proof of physical noncontact or complete visibility. SAM can miss fingers, merge objects or track the wrong region. Source-matched masks and reported confidence do not remove that limitation. No contrast threshold is fitted to the observed contact error.

The [public plan](../artifacts/feasibility/gripper-plan-20261029.json) and implementation were pushed in `7337ae8ef237ed8d2e21286e9cf6d6f9a70f638d` before the four calls. Freeze hash: `923b5f20f77e2805d935027b1cf013f87e71b33f62c52e9b2ef7197ab9e230ef`. All four completed, with 220 reported billed frames, no new Spark requests and **$0.044** estimated incremental cost. The [canonical full result](../artifacts/feasibility/gripper-results-20261029.json) preserves the entire earlier contact result and its failed operation.

| Cohort | v2 / current-view method | Clear checkpoints supported | Unclear checkpoints supported |
| --- | --- | ---: | ---: |
| Exposed episode 2 | Unchanged semantic baseline | 4/6 | 0/10 |
| Exposed episode 2 | Strict gripper separation | **0/6** | 0/10 |
| New episode 3 | Unchanged semantic baseline | 4/7, one positive unscored | 0/9 |
| New episode 3 | Strict gripper separation | **2/7**, one positive unscored | 0/9 |

The strict gate removes the borderline contact assessment, but its gripper evidence is unavailable because reported gripper scores fall below 0.8. It also discards too many clear views. **This is a failed retention intervention and is not adopted as a general acceptance gate.** Missing or weak gripper masks cannot verify separation. Together these trials admitted 54 new attempts, with $0.475506 estimated from reported usage; the original uncertain network operation may have an additional unreported charge.

A separately frozen [negative-only cached retest](GRIPPER-VETO-PROTOCOL-20261029.md) now evaluates whether near-mask proposals can add warnings while keeping every positive decision within the unchanged semantic baseline. It was selected after inspecting these failures and uses no new model calls. Its result must not be described as fresh or held-out validation.

The [completed warning comparison](GRIPPER-VETO-RESULTS-20261029.md) withholds both misleading contact candidates. Combined exposed retention is 3/6 versus the original 2/6, with 0/10 unclear support; additional-recording retention is 4/7 versus 3/7, with one positive unscored and 0/9 unclear support. The warning unnecessarily withholds one exposed clear view, so it does not preserve temporal-only retention of 4/6. All failed approaches remain published.
