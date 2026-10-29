# Gripper-contact warning and tracking retention

The experimental tracking pipeline now withholds both previously misclassified contact views. With the negative gripper warning included, clear-checkpoint retention improves from **2/6 to 3/6** on the exposed recordings and **3/7 to 4/7†* on the additional recordings. It supports **0/10 and 0/9 unclear checkpoints**, respectively. The warning loses one clear view compared with temporal confirmation alone; that tradeoff is retained, not relabeled away.

## Fixed real-video comparison

Four licensed ALOHA recordings supply 32 unchanged checkpoints. Episode 2 reuses the earlier visibility sources. Episode 3 adds two real online recordings, with seven clear and nine unclear references frozen before the new contact-model trial. They share exposed source families, objects and cameras. The subsequent warning rule is selected after inspecting all four recordings and the failed strict gate: **its zero-call comparison is a cached development retest, not held-out validation**.

| v2 candidate-tracking method | Exposed episode 2: clear retained | Unclear supported | Additional episode 3: clear retained | Unclear supported |
| --- | ---: | ---: | ---: | ---: |
| Original greedy confirmation | 2/6 | 0/10 | 3/7† | 0/9 |
| Current-view confirmation | 4/6 | 0/10 | 4/7† | 0/9 |
| Current-view + strict gripper separation | **0/6** | 0/10 | **2/7**† | 0/9 |
| Current-view + negative gripper warning | **3/6** | **0/10** | **4/7†** | **0/9** |

† One episode-3 positive assessment has an uncertain network outcome. It remains **unscored and in the seven-positive denominator**, with no retry or replacement. First views remain UNKNOWN because two distinct qualifying views are required. The warning adds one further clear abstention in episode 2.

Candidate decisions before temporal confirmation expose the original component error:

| Cohort | Candidate method | Clear retained | Unclear supported |
| --- | --- | ---: | ---: |
| Exposed episode 2 | Original v2 assessment | 6/6 | 1/10 |
| Exposed episode 2 | v2 + negative warning | 5/6 | **0/10** |
| Additional episode 3 | Original v2 assessment | 6/7, one unscored | 1/9 |
| Additional episode 3 | v2 + negative warning | 6/7, one unscored | **0/9** |

These are visibility/localization candidate decisions, not placement PASS or executed recovery. The original temporal policy already withheld the two contact views; the new warning corrects their misleading candidate-level support. All historical methods, the unsuccessful v3 boundary/enlargement rubric and strict-gripper losses remain in the canonical results.

## What changed

`confirmation_policy` finds a qualifying chain **ending at the current frame**. A dense intermediate sample cannot displace an earlier sufficiently spaced view. Every counted view requires its own source-bound positive semantics, a supported mask, confidence ≥0.8, complete assessed visibility and unambiguous identity. Two distinct JPEGs and 500-ms spacing remain required. Hard gaps and track changes reset history; future frames cannot retroactively qualify earlier views.

`gripper_veto` uses exact source-frame/PTS-bound target and reviewed gripper foreground proposals. The original fixed three-source-pixel Chebyshev margin is unchanged. A valid near proposal can only add a warning, including when its reported gripper score is weak. A far, missing or unavailable proposal cannot qualify a target or verify absence, separation or identity. The target's original acceptance gates remain required.

The experiment recomputes policies after each warning resets history, then intersects actual supported rows with the corresponding unchanged policy's exact source/frame/box/evidence decisions. This prevents a rejection from accidentally creating support by changing greedy sampling times. Published current-view chains must also pass complete-interval checks for intervening risks, hard gates and identifier changes. Original unscored outcomes cannot become scored decisions.

Both contact warnings occur at `remote_new:59` and `remote_3:59`. Minimum target/gripper distances are respectively 0 and 1 source pixels. Private overlays show plausible target/gripper masks, without an obvious swap to the tape. Pixel-level masks and low contrast do not establish physical contact or occlusion.

The unnecessary warning is `screwdriver_new:5`: the right-gripper proposal lies three pixels from the target and has reported score 0.544488. Its left proposal is farther away. The frozen margin is not tightened after observing this false warning. This exposed example demonstrates the remaining specificity problem; score values are not calibrated physical-contact probabilities.

## Receipts, checks and accounting

The [warning protocol](GRIPPER-VETO-PROTOCOL-20261029.md), code and [public plan](../artifacts/feasibility/gripper-veto-plan-20261029.json) were pushed in `9d83e9ce61b10916e22b9cba4614d4479376b4b9` before derivation. Freeze hash: `802b386d38be9a7c8cb19133ba5d01b2d6050420e0d8a801186dd9ce400cf9cb`. The receipt explicitly records post-inspection selection and **zero new provider calls**.

The [full warning result](../artifacts/feasibility/gripper-veto-results-20261029.json) preserves all 17 method summaries per cohort, per-frame warning/mask hashes, full policy rows and confirmation chains, localization sensitivity and historical failure/cost projections. It references the unchanged [strict gripper result](../artifacts/feasibility/gripper-results-20261029.json), which preserves the [original contact result](../artifacts/feasibility/contact-results-20261029.json). The [32-row checkpoint CSV](../artifacts/feasibility/gripper-veto-checkpoints-20261029.csv) makes individual regressions easy to inspect. Raw media, arbitrary model text and encoded masks remain private. Source licenses, pinned original hashes and lossless frame derivations are in [source credits](ONLINE-VIDEO-SOURCES.md) and the [contact protocol](CONTACT-PROTOCOL-20261029.md).

The earlier trials admitted **54 new provider attempts: 53 completed and one uncertain network outcome**, with **$0.475506 estimated from reported usage**. The uncertain request may have an additional unreported charge. The cached warning retest adds **zero requests and $0 incremental inference**. Reused historical model costs are not charged again.

An independent offline audit exactly reproduced the canonical result with API entry points patched to reject any inference attempt. All 611 snapshotted input/canonical files remained unchanged. It independently checked 34 main method scores, 24 localization-sensitivity scores, four candidate diagnostics and their partitions. Across sixteen full policy reports, all 22 supported rows were exact baseline subsets; all thirteen current-view supports had distinct, spaced, source-bound, risk-free chains. Original unscored judgments and accounting were preserved.

**1,167 Python 3.12 tests pass**, and Python 3.9 passes with two expected optional-parser skips. [Frozen implementation CI](https://github.com/yerbol1212/proofloop/actions/runs/40840048700) passes both Python jobs and the strict production frontend build. Authored tests cover missing baselines, intervening risk, track changes, nullable unassessed continuity, low-score negative warnings and the greedy timing counterexample. Software checks do not establish perception accuracy.

The result is a specific error fix with measured retention improvement and a visible false-warning cost. It does not establish reliability on unseen scenes, long/full occlusions, identity switches, continuous physical observation or action following a ProofLoop instruction. Live placement acceptance remains unchanged. New recordings frozen before further rule selection are needed to measure transfer.
