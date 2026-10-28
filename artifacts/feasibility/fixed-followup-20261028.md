# Fixed future-evidence evaluation — 2026-10-28

The bounded UNKNOWN policy was replayed against real Meta responses on a preselected recording timeline. **Guided Spark→SAM resolved one of two trajectories with future evidence; Spark alone resolved both.** A third, deliberately exhausted timeline remained UNKNOWN for both. The guided method did not outperform Spark and required two follow-ups to resolve its one successful trajectory.

[Machine-readable results and replay](online-fixed-followup-20261028.json) include source/frame hashes, the fixed timestamp schedule, typed evidence requests, individual verdicts and separated request estimates.

## Schedule frozen before inference

Three already exposed [MIT-licensed ALOHA coffee recordings](https://huggingface.co/datasets/lerobot/aloha_static_coffee/blob/1c52c4c794b6f93d931489263cf15e4331600bfc/README.md) supply six disjoint pairs. Every selected frame was visually reviewed before this run. The sole glass has the same authored dispense-region requirement, calibration, 0.8 confidence threshold and two-consistent-sample rule as prior work. Guided version remains `normalized_boxes_v2`; its module hash was frozen and rechecked. No thresholds, source frames or hyperparameters changed after results.

| Trajectory | Initial pair → expected | Fixed first follow-up → expected | Fixed second follow-up → expected |
| --- | --- | --- | --- |
| Episode 000000 | 16.5 / 18.5 s → UNKNOWN | 19.1 / 20.1 s → PASS | 20.7 / 21.7 s → PASS |
| Episode 000001 | 18.5 / 20.0 s → UNKNOWN | 20.5 / 21.5 s → PASS | None fits after this pair |
| Episode 000002, terminal | 0.5 / 21.5 s → UNKNOWN | None fits after this pair | None |

Each UNKNOWN label concerns an inconsistent pair: staging in one frame and dispense in the other. A later frame's visible placement does not satisfy the two-consistent-observation requirement by itself. Later PASS labels certify only scoped image-zone membership. Robot grippers overlap the glass in some views; annotations do not certify full amodal object visibility, safe dispensing or the whole procedure.

Contract hash: `e0b0db5b98adc7b9e7b2f15e85ad3d330e8dd393fff3f2030509a550cc8486ca`.

The schedule permits **at most two follow-up submissions**, requires both new images to be strictly later than the previously consumed pair, requires at least one source second between them, and rejects any previously used image hash. A known result, including FAIL, stops the trajectory. Source exhaustion or inference error also stops it. It never scans for a favorable frame or skips an inconvenient preselected pair.

All six frozen pairs were model-tested once for comparison. The policy was then replayed offline in order using those saved responses. Unused counterfactual inference is identified below; it is not counted as policy-required work. This is a fixed recording timeline analysis, not a live camera responding to a request or a robot executing a recommended action.

## Actual trajectory outcomes and requests

| Trajectory | Guided Spark→SAM | Spark alone |
| --- | --- | --- |
| Episode 000000 | UNKNOWN → UNKNOWN → PASS; two follow-ups | UNKNOWN → PASS; one follow-up, then stop |
| Episode 000001 | UNKNOWN → UNKNOWN; recording exhausted | UNKNOWN → PASS; one follow-up |
| Terminal episode 000002 | UNKNOWN; no admissible future pair | UNKNOWN; no admissible future pair |

The new typed request builder read saved bound SAM evidence and structured locator fields; it did not classify errors by parsing explanation text:

- Episode 000000 initially lacked supported full visibility in both views. The next fixed pair had ambiguous identity in both views. The request changed from showing the entire glass and boundaries to capturing distinguishing glass features. Only the second permitted follow-up supported PASS.
- Episode 000001 initially lacked full target visibility in one view and sufficient consistent samples. Its permitted future pair still lacked supported full visibility. The policy stopped because no later pair with the required spacing remained.
- The terminal episode's latest SAM score was below the unchanged threshold. Its request called for focused views, but the recording had no time for them, so the terminal state remained UNKNOWN.

The follow-up source validator accepted every consumed pair and checked exact chronological order, required spacing and disjoint hashes. A resolved result verifies the later observed placement only; it cannot establish that ProofLoop caused a physical correction. The prerecorded source also cannot satisfy a request for a different camera angle or improved lighting.

## Pair-level results and compute

| Condition | Matches across all six fixed pairs | Correct UNKNOWN labels | Correct PASS labels | Actual requests for all pairs | Estimated USD for all pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Guided Spark→SAM + checker | 4/6 | 3/3 | 1/3 | 11 | 0.058507 |
| Spark alone | 6/6 | 3/3 | 3/3 | 6 | 0.061083 |

These six cases include **three UNKNOWN and three PASS labels and no FAIL labels**. Neither condition incorrectly passed the selected UNKNOWN cases. Without wrong-placement FAIL controls, this targeted suite cannot estimate false acceptance of an incorrect placement. Always returning UNKNOWN would match three labels and accept no correct placement.

Policy-consumed work was **11 guided requests, estimated $0.058507**, and **five Spark requests, estimated $0.051561**. The already-resolved Spark episode 000000 trajectory did not need its second follow-up: that one counterfactual call cost an estimated **$0.009523** and is excluded from Spark's policy cost. Estimates are separately rounded and may differ by $0.000001 when summed. The actual run made **17 requests under a 24-request cap**, totaling an estimated **$0.119590** for all paid inference. Automatic retries were disabled. Usage estimates do not guarantee provider billing.

The integrity validator recomputed metrics, checked twelve condition/case decisions, replayed six grounded decisions and both inference sidecars. This checks internal source/report consistency, not independent physical truth or authenticated API origin.

## Limits and reproduction

All three recordings were exposed during earlier development. This adds **zero independent source recordings** and is not pooled with prior holdback scores. The glass task and fixed camera are narrow. Source selection, calibration, target uniqueness, label review and image-to-timestamp mapping remain caller-reviewed assertions; hashes do not provide trusted timestamps. The original robot never saw ProofLoop's recommendation.

Credit Tony Z. Zhao, Vikash Kumar, Sergey Levine and Chelsea Finn for ALOHA (RSS 2023), and Hugging Face / LeRobot for conversion. Original media and raw responses remain private. The [public reproduction guide](../../docs/REPRODUCING.md) provides exact pinned source hashes, byte caps, timestamps and separate optional Meta commands. Preparation works without a Meta key or model calls and never automatically approves a task contract.
