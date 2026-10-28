# Specific missing evidence, with a stopping rule

UNKNOWN is a request for support, not a physical repair or an assertion that a target is absent. ProofLoop diagnoses the unconfirmed checks from structured observations, source-bound SAM samples, locator fields and reviewed geometry. It does not select instructions by searching model explanations.

For example, a confirmed fork and an unconfirmed spoon produces a request for the **entire spoon in views 1 and 2**, alongside its required region and the unsupported detection. Both new images must still include all required targets: replacing evidence creates a new task revision and does not carry the earlier fork PASS into it.

## Supported gaps

The policy distinguishes missing current evidence, expired evidence, unsupported detections, multiple candidates, ambiguous identity, unsupported full visibility, low scores, uncertain region boundaries, invalid geometry, unavailable masks and inconsistent or insufficiently distinct samples. Gaps identify the affected object, check and source-view indices where support is available. A named region is a reviewed image rectangle, not a recovered 3D workspace.

Instructions keep the confidence threshold and task requirements fixed. A low score requests a focused, well-lit view; a boundary gap requests visible target edges and calibration review; conflicting observations request a consistent new pair. A missing detection never proves absence. More images cannot establish unsupported exact counts, complete event history, or a stability requirement beyond the supported two-view method: those contracts stop for review.

## Finite submissions

An unresolved inspection permits **at most two follow-up pairs** within the active task/session. The initial pair is separate. A follow-up is charged to this budget immediately before model dispatch; an API error still consumes the allocated attempt. Invalid requests and rejected old bytes make no model call and do not spend an attempt. Repeated **New inspection** or **Capture requested views** clicks cannot reset the counter.

Each follow-up requires two distinct, entirely new source hashes, the reviewed regions, explicit target uniqueness and the existing timing confirmations. The engine blocks additional captures when the budget is exhausted. UNKNOWN cannot authorize a move, including when another check already supports a known misplaced target. A fully known PASS or FAIL ends that acquisition chain. Activating a newly reviewed task starts a new budget; this is a scoped interaction limit, not a global anti-sampling guarantee across reviewed tasks or restarted servers.

Uploaded timing and camera times retain their existing declared provenance. A modified old image can defeat a byte-reuse guard. There is no independent camera timestamp, ownership attestation or proof that a requested action occurred.

## Recorded timelines

Recorded follow-ups must use strictly later source timestamps and no earlier source bytes. When an approved candidate schedule is supplied, the next chronological qualifying pair is fixed before inference; the policy cannot skip to a nicer result. It stops at a known result, its two-follow-up limit, an error, or exhaustion of that schedule. Opening Video Lab makes no model calls and sends no command to a camera or robot.

The [actual fixed-schedule test](../artifacts/feasibility/fixed-followup-20261028.md) reuses three licensed coffee recordings. Guided Spark→SAM resolved one timeline after two follow-ups; another remained UNKNOWN when its recording ended. Spark resolved the viable timelines on the first follow-up. The separately scored final future pair unused by Spark is excluded from that policy's consumption totals.

The six scored pairs matched **4/6** labels with guided Spark→SAM and **6/6** with Spark alone. They include three UNKNOWN and three PASS labels, with no FAIL control, so they cannot estimate wrong-placement false acceptance. These are development observations and retrospective policy replay on a frozen schedule, not physical interactive recovery or evidence of a SAM accuracy advantage.

## Inspect the implementation

- `proofloop/acquisition.py`: pure typed policy and source-pair checks.
- `proofloop/interactive.py`: revision, attempt budget and model-dispatch enforcement.
- `proofloop/followup.py`: replay of the fixed source schedule from saved actual responses.
- `web/src/AcquisitionPanel.tsx`: object/view gaps and explicit stopping state.

Policy and lifecycle tests cover two allowed follow-ups, error consumption, rejected reuse, repeated capture requests, mixed FAIL/UNKNOWN, terminal timelines and newly reviewed task changes. Controlled interface fixtures exercise the disabled exhausted-state controls; they are not live model measurements.
