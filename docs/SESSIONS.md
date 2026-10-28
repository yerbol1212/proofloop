# Recorded phone correction sessions

The session runner connects an approved location protocol, calibrated regions and selected original video frames to SAM, the deterministic checker and the bounded planner. It evaluates a recorded procedure; it does not control a robot, recognize every action or establish that a person followed its later proposal.

Start with the [phone recording checklist](CAPTURE.md). One continuous clip, at most 120 seconds, contains four stable checkpoints. The current session supports two distinct target object classes and two regions. Each checkpoint uses exactly two original frames at least one second apart. Stage labels and ownership are human declarations, and the original video remains local.

## Evidence flow

1. Preflight and extract the clip with `proofloop.capture`. Inspect the displayed JPEGs and adjust selection if hands, movement or a shifted camera invalidate a checkpoint.
2. Draft location requirements with Spark. Confirm that there is one visually distinct target instance of each supplied class. This explicit scene contract supports placement checks; it is not a count inferred by the model.
3. Select normalized LEFT/RIGHT rectangles on the displayed, autorotated images. Supply the two object-to-SAM concept mappings. Region IDs follow the task's written labels; a rotated recording can place those named regions above/below each other in image coordinates. The generated-scene calibration is not reused automatically.
4. Review the complete session contract before approval. Its hash binds the protocol, rectangles, concepts, uniqueness flags, clip hash, stage windows, frame hashes and actual source timestamps. Any edit requires a new approval.
5. Run at most 16 new SAM image calls, with no retries. Invalid approvals, changed files and invalid timelines stop before inference. A malformed returned source/model/decoder binding stops further calls immediately. The original source and frames are checked again before the report can support a completed loop.

The [cutlery preset](../examples/cutlery/README.md) supplies an authored, unapproved fork-left/spoon-right starter and concrete SAM concepts. It does not supply calibrated camera rectangles or establish that your real cutlery is detectable. The `unique_target: true` flags are proposed scene assumptions that the operator must confirm during session approval.

The initial correct stage can pass from current positive evidence. At the end of that hold, the declared movement begins and old facts are invalidated. The wrong-placement stage can fail from stable positive location evidence. The planner proposes a move and a subsequent inspection under a small declared action model. Its prediction never becomes a measured fact.

Covering invalidates evidence again; an unavailable object stays unknown. The correction starts a new attempt and invalidates the old state. Only fresh post-correction frame evidence can support the final pass. The original wrong or unknown result remains in the trace.

## Local commands

After capture extraction, create the private file `artifacts/local/regions.json`, mapping `left` and `right` to approved normalized `[x1,y1,x2,y2]` rectangles. They must be nonoverlapping and match the actual camera view. Review `examples/cutlery/concepts.json`, which maps `fork` and `spoon` to those concrete SAM noun phrases; use a separate private concepts file if the reviewed terms need changing.

Draft a protocol without claiming it is already approved:

```bash
.venv/bin/python -m proofloop.compiler \
  --requirements 'The unique fork must be located in left and the unique spoon in right.' \
  --objects fork spoon --regions left right \
  --predicate-kind location --unique-objects fork spoon
```

Use the printed Spark draft path in place of `COMPILED_DRAFT`. The authored `examples/cutlery/protocol.json` can also be used as an explicit unapproved starter, but it must not be described as a Spark-generated result:

```bash
.venv/bin/python -m proofloop.session draft \
  --capture inputs/local/phone/run-01/manifest.json \
  --protocol COMPILED_DRAFT \
  --regions artifacts/local/regions.json \
  --concepts examples/cutlery/concepts.json \
  --output artifacts/local/session-draft.json
```

Inspect the draft, including source frames, requirements, region boundaries and unique-target assumption. After the local operator reviews those exact inputs, use its printed hash in place of `REVIEWED_HASH`:

```bash
.venv/bin/python -m proofloop.session approve \
  --draft artifacts/local/session-draft.json --hash REVIEWED_HASH \
  --confirm-unique-targets --output artifacts/local/session-approved.json

.venv/bin/python -m proofloop.session run \
  --approval artifacts/local/session-approved.json \
  --output artifacts/local/phone-session-01
```

These commands never silently approve a model draft. Local approval records an operator assertion; it is not cryptographic proof of ownership or a detector of forged media. Use new artifact paths for another run. The runner never overwrites evidence or silently reuses prior responses.

If the user explicitly delegates the next processing steps, technical review can instead be recorded with `--review-origin assistant_under_user_authorization` and `--authorization-reference` containing the direct user instruction. That mode preserves the exact contract hash and records `human_reviewed_exact_contract: false`. Assistant-selected timestamps similarly use `ground_truth_origin: user_task_with_assistant_reviewed_timestamps` with explicit `annotation_review` authority and `human_reviewed_offsets: false`. These fields identify who performed the review; they never turn an assistant review into a human signoff.

The capture manifest is immutable evidence of the extraction step. Its `extracted_pending_visual_review` status records that earlier moment; the separate, hash-bound session approval records the subsequent review without modifying the manifest.

## Interpreting results

The report contains `stage_statuses`, `trace`, `gates`, reported usage, failures and provenance. `recorded_observation_loop_supported` means the selected recorded observations support the scripted pass → fail → unknown → fresh pass pattern under the approved contract. It does not mean an action was automatically recognized or executed. `physical_action_execution_verified` remains false.

The planner's optimality is limited to its declared finite model: moving one object costs 1 and a subsequent inspection costs 0.25. Physical obstacles, collisions and robot motion are outside that model. Unknown locations request inspection before a move is planned.

The [authored session diagnostic](../artifacts/feasibility/session-oracle-20261026.json) exercises this flow with synthetic SAM-shaped results and zero model requests. The [decoder experiment](../artifacts/feasibility/capture-decoder-20261026.md) uses actual FFmpeg on generated color patterns. The separate [first phone experiment](../artifacts/feasibility/phone-session-20261028.md) supports the expected observation loop on one real recorded clip with actual Meta calls. Interactive correction and broader evaluation remain separate gates.

Raw clips and extracted images stay in ignored `inputs/local/phone/`; approvals, model responses and traces stay in ignored `artifacts/local/`. Only separately reviewed public-safe summaries should be published.
