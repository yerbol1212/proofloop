# Feasibility status

Updated: 28 October 2026, Asia/Almaty. Earlier reports retain their original experiment dates.

## Gates

| Gate | Current evidence | Status |
| --- | --- | --- |
| Meta credits | Participant dashboard displays $150 free credits | Confirmed in browser |
| API readiness | Authenticated model listing and real Spark/SAM requests succeeded | Passed |
| Deterministic checker and planner | 17/17 authored synthetic episodes, 24/24 stage assertions and 1/1 bounded planner assertion match expectations | Passed on oracle facts |
| Local tests | 228 tests cover the core, compiler, grounding, capture, recorded sessions, interactive engine and loopback HTTP boundary | Passed; system Python skips the optional parser test |
| Official SAM codec | Official parser independently re-decoded all 16 phone responses; 12 nonempty masks matched saved rasters and four covered responses were empty | Passed on phone inputs |
| Real Spark structured output | Strict smoke response and two requirement-dependent protocol drafts matched declared expectations; drafts remain unapproved | Passed initial live checks |
| Positive placement contract | Live Spark fork-left/spoon-right protocol; stable positive location evidence supports wrong-region fail and corrected pass | Passed on one recorded clip |
| Phone-video importer | Original 47.4-second phone clip; rotation handled and eight actual timestamped JPEGs extracted | Passed on phone input |
| Recorded correction session | Frozen contract, three unknown invalidation snapshots, hypothetical move/inspect plan and fresh corrected-stage evidence | Passed selected-frame observation loop |
| Available fork/spoon objects | 16 genuine SAM calls on actual cutlery frames; all four stage outcomes matched the scripted check | Passed initial phone case |
| Real SAM grounding and mask parsing | Correct and wrong-region placement plus covered-scene abstention on actual phone frames | Passed initial phone case |
| Physical correction and fresh verification | New post-correction frames pass; recorded before the planner proposal | Recorded observation supported; interactive loop pending |
| Interactive workbench | Real recorded seed, repair invalidation, paired uploads, numeric calibration, offline fixture fail/pass/unknown/request-error/disconnect states | Browser flow passed with authored fixtures; new physical correction pending |
| Synthetic scene inputs | Four generated scenes with exact prompts, hashes, annotations and provenance | Used in live Meta calls; not physical evidence |
| Public repository | [yerbol1212/proofloop](https://github.com/yerbol1212/proofloop), verified public; tested build milestones published | Created |

## First owned phone result

The [28 October phone report](../artifacts/feasibility/phone-session-20261028.md) records one 47.4-second clip, eight selected original frames, one live Spark compilation and 16 live SAM image calls. The checker returned **pass → fail → unknown → pass**. The bounded planner proposed moving the fork from RIGHT to LEFT and inspecting it afterward. Its prediction never wrote measured facts or cleared the wrong-state result; fresh post-correction observations supplied the last pass.

Technical timestamp selection, calibration and exact-contract review were performed by the assistant under the user's direct instruction to proceed. The report explicitly records that a person did not review the exact contract or selected offsets. Region names follow the handwritten paper labels: they correspond to upper/lower bands in the displayed portrait video. Calibration and the contract were frozen before SAM inference.

The estimated cost for this round is $0.045654 at the published Standard rates. SAM median latency was 1.2425 seconds and Spark took 4.987 seconds. The report preserves the one-clip, declared-unique-target and selected-frame limits. It proves neither complete visibility nor exact counts; it is not an interactive repair or held-out accuracy result. Original media, extracted images, raw responses and private authorization text remain local.

## Earlier measured results

The [interactive guide](INTERACTIVE.md) now describes a local React workbench connected to the real SAM adapter, grounding and checker. Browser tests covered upload and confirmation gating, calibration, immediate evidence invalidation and fresh result rendering. Separate injected-client fixtures exercised fail, pass, unknown, request errors and disconnection; these made zero Meta calls and establish UI/software behavior only. See the [interface validation record](INTERFACE-VALIDATION.md). The real current-scene gate requires two new operator photos after starting a repair.

The synthetic oracle diagnostic contains 17 authored episodes and 24 stage assertions. All match expected checker outputs. The bounded planner assertion passes. These are authored software diagnostics, not model accuracy or physical benchmark scores. The raw [oracle report](../artifacts/feasibility/oracle-report.md) preserves provenance and per-case outcomes.

The Spark compiler uses a strict schema and local semantic validation. Two initial real Spark compilations produced the requested object/region counts. Swapping the requested regions changed the protocol checks and content hash. Those two initial drafts use count predicates and require human approval.

The SAM-to-checker bridge accepts decoded masks with source hashes, approved region rectangles and explicit evidence origin. It permits positive visibility and count lower bounds, requires distinct sources for repeated evidence, and never automatically sets an absence-validation contract. Consequently exact-count passes remain unknown unless independently supported. This is a deliberate limitation of the current conservative observation path.

The separate authored visibility protocol passed on two distinct generated source hashes. A screwdriver in the wrong region was grounded there, while unproven absence stayed unknown. The occluded screwdriver also stayed unknown. The compiler's exact-count drafts have not passed through this observation path; the completeness contract remains an implementation gate.

The added `location` predicate checks positive placement under an explicit, operator-confirmed unique-target scene contract. Its compiler mode matched the supplied synthetic requirements in a [real Spark call](../artifacts/feasibility/live-location-20261026.json). Software tests exercise stable wrong placement → fail, missing or multiple targets → unknown, and fresh corrected placement → pass. These tests do not establish physical perception or infer scene uniqueness. The exact-count limitation remains unchanged.

The [decoder experiment](../artifacts/feasibility/capture-decoder-20261026.md) validates real extraction from generated video, including a true display-rotation matrix and off-grid frame requests. The [session diagnostic](../artifacts/feasibility/session-oracle-20261026.json) exercises the recorded observation loop with authored SAM-shaped results. Both are explicitly synthetic, and the session diagnostic makes zero model calls. The [session guide](SESSIONS.md) documents approval and inference boundaries.

The participant's available objects are a fork and spoon. The [cutlery concept probe](../artifacts/feasibility/cutlery-concepts-20261027.md) used real SAM on one generated image and returned the intended positive detections. This adds an estimated $0.005 in image calls and does not validate the participant's physical cutlery. The old tool scenes remain historical evidence; the current [capture guide](CAPTURE.md) and [starter](../examples/cutlery/README.md) use the available objects.

The [initial live report](../artifacts/feasibility/live-meta-20261026.md) records 12 real model calls plus one model listing, measured latency and an estimated Standard list-price cost of $0.03352225. It does not establish camera accuracy or a physical correction. The [capture checklist](CAPTURE.md) defines the next real-media gate.

## Publication boundaries

This file contains only public-safe access status, not account identifiers, API keys, private Discord content or unreviewed photos. Full raw model output stays in `artifacts/local/` until reviewed. Passing a mock contract test establishes adapter behavior under that mock; it does not establish model access or quality.
