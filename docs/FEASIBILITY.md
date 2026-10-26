# Feasibility status

Updated: 27 October 2026, Asia/Almaty. The recorded live API and decoder experiments are dated 26 October in UTC.

## Gates

| Gate | Current evidence | Status |
| --- | --- | --- |
| Meta credits | Participant dashboard displays $150 free credits | Confirmed in browser |
| API readiness | Authenticated model listing and real Spark/SAM requests succeeded | Passed |
| Deterministic checker and planner | 17/17 authored synthetic episodes, 24/24 stage assertions and 1/1 bounded planner assertion match expectations | Passed on oracle facts |
| Local tests | 167 tests cover the core, compiler modes, grounding, accounting, capture and recorded-session contracts | Passed |
| Official SAM codec | `meta-sam-parser==0.1.0` decoded actual live image masks; box-local raster coordinates mapped into image coordinates | Passed on generated inputs |
| Real Spark structured output | Strict smoke response and two requirement-dependent protocol drafts matched declared expectations; drafts remain unapproved | Passed initial live checks |
| Positive placement contract | New explicit unique-target location mode compiled correctly with live Spark; wrong-region checks tested on authored observations | Passed software and draft gates; camera pending |
| Phone-video importer | Actual FFmpeg/FFprobe: 4 synthetic cases, 26 extracted JPEGs; rotation, returned PTS and HD resizing checked | Passed decoder gate |
| Recorded correction session | Approved contract binding, fresh stage-isolated observations, hypothetical move/inspect plan and failure handling tested with authored SAM-shaped results | Passed software integration; camera pending |
| Available fork/spoon objects | Two real SAM calls found the intended classes/sides on one generated image; current capture preset uses fork LEFT and spoon RIGHT | Passed initial concept probe; actual cutlery pending |
| Real SAM grounding and mask parsing | Eight scene calls; intended positive regions, wrong-region evidence and conservative occlusion behavior | Passed on generated inputs |
| Physical correction and fresh verification | No owned camera recordings supplied yet | Pending media and live perception |
| Synthetic scene inputs | Four generated scenes with exact prompts, hashes, annotations and provenance | Used in live Meta calls; not physical evidence |
| Public repository | [yerbol1212/proofloop](https://github.com/yerbol1212/proofloop), verified public; tested build milestones published | Created |

## Initial measured results

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
