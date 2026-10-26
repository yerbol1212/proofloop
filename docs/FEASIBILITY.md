# Feasibility status

Date: 26 October 2026, Asia/Almaty.

## Gates

| Gate | Current evidence | Status |
| --- | --- | --- |
| Meta credits | Participant dashboard displays $150 free credits | Confirmed in browser |
| API readiness | Authenticated model listing and real Spark/SAM requests succeeded | Passed |
| Deterministic checker and planner | 17/17 authored synthetic episodes, 24/24 stage assertions and 1/1 bounded planner assertion match expectations | Passed on oracle facts |
| Local tests | 104 tests in Python 3.12 cover the official decoder, compiler, grounding, request accounting and scene harness | Passed |
| Official SAM codec | `meta-sam-parser==0.1.0` decoded actual live image masks; box-local raster coordinates mapped into image coordinates | Passed on generated inputs |
| Real Spark structured output | Strict smoke response and two requirement-dependent protocol drafts matched declared expectations; drafts remain unapproved | Passed initial live checks |
| Real SAM grounding and mask parsing | Eight scene calls; intended positive regions, wrong-region evidence and conservative occlusion behavior | Passed on generated inputs |
| Physical correction and fresh verification | No owned camera recordings supplied yet | Pending media and live perception |
| Synthetic scene inputs | Four generated scenes with exact prompts, hashes, annotations and provenance | Used in live Meta calls; not physical evidence |
| Public repository | [yerbol1212/proofloop](https://github.com/yerbol1212/proofloop), verified public; tested build milestones published | Created |

## Initial measured results

The synthetic oracle diagnostic contains 17 authored episodes and 24 stage assertions. All match expected checker outputs. The bounded planner assertion passes. These are authored software diagnostics, not model accuracy or physical benchmark scores. The raw [oracle report](../artifacts/feasibility/oracle-report.md) preserves provenance and per-case outcomes.

The Spark compiler uses a strict schema and local semantic validation. Two real Spark compilations produced the requested object/region counts. Swapping the requested regions changed the protocol checks and content hash. Both drafts require human approval and support count predicates only at this stage.

The SAM-to-checker bridge accepts decoded masks with source hashes, approved region rectangles and explicit evidence origin. It permits positive visibility and count lower bounds, requires distinct sources for repeated evidence, and never automatically sets an absence-validation contract. Consequently exact-count passes remain unknown unless independently supported. This is a deliberate limitation of the current conservative observation path.

The separate authored visibility protocol passed on two distinct generated source hashes. A screwdriver in the wrong region was grounded there, while unproven absence stayed unknown. The occluded screwdriver also stayed unknown. The compiler's exact-count drafts have not passed through this observation path; the completeness contract remains an implementation gate.

The [live report](../artifacts/feasibility/live-meta-20261026.md) records 12 real model calls plus one model listing, measured latency and an estimated Standard list-price cost of $0.03352225. It does not establish camera accuracy or a physical correction. The [capture checklist](CAPTURE.md) defines the next real-media gate.

## Publication boundaries

This file contains only public-safe access status, not account identifiers, API keys, private Discord content or unreviewed photos. Full raw model output stays in `artifacts/local/` until reviewed. Passing a mock contract test establishes adapter behavior under that mock; it does not establish model access or quality.
