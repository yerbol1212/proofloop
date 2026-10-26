# Feasibility status

Date: 26 October 2026, Asia/Almaty.

## Gates

| Gate | Current evidence | Status |
| --- | --- | --- |
| Meta credits | Participant dashboard displays $150 free credits | Confirmed in browser |
| API readiness | Dashboard says no payment method; API key creation is disabled and no key is listed | Blocked on account setup |
| Deterministic checker and planner | 17/17 authored synthetic episodes, 24/24 stage assertions and 1/1 bounded planner assertion match expectations | Passed on oracle facts |
| Local tests | 83 tests pass in Python 3.12 including the official mask decoder, compiler contracts and grounding bridge | Passed |
| Official SAM codec | Installed `meta-sam-parser==0.1.0`; documented encoded test mask decoded successfully | Passed locally, not live perception |
| Real Spark structured output | No API inference performed yet | Pending usable key/access |
| Real SAM grounding and mask parsing | No API inference performed yet | Pending usable key/access and suitable media |
| Physical correction and fresh verification | No owned camera recordings supplied yet | Pending media and live perception |
| Synthetic scene inputs | Four generated scenes with exact prompts, hashes, annotations and provenance | Ready for real API tests; not physical evidence |
| Public repository | [yerbol1212/proofloop](https://github.com/yerbol1212/proofloop), verified public; initial milestone pushed; initial GitHub Actions checks passed | Created |

## Initial measured results

The synthetic oracle diagnostic contains 17 authored episodes and 24 stage assertions. All match expected checker outputs. The bounded planner assertion passes. These are authored software diagnostics, not model accuracy or physical benchmark scores. The raw [oracle report](../artifacts/feasibility/oracle-report.md) preserves provenance and per-case outcomes.

The Spark compiler is implemented with a strict schema and local semantic validation. It requires human approval and supports count predicates only at this stage. The official SAM parser is exercised with a published encoded fixture. Neither test used a live model.

The SAM-to-checker bridge accepts decoded masks with source hashes, approved region rectangles and explicit evidence origin. It permits positive visibility and count lower bounds, requires distinct sources for repeated evidence, and never automatically sets an absence-validation contract. Consequently exact-count passes remain unknown unless independently supported. This is a deliberate limitation of the current conservative observation path.

No live Meta request, API charge or physical-validation result is claimed by offline tests. The capture checklist defines the smallest real-media check.

## Publication boundaries

This file contains only public-safe access status, not account identifiers, API keys, private Discord content or unreviewed photos. Full raw model output stays in `artifacts/local/` until reviewed. Passing a mock contract test establishes adapter behavior under that mock; it does not establish model access or quality.
