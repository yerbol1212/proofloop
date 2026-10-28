# Contributions and provenance

Project development began on 26 October 2026 after the hackathon kickoff. The repository contains newly created ProofLoop code and tests. It does not contain a pre-existing proprietary application.

## Event contributions

- Declarative protocol validation and evidence-linked pass/fail/unknown checks.
- Observation freshness, invalidation and completion-attempt boundaries.
- Bounded repair planning with explicit preconditions, effects and action costs.
- Synthetic oracle diagnostic fixtures and transparent result reporting.
- Meta Responses API adapter and credential-safe live access tests.
- Spark protocol draft compiler with local validation, explicit reviewed activation and declared reviewer provenance.
- Conservative SAM-mask-to-checker grounding with explicit evidence provenance.
- Four generated scene inputs with prompt and source-hash manifests, labeled synthetic.
- Bounded live SAM scene harness with source-validated resume, private response caches and reviewed public results.
- Live-model feasibility checks, correct mapping of box-local masks and accounting for reported usage on unsuccessful responses.
- Local password form that saves an ignored credential file without putting the key in chat.
- Positive placement predicates, explicit unique-target scene contracts and separate Spark compiler modes for counts, visibility and location.
- Phone-video preflight and timestamp-preserving JPEG extraction, including display rotation and source/hash binding.
- Reviewable recorded-session contracts that bind calibration, model concepts, selected frames and unique-target assumptions.
- Authored fork/spoon capture preset with unapproved placement requirements and separate camera calibration.
- Stage-isolated correction traces with movement/occlusion invalidation, a bounded hypothetical repair and fresh post-correction evidence.
- First owned phone experiment with actual Spark/SAM calls, explicit assistant-review provenance and a separately reviewed public results summary.
- Local React workbench with recorded-evidence overlays, per-object placement results, repair proposals, explicit camera activation, paired photo upload, region calibration and a fresh-verification workflow.
- Revision-bound one-use capture tokens, explicit operator scene/timing confirmations, prior-image rejection, bounded per-target inspection budgets and stale-result handling.
- Loopback HTTP adapter with private media registration, same-origin mutation checks and offline transport/interactive-engine tests.
- Editable browser task contracts binding requirements, object inventory, concept descriptions and perception mode; draft/active separation and evidence invalidation on activation.
- Reason-specific UNKNOWN inspection requests that prioritize better evidence before physical repair and do not invent absence or failure causes.
- Optional task-independent Spark localization followed by real box-prompted SAM refinement, preserving explicit Spark-assigned identity limits and missing-target abstention.
- Source-bound recorded-video evaluation with frozen authored labels, same-image Spark baseline, explicit request budgets, per-condition costs and safe public exports.
- Read-only Video lab with licensed source attribution, original image pairs, calibrated zones and preserved successful/failed/unknown results.
- Six real external coffee recordings tested as 12 scoped placement pairs; original concept-only SAM failures and guided-method limits retained in public results.

This list describes implementation scope. The current software suite has 321 passing Python 3.12 tests; GitHub CI passed Python 3.9/3.12 at b6a5ebe. Live and physical validation status is recorded separately in [feasibility status](docs/FEASIBILITY.md); passing tests do not establish model accuracy or physical closed-loop recovery.

## External components

Python and its standard library are external infrastructure. Muse Spark and SAM 3.1 are pretrained Meta models, not models trained or created by this project. The optional official `meta-sam-parser==0.1.0` dependency is distributed under the SAM License; it is installed separately and not vendored in this repository. See [Meta's parser repository](https://github.com/meta-models/meta-sam). Synthetic fixtures are authored test data, not camera recordings or industrial measurements.

External ALOHA coffee recordings were created by the ALOHA authors and distributed through LeRobot; they are not event-created ProofLoop assets. Dataset cards declare MIT, with exact source/license attribution retained in [online source credits](docs/ONLINE-VIDEO-SOURCES.md). Original video, extracted frames, locator outputs and masks stay local. ProofLoop's placement labels are newly authored for selected states, not official ALOHA labels or natural robot-error annotations. The [online results](artifacts/feasibility/online-video-20261028.md) report the original SAM failure and that guided SAM did not outperform Spark alone.

Codex assisted with code, tests and research. AI assistance does not establish original model capability or replace verification. All human contributors must be added to the hackathon submission under the event rules.

FFmpeg and FFprobe are external command-line media tools, not project-authored decoders. Local decoder integration used version 9.0 binaries from [OSXExperts](https://www.osxexperts.net/), verified against the publisher's binary SHA-256 values. The binaries and generated test videos remain in ignored local storage and are not redistributed here. The extractor also supports tools already available on PATH.

React, Vite, TypeScript and the self-hosted Inter font package are external frontend components installed through npm. Image-generated UI concepts guided the interface before implementation; those private design references are not runtime evidence or app backgrounds. Dependency versions are pinned in the lockfile.
