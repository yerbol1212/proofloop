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
- Guided v2 normalized-coordinate conversion, separate identity/visibility/boundary uncertainty, unchanged confidence gates and method-version-bound task review.
- Explicit development-reprocessing versus new-source evaluation roles and versioned Video lab comparisons; previous failed runs remain unchanged.
- Source-bound recorded-video evaluation with frozen authored labels, same-image Spark baseline, explicit request budgets, per-condition costs and safe public exports.
- Read-only Video lab with licensed source attribution, original image pairs, calibrated zones and preserved successful/failed/unknown results.
- Eight original external coffee recordings and 16 unique scoped pairs in the earlier placement cohorts; guided v2's four development replay and four fresh-source placement cases remain separately reported.
- Strict report auditing with recomputed metrics, exact case/check/source binding, guided-version validation, grounded decision/sidecar replay and 49 focused benchmark/lab tests; four historical public summaries remain unchanged.
- Linked recovery receipts with offline parent, planner, contract, media and mask replay, plus Evidence log presentation and a read-only recovery API; no proof of executed physical motion.
- A separately frozen eight-case temporal development stress cohort on four previously tested sources: guided 7/8 versus Spark 8/8; all four UNKNOWN labels unverified by both, one extra guided abstention on a FAIL control, and no incorrect passes in this selected sample.

- Typed UNKNOWN acquisition from validated structured locator fields, bound SAM measurements and checker observations, with per-object/per-view evidence gaps instead of explanation-text classification.
- Server-enforced two-follow-up budgets across unresolved inspections, pre-dispatch attempt accounting, reusable-image rejection, explicit terminal states and no threshold relaxation or search-until-PASS behavior.
- Portable one-to-six-target contracts with two reviewed named regions and an authored, unapproved one-glass coffee preset; changed requirements or regions invalidate previous evidence.
- An authored placement preparation checkpoint requiring an active reviewed contract, fresh PASS and offline evidence replay before a separate operator-declared continuation record. The declaration does not report observed physical action or completion of a whole procedure.
- A fixed future-evidence development evaluation with six preselected pairs from three already exposed coffee recordings: guided 4/6 versus Spark 6/6; chronological policy replay separates needed observations from unused counterfactual inference.
- A different-task candy recording with unchanged guided-v2 inference: UNKNOWN → PASS, guided 2/2 and Spark 2/2. Early packet/label-area anchors were rejected by the unchanged SAM confidence gate, rather than independently identified as distractors.
- A dependency-free reproduction CLI with explicit bounded downloads, pinned source sizes/hashes, actual decoded timestamps, retained attribution, unapproved preparation and separate optional Meta inference. A fresh source checkout downloaded one clip and extracted four frames without API calls.
- Deterministic requirement/observation/next-step diagnostics with typed UNKNOWN gaps, historical/current scope and preserved stopping rules; these do not infer physical causes or authorize movement.
- Failure-focused evaluation from validated reports, with coverage beside selective accuracy, separate known-FAIL and UNKNOWN-label acceptance, exact source-file grouping and selectable per-check disagreements. Published results and model thresholds remain unchanged.

This list describes implementation scope. The full Python 3.12 suite passed **493/493 tests**, and the strict production frontend build passed. Publication CI remains a separate check. Offline tests exercise contracts, acquisition budgets, source ordering, replay, task portability, preparation gates and private record handling. Live and physical validation status is recorded separately in [feasibility status](docs/FEASIBILITY.md). Passing software tests does not establish model accuracy or physical closed-loop recovery.

## External components

Python and its standard library are external infrastructure. Muse Spark and SAM 3.1 are pretrained Meta models, not models trained or created by this project. The optional official `meta-sam-parser==0.1.0` dependency is distributed under the SAM License; it is installed separately and not vendored in this repository. See [Meta's parser repository](https://github.com/meta-models/meta-sam). Synthetic fixtures are authored test data, not camera recordings or industrial measurements.

External ALOHA coffee recordings were created by the ALOHA authors and distributed through LeRobot; they are not event-created ProofLoop assets. Dataset cards declare MIT, with exact source/license attribution retained in [online source credits](docs/ONLINE-VIDEO-SOURCES.md). Original video, extracted frames, locator outputs and masks stay local. ProofLoop's placement labels are newly authored for selected states, not official ALOHA labels or natural robot-error annotations. The [historical results](artifacts/feasibility/online-video-20261028.md) retain the original SAM failure and guided v1 result of 2/4. The [guided-fix report](artifacts/feasibility/guided-fix-20261028.md) separates v2's 4/4 development replay from its 3/4 fresh-source result with one UNKNOWN; text SAM matched 0/4 and Spark 4/4 on the fresh cohort. This is neither a complete fix nor better accuracy than Spark.

The [review record](docs/AI-SYSTEMS-REVIEW-20261028.md), [recovery guide](docs/RECOVERY.md) and [temporal stress report](artifacts/feasibility/temporal-stress-20261028.md) distinguish implemented consistency checks from unproved physical execution. Stress cases reuse old sources and do not turn the earlier fresh-source guided 3/4 result into a stronger accuracy claim.

The [fixed future-evidence report](artifacts/feasibility/fixed-followup-20261028.md) reuses three previously tested coffee episodes and adds no independent coffee recordings. The [candy transfer report](docs/NEW-TASK-TRANSFER-20261028.md) adds one different manipulation recording under its explicitly MIT-declared dataset card; it is a two-case development check without a FAIL control. These cohorts are not pooled into the earlier fresh-source result. Neither demonstrates a guided accuracy advantage, complete procedural understanding, physical response to a ProofLoop request or validated customer demand.

The new [workflow](docs/WORKFLOW.md), [reproduction guide](docs/REPRODUCING.md) and [demo script](docs/DEMO-SCRIPT.md) identify operator declarations, external assets and the event-created functionality separately.

Codex assisted with code, tests and research. AI assistance does not establish original model capability or replace verification. All human contributors must be added to the hackathon submission under the event rules.

FFmpeg and FFprobe are external command-line media tools, not project-authored decoders. Local decoder integration used version 9.0 binaries from [OSXExperts](https://www.osxexperts.net/), verified against the publisher's binary SHA-256 values. The binaries and generated test videos remain in ignored local storage and are not redistributed here. The extractor also supports tools already available on PATH.

React, Vite, TypeScript and the self-hosted Inter font package are external frontend components installed through npm. Image-generated UI concepts guided the interface before implementation; those private design references are not runtime evidence or app backgrounds. Dependency versions are pinned in the lockfile.
