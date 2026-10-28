# Recorded video evaluation

ProofLoop can test scoped visual acceptance checks on existing video, reducing the need for a participant to record every experiment. It does **not** treat a recorded person or robot as someone following a subsequently generated repair. Selected frames establish neither continuous tracking nor completion of hidden steps.

`python -m proofloop.benchmark` provides a frozen-manifest validator, bounded live runner, two-condition comparison and allowlisted summary export. Run commands from the repository root using `.venv/bin/python` when official SAM mask decoding is needed. Raw media, frozen approvals and complete model responses stay under ignored `inputs/local/` and `artifacts/local/` directories.

## Before inference

1. Find a publisher-provided recording and inspect its actual dataset/media license. A publicly viewable video alone is insufficient permission. Keep the publisher, landing page, exact media URL, license name, license URL and attribution in source metadata. Record any crop/resize under each frame's `transform` field.
2. Save the video and two source frames per case under `inputs/local/`. Record SHA-256 digests, original source timestamps and video duration. Use distinct frames at least one source second apart. Reuse of an identical image across cases is rejected; sharing a video source is allowed and creates correlated cases.
3. Specify the requirements, protocol, object-to-SAM concepts and calibrated rectangles before seeing model outputs. Rectangles must be normalized, nonoverlapping and appropriate to both selected views. A fixed rectangle is invalid when camera movement changes what the region means.
4. Review source frames and freeze each benchmark case's per-check and aggregate labels. Explain the visual basis and uncertainty. Mark exploratory material `illustrative` with `annotation: null`; it receives predictions but **no accuracy score**. Do not select successful outputs and relabel them as ground truth.
5. Approve the exact contract hash. Location predicates additionally require an explicit single-target scene assumption. Human and assistant review have different provenance fields. An approval records a caller assertion; its hash is not a trusted timestamp or independent proof that a review happened before all model exposure.

The initial schema supports `location` and `visible` predicates, up to eight objects, 16 checks and 48 cases. It deliberately does not add automatic absence or exact-count claims. The existing evidence store still applies confidence, stability and source-age checks. A zero detection is UNKNOWN; a positively supported unique target in the wrong calibrated region can be FAIL.

## Manifest shape

A file contains this envelope:

```json
{
  "contract": {
    "schema_version": "1.0",
    "evidence_origin": "external_recorded_video",
    "dataset_id": "reviewed-video-experiment",
    "sources": [],
    "requirements": "Approved visual acceptance requirements",
    "protocol": {},
    "concepts": {"glass": "glass"},
    "cases": []
  },
  "contract_hash": "SHA-256 of canonical contract JSON",
  "approval": {
    "approved": true,
    "annotations_frozen_before_inference": true,
    "unique_targets_confirmed": true,
    "origin": "local_operator",
    "human_reviewed_exact_contract": true
  }
}
```

Each source has `id`, `publisher`, `page_url`, `media_url`, `license: {name, url, attribution}`, `file`, `sha256` and integer `duration_ms`. Paths can be project-relative or absolute but must resolve inside `inputs/local/`, including through symlinks. Source digests are checked as well as image digests. Provenance strings do not independently establish a license's validity; review the linked publisher terms.

Each case has:

```json
{
  "id": "before-placement",
  "source_id": "source-video-id",
  "role": "benchmark",
  "regions": {"dispense": [0.39, 0.20, 0.63, 0.44], "staging": [0.68, 0.20, 0.86, 0.44]},
  "frames": [
    {"file": "inputs/local/example/frame-0500.jpg", "sha256": "EXACT_IMAGE_DIGEST", "source_timestamp_ms": 500},
    {"file": "inputs/local/example/frame-1500.jpg", "sha256": "ANOTHER_IMAGE_DIGEST", "source_timestamp_ms": 1500}
  ],
  "evaluation_timestamp_ms": 1500,
  "annotation": {
    "status": "fail",
    "checks": {"glass-dispense": "fail"},
    "reason": "Reviewer explains visible support for the declared label.",
    "origin": "source_frames_reviewed_before_inference"
  }
}
```

These coordinates and labels illustrate the structure; they are not valid ground truth for an arbitrary image. For illustrative cases, replace `role` with `illustrative` and `annotation` with `null`. Annotation aggregate is FAIL if any check fails, PASS if all pass, otherwise UNKNOWN. All check IDs must be labeled.

Construct an approval programmatically after review:

```python
from proofloop.benchmark import approve_manifest, manifest_hash

envelope = approve_manifest(
    contract,
    manifest_hash(contract),
    confirm_unique_targets=True,
    review_origin="assistant_under_user_authorization",
    authorization_reference="Reference to the user's direct instruction authorizing the review",
)
```

Assistant approvals set `human_reviewed_exact_contract` to false; they cannot silently impersonate a human review. The default origin is `local_operator`, for an actual operator's review.

## Validate and run

```sh
.venv/bin/python -m proofloop.benchmark validate artifacts/local/online-review/benchmark-manifest.json
.venv/bin/python -m proofloop.benchmark run artifacts/local/online-review/benchmark-manifest.json --output artifacts/local/online-run-001 --max-requests 6
```

The validator reports required attempts **before spending**. For two cases with one object, SAM needs four image requests and the Spark-only baseline needs two requests, each containing the same two images. An insufficient explicit budget stops before creating a client. The runner disables automatic retries, uses a new empty output directory and rechecks input hashes around inference. It stops on the first invalid or failed response, saves a partial error report and creates no substitute inference. A failed request can still be billed.

Use `--condition sam` or `--condition spark` to run one condition deliberately. Do not mix a condition run from one manifest with results from another. Both conditions use separate client ledgers; their latency and estimated cost appear separately. The shared Meta adapter's estimated rates are not a billing hard stop.

The complete private report includes grounded observations and source-bound results. The CLI prints only the compact status, metrics and estimated costs. Local raw results can contain the exact media encoded in model requests or provider IDs; never commit them.

## Fair comparison and interpretation

- **SAM + deterministic checks:** the original decoded-mask grounder and evidence store evaluate approved requirements; hypothetical repair effects cannot create observations.
- **Spark-only baseline:** Spark receives the same exact image bytes, approved requirements, protocol, calibrated regions and source times. Its prompt explicitly permits UNKNOWN, explains absence/uniqueness limits and requires one result per check. It sees no ground-truth annotations and no SAM output. Responses must bind both source hashes in order and identify every check exactly once.

Metrics include the confusion matrix, incorrect passes among nonpassing labels, correct acceptance among passing labels, UNKNOWN rate, decision coverage, accuracy and balanced accuracy across present classes. Missing predictions count against accuracy and are reported separately; they are never represented as a completed UNKNOWN inference. An always-UNKNOWN system cannot obtain correct acceptance of positive cases. Incorrectly passing an UNKNOWN label also counts as an incorrect pass.

These metrics apply to selected, caller-reviewed cases. A handful of frames from one video is an exploratory integration result, not held-out population accuracy, industrial reliability or proof that ProofLoop outperforms a baseline generally. Report class counts, source counts and selection method alongside every comparison. A larger predeclared evaluation should use multiple sources, diverse failure modes and independently reviewed labels; keep development clips separate from held-out evaluation.

## Optional Spark-guided SAM observation

The original default remains `sam` plus `spark`, so previously reviewed budgets and runs keep their meaning. An optional third condition, `guided`, uses a task-independent Spark locator to propose boxes, followed by real SAM box-prompted mask refinement. The locator receives object concepts and source dimensions, with **no acceptance requirements, calibrated regions, desired placement, manual boxes or ground-truth labels**. The same deterministic grounder then evaluates only actual SAM results against the approved checks.

This is a different observation method, not an independent identity check: **Spark assigns semantic identity; SAM refines the prompted boundary.** A good mask does not independently prove that Spark chose the correct object. The method keeps unmodified SAM confidence and does not invent a semantic-confidence score. Missing, ambiguous or multiple locator candidates produce no SAM request/result for that target and remain UNKNOWN in the evidence bridge.

Enable it explicitly, or compare all three conditions:

```sh
.venv/bin/python -m proofloop.benchmark run artifacts/local/online-review/manifest-approved.json --output artifacts/local/online-benchmark-comparison-001 --max-requests 24 --condition sam --condition spark --condition guided
```

That example budget is correct for **four cases with one object**: eight SAM requests, four Spark-only requests and at most 12 guided requests. Use `validate` and `request_requirements(envelope, ("sam", "spark", "guided"))` to calculate the actual experiment's upper bound. Guided needs at most one Spark request plus two SAM requests per object per case; ambiguous localization can use fewer requests. A condition's ledger reports actual attempts separately from its planned upper bound.

Private `guided/<case-id>.json` files retain locator output, source bindings, omitted targets, exact SAM prompts and returned masks. The final report and Video Lab expose the method's identity limitation. They do not turn locator boxes, hypothetical object positions or missing candidates into fabricated SAM evidence.

The current `normalized_boxes_v2` locator uses Spark's normalized 0–1000 coordinates and strictly converts outer box edges into inclusive SAM source pixels. Identity ambiguity, full visibility and boundary precision are separate. The report exports its `observation_version`; old reports without it retain their original meaning. See the [method and measured revision](GUIDED-PERCEPTION.md).

Declare evaluation exposure before a run using `--evaluation-role development_reprocessing` for already exposed cases or `--evaluation-role new_source_holdback` for reserved sources. The optional role is validated before inference, saved in the report/export, and shown separately in Video Lab. It is a caller declaration, not independent proof of source isolation. A replay of known failures cannot become a new-source score by changing its label. Legacy runs omit this field.

When a first concept fails, preserve that experiment unchanged. If it motivates this new observation method, test the new method under a new explicit condition/run and report the development exposure. Do not relabel failed cases or silently replace the original method's results.

## Reviewable public summary

```sh
.venv/bin/python -m proofloop.benchmark export artifacts/local/online-review/benchmark-manifest.json artifacts/local/online-run-001/report.json --output artifacts/feasibility/online-run-001-summary.json
```

This explicit export selects publisher/license attribution, source digests, case status, numeric metrics, aggregate timing and estimated cost. It omits local media paths, raw frames, masks, provider IDs, authorization text and raw prompts. Review the resulting file before committing. No source media is published automatically.

## Verification

`python -m unittest tests.test_benchmark` uses clearly authored, nonphotographic transport and adapter-shape fixtures. It checks frozen labels, source binding, duplicate/stale/ambiguous observations, calibration, review provenance, budget rejection, equal-image baseline input, all-UNKNOWN behavior, error handling and export filtering. Passing these tests is software verification, not model-performance evidence.
