# Contributions and provenance

Project development began on 26 October 2026 after the hackathon kickoff. The repository contains newly created ProofLoop code and tests. It does not contain a pre-existing proprietary application.

## Event contributions

- Declarative protocol validation and evidence-linked pass/fail/unknown checks.
- Observation freshness, invalidation and completion-attempt boundaries.
- Bounded repair planning with explicit preconditions, effects and action costs.
- Synthetic oracle diagnostic fixtures and transparent result reporting.
- Meta Responses API adapter and credential-safe live access tests.
- Spark protocol draft compiler with local validation and mandatory human review.
- Conservative SAM-mask-to-checker grounding with explicit evidence provenance.
- Four generated scene inputs with prompt and source-hash manifests, labeled synthetic.
- Bounded live SAM scene harness with source-validated resume, private response caches and reviewed public results.
- Live-model feasibility checks, correct mapping of box-local masks and accounting for reported usage on unsuccessful responses.
- Local password form that saves an ignored credential file without putting the key in chat.
- Positive placement predicates, explicit unique-target scene contracts and separate Spark compiler modes for counts, visibility and location.
- Phone-video preflight and timestamp-preserving JPEG extraction, including display rotation and source/hash binding.
- Reviewable recorded-session contracts that bind calibration, model concepts, selected frames and unique-target assumptions.
- Stage-isolated correction traces with movement/occlusion invalidation, a bounded hypothetical repair and fresh post-correction evidence.

This list describes the implementation scope; live and physical validation status is recorded separately in `docs/FEASIBILITY.md`.

## External components

Python and its standard library are external infrastructure. Muse Spark and SAM 3.1 are pretrained Meta models, not models trained or created by this project. The optional official `meta-sam-parser==0.1.0` dependency is distributed under the SAM License; it is installed separately and not vendored in this repository. See [Meta's parser repository](https://github.com/meta-models/meta-sam). Synthetic fixtures are authored test data, not camera recordings or industrial measurements.

Codex assisted with code, tests and research. AI assistance does not establish original model capability or replace verification. All human contributors must be added to the hackathon submission under the event rules.

FFmpeg and FFprobe are external command-line media tools, not project-authored decoders. Local decoder integration used version 9.0 binaries from [OSXExperts](https://www.osxexperts.net/), verified against the publisher's binary SHA-256 values. The binaries and generated test videos remain in ignored local storage and are not redistributed here. The extractor also supports tools already available on PATH.
