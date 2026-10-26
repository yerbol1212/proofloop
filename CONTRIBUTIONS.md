# Contributions and provenance

Project development began on 26 October 2026 after the hackathon kickoff. The repository contains newly created ProofLoop code and tests. It does not contain a pre-existing proprietary application.

## Event contributions

- Declarative protocol validation and evidence-linked pass/fail/unknown checks.
- Observation freshness, invalidation and completion-attempt boundaries.
- Bounded repair planning with explicit preconditions, effects and action costs.
- Synthetic oracle diagnostic fixtures and transparent result reporting.
- Meta Responses API adapter and credential-safe live access tests.

This list describes the implementation scope; live and physical validation status is recorded separately in `docs/FEASIBILITY.md`.

## External components

Python and its standard library are external infrastructure. Muse Spark and SAM 3.1 are pretrained Meta models, not models trained or created by this project. The optional official `meta-sam-parser==0.1.0` dependency is distributed under the SAM License; it is installed separately and not vendored in this repository. See [Meta's parser repository](https://github.com/meta-models/meta-sam). Synthetic fixtures are authored test data, not camera recordings or industrial measurements.

Codex assisted with code, tests and research. AI assistance does not establish original model capability or replace verification. All human contributors must be added to the hackathon submission under the event rules.
